// layer.cu — hyper-connections, RMSNorm, dense MLP.
//
// The residual stream in this model is FOUR streams of 4096, not one: mHC (Xie et al. 2026)
// replaces the ordinary residual with a learned, Sinkhorn-projected doubly-stochastic mix. Every
// one of the 45 backbone layers runs two of these (attention site and MLP site). The MTP block at
// layer 45 runs none — it has no hc_* tensors at all and uses a plain pre-norm residual.
//
// hc_fn is [24, 16384] bf16 = 786 KB, twice per layer, 0.4% of B_tok — but DeepSeek-V4's engine
// measured the equivalent compose step at 9.4% of decode TIME for 1.2% of the bytes, because a
// warp-per-output-row launch has nowhere near enough memory-level parallelism at batch 1. So this
// uses a full block per output row and folds the norm reduction into the same pass.
#include "layer.h"
#include "gemv.h"
#include "glm5_config.h"
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

namespace glm5 {

static constexpr int HC = HC_MULT;        // 4
static constexpr int MIX = HC_MIX;        // 24
static constexpr int HCD = HC_HCD;        // 16384

size_t hc_workspace_floats() { return MIX; }
size_t dense_mlp_workspace_floats() { return 2 * (size_t)DENSE_INTER; }
size_t dense_mlp_batch_workspace_floats(int B) { return 2 * (size_t)B * DENSE_INTER; }

// mix[m] = sum_j (streams[j] * rsqrt(mean(streams^2) + eps)) * fn[m][j]
//
// The rsqrt is recomputed independently in every block rather than being a separate kernel: each
// block already streams all of `streams` for its own dot product, so accumulating sum(x^2) from
// the same registers costs one extra FMA per element and zero extra loads, and removes a
// cross-block dependency from a phase that is almost entirely latency.
template <int BS>
__global__ void k_hc_mix(float* __restrict__ mix, const float* __restrict__ streams,
                         const __nv_bfloat16* __restrict__ fn, float eps) {
    const int m = blockIdx.x;
    const __nv_bfloat16* fr = fn + (size_t)m * HCD;
    float dot = 0.f, sq = 0.f;
    for (int j = threadIdx.x; j < HCD; j += BS) {
        const float v = streams[j];
        dot += v * __bfloat162float(fr[j]);
        sq  += v * v;
    }
    __shared__ float rd[BS / 32], rs[BS / 32];
    for (int o = 16; o; o >>= 1) { dot += __shfl_down_sync(0xffffffff, dot, o);
                                   sq  += __shfl_down_sync(0xffffffff, sq, o); }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { rd[warp] = dot; rs[warp] = sq; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float d = 0, s = 0;
        for (int w = 0; w < BS / 32; ++w) { d += rd[w]; s += rs[w]; }
        mix[m] = d * rsqrtf(s / (float)HCD + eps);
    }
}

// pre / post / comb, then Sinkhorn-Knopp onto the doubly-stochastic manifold.
// 4x4 and 20 iterations: one thread, because the alternative is 40 block syncs to save ~600 flops.
__global__ void k_hc_gates(float* __restrict__ collapsed, float* __restrict__ post,
                           float* __restrict__ comb, const float* __restrict__ mix,
                           const float* __restrict__ base, const float* __restrict__ scale,
                           const float* __restrict__ streams, int iters, float eps) {
    __shared__ float pre[HC];
    if (threadIdx.x == 0) {
        const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
        for (int h = 0; h < HC; ++h) {
            pre[h]  = 1.f / (1.f + __expf(-(mix[h] * s0 + base[h]))) + eps;
            post[h] = 2.f / (1.f + __expf(-(mix[HC + h] * s1 + base[HC + h])));
        }
        // comb: softmax over the last dim, then + eps, then alternating column/row normalisation.
        float c[HC * HC];
        for (int i = 0; i < HC; ++i) {
            float mx = -1e30f;
            for (int j = 0; j < HC; ++j) {
                c[i * HC + j] = mix[2 * HC + i * HC + j] * s2 + base[2 * HC + i * HC + j];
                mx = fmaxf(mx, c[i * HC + j]);
            }
            float sum = 0.f;
            for (int j = 0; j < HC; ++j) { c[i * HC + j] = __expf(c[i * HC + j] - mx); sum += c[i * HC + j]; }
            for (int j = 0; j < HC; ++j) c[i * HC + j] = c[i * HC + j] / sum + eps;
        }
        // reference does ONE column normalisation, then (iters-1) x (row, column)
        for (int j = 0; j < HC; ++j) {
            float s = eps; for (int i = 0; i < HC; ++i) s += c[i * HC + j];
            for (int i = 0; i < HC; ++i) c[i * HC + j] /= s;
        }
        for (int it = 0; it < iters - 1; ++it) {
            for (int i = 0; i < HC; ++i) {
                float s = eps; for (int j = 0; j < HC; ++j) s += c[i * HC + j];
                for (int j = 0; j < HC; ++j) c[i * HC + j] /= s;
            }
            for (int j = 0; j < HC; ++j) {
                float s = eps; for (int i = 0; i < HC; ++i) s += c[i * HC + j];
                for (int i = 0; i < HC; ++i) c[i * HC + j] /= s;
            }
        }
        for (int i = 0; i < HC * HC; ++i) comb[i] = c[i];
    }
    __syncthreads();
    // collapsed[d] = sum_h pre[h] * streams[h][d]
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < HIDDEN; d += gridDim.x * blockDim.x) {
        float a = 0.f;
        for (int h = 0; h < HC; ++h) a += pre[h] * streams[(size_t)h * HIDDEN + d];
        collapsed[d] = a;
    }
}

// streams[h][d] = post[h]*sub[d] + sum_i comb[i][h]*residual[i][d]     (comb transposed)
__global__ void k_hc_apply(float* __restrict__ streams, const float* __restrict__ residual,
                           const float* __restrict__ sub, const float* __restrict__ post,
                           const float* __restrict__ comb) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= HIDDEN) return;
    const float sd = sub[d];
    float r[HC];
    #pragma unroll
    for (int i = 0; i < HC; ++i) r[i] = residual[(size_t)i * HIDDEN + d];
    #pragma unroll
    for (int h = 0; h < HC; ++h) {
        float a = post[h] * sd;
        #pragma unroll
        for (int i = 0; i < HC; ++i) a += comb[i * HC + h] * r[i];
        streams[(size_t)h * HIDDEN + d] = a;
    }
}

__global__ void k_hc_head_mean(float* __restrict__ out, const float* __restrict__ streams) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= HIDDEN) return;
    float a = 0.f;
    for (int h = 0; h < HC; ++h) a += streams[(size_t)h * HIDDEN + d];
    out[d] = a / (float)HC;
}

// ---- RMSNorm (weighted) ----------------------------------------------------------------------
template <int BS, typename WT>
__global__ void k_rmsnorm(float* __restrict__ y, const float* __restrict__ x,
                          const WT* __restrict__ w, int n, float eps) {
    float sq = 0.f;
    for (int i = threadIdx.x; i < n; i += BS) { const float v = x[i]; sq += v * v; }
    __shared__ float rs[BS / 32];
    for (int o = 16; o; o >>= 1) sq += __shfl_down_sync(0xffffffff, sq, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) rs[warp] = sq;
    __syncthreads();
    __shared__ float inv;
    if (threadIdx.x == 0) {
        float s = 0; for (int k = 0; k < BS / 32; ++k) s += rs[k];
        inv = rsqrtf(s / (float)n + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += BS) {
        float wv;
        if constexpr (sizeof(WT) == 2) wv = __bfloat162float(((const __nv_bfloat16*)w)[i]);
        else wv = ((const float*)w)[i];
        y[i] = x[i] * inv * wv;
    }
}

// ---- dense MLP (layers 0..2), clamped SwiGLU --------------------------------------------------
__global__ void k_swiglu_clamped(float* __restrict__ out, const float* __restrict__ gate,
                                 const float* __restrict__ up, int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = fminf(gate[i], limit);                       // upper clamp only
    const float u = fminf(fmaxf(up[i], -limit), limit);
    out[i] = (g / (1.f + __expf(-g))) * u;
}

// ---- entry points ------------------------------------------------------------------------------
void hc_compose(const float* streams, const HcWeights& W, float* collapsed, float* post,
                float* comb, float* ws, cudaStream_t s) {
    k_hc_mix<256><<<MIX, 256, 0, s>>>(ws, streams, (const __nv_bfloat16*)W.fn, RMS_EPS);
    // ONE block: the gate/Sinkhorn section runs on thread 0, so a multi-block launch would have
    // every block redundantly redo the Sinkhorn and race to write identical post/comb. Identical
    // values make it benign in practice and invisible in a gate, which is exactly why it is not
    // left in. The collapse loop is grid-stride, so one block still covers all 4096 lanes.
    k_hc_gates<<<1, 256, 0, s>>>(collapsed, post, comb, ws, W.base, W.scale, streams,
                                 HC_SINKHORN_ITERS, HC_EPS);
}

void hc_apply(float* streams, const float* residual, const float* sub, const float* post,
              const float* comb, cudaStream_t s) {
    k_hc_apply<<<(HIDDEN + 255) / 256, 256, 0, s>>>(streams, residual, sub, post, comb);
}

void hc_head_mean(const float* streams, float* out, cudaStream_t s) {
    k_hc_head_mean<<<(HIDDEN + 255) / 256, 256, 0, s>>>(out, streams);
}

void rmsnorm(float* y, const float* x, const void* w, int dtype, int n, cudaStream_t s) {
    if (dtype == GEMV_F32) k_rmsnorm<256, float><<<1, 256, 0, s>>>(y, x, (const float*)w, n, RMS_EPS);
    else k_rmsnorm<256, __nv_bfloat16><<<1, 256, 0, s>>>(y, x, (const __nv_bfloat16*)w, n, RMS_EPS);
}

void dense_mlp(const float* x, const DenseMlp& M, float* y, float* ws, cudaStream_t s) {
    float* g = ws;
    float* u = ws + M.inter;
    gemv(g, M.gate, x, M.inter, HIDDEN, M.dtype, s);
    gemv(u, M.up,   x, M.inter, HIDDEN, M.dtype, s);
    k_swiglu_clamped<<<(M.inter + 255) / 256, 256, 0, s>>>(g, g, u, M.inter, SWIGLU_LIMIT);
    gemv(y, M.down, g, HIDDEN, M.inter, M.dtype, s);
}

void dense_mlp_batch(const float* x, const DenseMlp& M, float* y, float* ws, int B, cudaStream_t s) {
    const size_t BI = (size_t)B * M.inter;
    float* g = ws;
    float* u = ws + BI;
    gemm(g, M.gate, x, B, M.inter, HIDDEN, M.dtype, s);
    gemm(u, M.up,   x, B, M.inter, HIDDEN, M.dtype, s);
    // The clamp is asymmetric — gate is min()'d at +10, up is clamped both ways. k_swiglu_clamped
    // holds that; it is elementwise, so one launch covers the whole batch.
    k_swiglu_clamped<<<(int)((BI + 255) / 256), 256, 0, s>>>(g, g, u, (int)BI, SWIGLU_LIMIT);
    gemm(y, M.down, g, B, HIDDEN, M.inter, M.dtype, s);
}

}  // namespace glm5
