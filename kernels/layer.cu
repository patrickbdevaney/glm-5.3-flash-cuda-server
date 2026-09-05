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

// pdot[MIX*SPLIT] + psq[MIX*SPLIT]
static constexpr int HC_SPLIT = 4;
size_t hc_workspace_floats() { return 2 * (size_t)MIX * HC_SPLIT; }
size_t dense_mlp_workspace_floats() { return 2 * (size_t)DENSE_INTER; }
size_t dense_mlp_batch_workspace_floats(int B) { return 2 * (size_t)B * DENSE_INTER; }

// mix[m] = sum_j (streams[j] * rsqrt(mean(streams^2) + eps)) * fn[m][j]
//
// The rsqrt is recomputed independently in every block rather than being a separate kernel: each
// block already streams all of `streams` for its own dot product, so accumulating sum(x^2) from
// the same registers costs one extra FMA per element and zero extra loads, and removes a
// cross-block dependency from a phase that is almost entirely latency.
// mix[m] = sum_j (streams[j] * rsqrt(mean(streams^2) + eps)) * fn[m][j]
//
// The rsqrt is recomputed independently in every block rather than being a separate kernel: each
// block already streams all of `streams` for its own dot product, so accumulating sum(x^2) from
// the same registers costs one extra FMA per element and zero extra loads, and removes a
// cross-block dependency from a phase that is almost entirely latency.
//
// SPLIT: the first version launched MIX=24 blocks, which is roughly one per SM on this box and
// leaves the machine idle in a phase dprof measured at 8% of achievable bandwidth. Each row is
// now split SPLIT ways into partials that k_hc_gates reduces; the reduction is 24x4 floats and
// costs nothing. Loads are bf16x2 / float2 rather than scalar -- fn comes from the checkpoint and
// is only 4-byte aligned, so a bfloat162 (4 B) is the widest legal load, not a float4.
template <int BS, int SPLIT>
__global__ void k_hc_mix(float* __restrict__ pdot, float* __restrict__ psq,
                         const float* __restrict__ streams,
                         const __nv_bfloat16* __restrict__ fn) {
    const int m = blockIdx.x, sp = blockIdx.y;
    const int chunk = HCD / SPLIT, j0 = sp * chunk;
    const __nv_bfloat162* fr = reinterpret_cast<const __nv_bfloat162*>(fn + (size_t)m * HCD + j0);
    const float2* sv = reinterpret_cast<const float2*>(streams + j0);
    float dot = 0.f, sq = 0.f;
    for (int j = threadIdx.x; j < (chunk >> 1); j += BS) {
        const float2 v = sv[j];
        const float2 f = __bfloat1622float2(fr[j]);
        dot = fmaf(v.x, f.x, dot); dot = fmaf(v.y, f.y, dot);
        sq  = fmaf(v.x, v.x, sq);  sq  = fmaf(v.y, v.y, sq);
    }
    __shared__ float rd[BS / 32], rs[BS / 32];
    for (int o = 16; o; o >>= 1) { dot += __shfl_down_sync(0xffffffff, dot, o);
                                   sq  += __shfl_down_sync(0xffffffff, sq, o); }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { rd[warp] = dot; rs[warp] = sq; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float d = 0, q = 0;
        for (int w = 0; w < BS / 32; ++w) { d += rd[w]; q += rs[w]; }
        pdot[m * SPLIT + sp] = d;
        psq [m * SPLIT + sp] = q;
    }
}

// pre / post / comb, then Sinkhorn-Knopp onto the doubly-stochastic manifold.
//
// The 4x4 Sinkhorn runs on ONE WARP, sixteen lanes holding one matrix element each. The first
// version ran all 20 iterations on thread 0 of a single block -- 20 x (4 row sums + 4 column
// sums) walked serially by one lane while 255 others waited, and it dominated a phase that moves
// only 35 MB per token. Row i lives in lanes 4i..4i+3 and column j in lanes j, j+4, j+8, j+12, so
// a row reduction is __shfl_xor over bits 0-1 and a column reduction over bits 2-3. Lanes 16-31
// take part in the shuffles (they must, for convergence) and their values are discarded; no mask
// smaller than the full warp is correct here.
//
// Only block 0 writes post/comb. Every block computes `pre` for itself -- four sigmoids -- which
// is what lets the collapse loop run on a real grid instead of the single block the serial
// Sinkhorn used to force. Two blocks writing identical values to post/comb would be benign in
// practice and invisible in a gate, which is exactly why it is not left in.
__global__ void k_hc_gates(float* __restrict__ collapsed, float* __restrict__ post,
                           float* __restrict__ comb, const float* __restrict__ pdot,
                           const float* __restrict__ psq,
                           const float* __restrict__ base, const float* __restrict__ scale,
                           const float* __restrict__ streams, int split, int iters, float eps) {
    __shared__ float mix[MIX];
    __shared__ float pre[HC];
    if (threadIdx.x < MIX) {
        float d = 0.f, q = 0.f;
        for (int sp = 0; sp < split; ++sp) { d += pdot[threadIdx.x * split + sp];
                                             q += psq [threadIdx.x * split + sp]; }
        mix[threadIdx.x] = d * rsqrtf(q / (float)HCD + RMS_EPS);
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        const int t = threadIdx.x;
        const unsigned FULL = 0xffffffffu;
        const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
        if (t < HC) {
            pre[t] = 1.f / (1.f + __expf(-(mix[t] * s0 + base[t]))) + eps;
            if (blockIdx.x == 0)
                post[t] = 2.f / (1.f + __expf(-(mix[HC + t] * s1 + base[HC + t])));
        }
        // comb: softmax over the last dim, then + eps, then alternating column/row normalisation.
        float c = (t < HC * HC) ? (mix[2 * HC + t] * s2 + base[2 * HC + t]) : -1e30f;
        float mx = c;
        mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, 2));
        float e = __expf(c - mx);
        float sum = e;
        sum += __shfl_xor_sync(FULL, sum, 1);
        sum += __shfl_xor_sync(FULL, sum, 2);
        c = e / sum + eps;
        // reference does ONE column normalisation, then (iters-1) x (row, column)
        {
            float q = c; q += __shfl_xor_sync(FULL, q, 4); q += __shfl_xor_sync(FULL, q, 8);
            c /= (q + eps);
        }
        for (int it = 0; it < iters - 1; ++it) {
            float r = c; r += __shfl_xor_sync(FULL, r, 1); r += __shfl_xor_sync(FULL, r, 2);
            c /= (r + eps);
            float q = c; q += __shfl_xor_sync(FULL, q, 4); q += __shfl_xor_sync(FULL, q, 8);
            c /= (q + eps);
        }
        if (blockIdx.x == 0 && t < HC * HC) comb[t] = c;
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
    k_hc_mix<128, HC_SPLIT><<<dim3(MIX, HC_SPLIT), 128, 0, s>>>(
        ws, ws + MIX * HC_SPLIT, streams, (const __nv_bfloat16*)W.fn);
    // 16 blocks: block 0 owns post/comb, every block computes `pre` for its own slice of the
    // collapse. See the note on the kernel for why this is not a race.
    k_hc_gates<<<16, 256, 0, s>>>(collapsed, post, comb, ws, ws + MIX * HC_SPLIT,
                                  W.base, W.scale, streams, HC_SPLIT, HC_SINKHORN_ITERS, HC_EPS);
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
