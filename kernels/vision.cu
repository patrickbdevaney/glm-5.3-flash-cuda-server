// vision.cu — the 24-block vision tower. See include/vision.h for why it exists.
//
// Traps this code exists to get right, all of them differences from the language model:
//   * the block MLPs use a CLAMPED SwiGLU (gate clamped above at 10, up clamped both ways);
//   * the merger uses a LAYERNORM WITH BIAS, not RMSNorm, and a GELU, not SiLU;
//   * q_norm/k_norm are RMSNorm over head_dim=64, applied per head BEFORE rope;
//   * attention is BIDIRECTIONAL -- no causal mask anywhere in the tower;
//   * rope is rotate_half over the full head_dim, not the interleaved convention.
#include "vision.h"
#include "layer.h"
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

namespace glm5 {

#define CU(x) do { cudaError_t e_=(x); if(e_){ fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(e_)); abort(); } } while(0)
#define KCHK(n) do { cudaError_t e_=cudaGetLastError(); if(e_){ \
    fprintf(stderr,"cuda launch %s (%s:%d): %s\n", n, __FILE__, __LINE__, cudaGetErrorString(e_)); \
    abort(); } } while(0)

static __device__ __forceinline__ float bf(const void* p, size_t i) {
    return __bfloat162float(((const __nv_bfloat16*)p)[i]);
}

// y[m][n] += b[n]
__global__ void k_add_bias(float* __restrict__ y, const void* __restrict__ b, int M, int N) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)M * N) return;
    y[i] += bf(b, i % N);
}

// RMSNorm each of M rows of width N. One block per row.
template <int BS>
__global__ void k_rmsnorm_rows(float* __restrict__ y, const float* __restrict__ x,
                               const void* __restrict__ w, int N, float eps) {
    const int r = blockIdx.x;
    const float* xr = x + (size_t)r * N;
    float* yr = y + (size_t)r * N;
    float s = 0.f;
    for (int i = threadIdx.x; i < N; i += BS) { const float v = xr[i]; s += v * v; }
    __shared__ float red[BS / 32];
    for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    __shared__ float inv;
    if (threadIdx.x == 0) { float t = 0; for (int k = 0; k < BS / 32; ++k) t += red[k];
                            inv = rsqrtf(t / N + eps); }
    __syncthreads();
    for (int i = threadIdx.x; i < N; i += BS) yr[i] = xr[i] * inv * bf(w, i);
}

// RMSNorm over head_dim, for every (row, head). q is [seq, heads, hd] contiguous.
//
// LAUNCHED WITH EXACTLY ONE WARP. The reduction is __shfl_down_sync, which is warp-local, so a
// 64-thread block would silently drop half the sum-of-squares and rescale every head by ~sqrt(2).
// That is what it did: the tower came back at cos 0.690 -- wrong, but plausible enough to look
// like a numerics issue rather than a bug.
__global__ void k_rmsnorm_heads(float* __restrict__ q, const void* __restrict__ w, int hd) {
    const int rh = blockIdx.x;                 // one block per (row, head)
    float* v = q + (size_t)rh * hd;
    float s = 0.f;
    for (int i = threadIdx.x; i < hd; i += 32) s += v[i] * v[i];
    for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    const float inv = rsqrtf(__shfl_sync(0xffffffff, s, 0) / hd + VIS_EPS);
    for (int i = threadIdx.x; i < 32 * ((hd + 31) / 32); i += 32)
        if (i < hd) v[i] = v[i] * inv * bf(w, i);
}

// rotate_half rope over the full head_dim: y = x*cos + rotate_half(x)*sin, where rotate_half
// maps (x1, x2) -> (-x2, x1) on the two halves. NOT the interleaved convention.
__global__ void k_rope_vision(float* __restrict__ q, float* __restrict__ k,
                              const float* __restrict__ cos, const float* __restrict__ sin,
                              int seq, int heads, int hd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = hd >> 1;
    if (i >= seq * heads * half) return;
    const int j = i % half;                 // index within the first half
    const int h = (i / half) % heads;
    const int r = i / (half * heads);
    const size_t base = ((size_t)r * heads + h) * hd;
    const float c1 = cos[(size_t)r * hd + j],       s1 = sin[(size_t)r * hd + j];
    const float c2 = cos[(size_t)r * hd + j + half], s2 = sin[(size_t)r * hd + j + half];
    const float q1 = q[base + j], q2 = q[base + j + half];
    q[base + j]        = q1 * c1 - q2 * s1;
    q[base + j + half] = q2 * c2 + q1 * s2;
    const float k1 = k[base + j], k2 = k[base + j + half];
    k[base + j]        = k1 * c1 - k2 * s1;
    k[base + j + half] = k2 * c2 + k1 * s2;
}

// Bidirectional attention -- no causal mask anywhere in the tower. One block per (query, head),
// with the whole score row held in shared memory: at 1024 patches that is 4 KB, so scores never
// round-trip to global and the three phases (dot, softmax, context) fuse into one pass.
template <int BS>
__global__ void k_vis_attn(float* __restrict__ out, const float* __restrict__ q,
                           const float* __restrict__ k, const float* __restrict__ v,
                           int seq, int heads, int hd, float scale) {
    const int i = blockIdx.x, h = blockIdx.y;
    extern __shared__ float sh[];
    float* qv = sh;              // [hd]
    float* sc = sh + hd;         // [seq]
    for (int d = threadIdx.x; d < hd; d += BS) qv[d] = q[((size_t)i * heads + h) * hd + d];
    __syncthreads();

    __shared__ float red[BS / 32];
    float m = -1e30f;
    for (int j = threadIdx.x; j < seq; j += BS) {
        const float* kv = k + ((size_t)j * heads + h) * hd;
        float d = 0.f;
        for (int t = 0; t < hd; ++t) d = fmaf(qv[t], kv[t], d);
        d *= scale; sc[j] = d; m = fmaxf(m, d);
    }
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    __shared__ float mx;
    if (threadIdx.x == 0) { float t = -1e30f; for (int c = 0; c < BS / 32; ++c) t = fmaxf(t, red[c]); mx = t; }
    __syncthreads();

    float sum = 0.f;
    for (int j = threadIdx.x; j < seq; j += BS) { const float e = __expf(sc[j] - mx); sc[j] = e; sum += e; }
    for (int o = 16; o; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    __shared__ float tot;
    if (threadIdx.x == 0) { float t = 0; for (int c = 0; c < BS / 32; ++c) t += red[c]; tot = t; }
    __syncthreads();

    for (int d = threadIdx.x; d < hd; d += BS) {
        float a = 0.f;
        for (int j = 0; j < seq; ++j) a = fmaf(sc[j], v[((size_t)j * heads + h) * hd + d], a);
        out[((size_t)i * heads + h) * hd + d] = a / tot;
    }
}

// [seq, heads, hd] -> [seq, heads*hd] is a no-op in memory; this transposes q/k/v out of the
// packed qkv layout [seq, 3, heads, hd] instead.
__global__ void k_split_qkv(float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
                            const float* __restrict__ qkv, int seq, int heads, int hd) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t)seq * heads * hd;
    if (i >= n) return;
    const int d = i % hd, h = (i / hd) % heads, r = i / ((size_t)hd * heads);
    const size_t src = ((size_t)r * 3 * heads + h) * hd + d;
    q[i] = qkv[src];
    k[i] = qkv[src + (size_t)heads * hd];
    v[i] = qkv[src + 2 * (size_t)heads * hd];
}

// SwiGLU with the tower's clamps: gate clamped ABOVE only, up clamped both ways.
__global__ void k_swiglu_clamp(float* __restrict__ out, const float* __restrict__ gate,
                               const float* __restrict__ up, float lim, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = fminf(gate[i], lim);
    const float u = fmaxf(fminf(up[i], lim), -lim);
    out[i] = (g / (1.f + __expf(-g))) * u;          // SiLU(gate) * up
}

// LayerNorm WITH BIAS (the merger's post_projection_norm), then GELU. Not RMSNorm.
template <int BS>
__global__ void k_layernorm_gelu(float* __restrict__ y, const float* __restrict__ x,
                                 const void* __restrict__ w, const void* __restrict__ b,
                                 int N, float eps) {
    const int r = blockIdx.x;
    const float* xr = x + (size_t)r * N;
    float* yr = y + (size_t)r * N;
    float s = 0.f;
    for (int i = threadIdx.x; i < N; i += BS) s += xr[i];
    __shared__ float red[BS / 32];
    for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    __shared__ float mean;
    if (threadIdx.x == 0) { float t = 0; for (int k = 0; k < BS / 32; ++k) t += red[k]; mean = t / N; }
    __syncthreads();
    float vsum = 0.f;
    for (int i = threadIdx.x; i < N; i += BS) { const float d = xr[i] - mean; vsum += d * d; }
    for (int o = 16; o; o >>= 1) vsum += __shfl_down_sync(0xffffffff, vsum, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = vsum;
    __syncthreads();
    __shared__ float inv;
    if (threadIdx.x == 0) { float t = 0; for (int k = 0; k < BS / 32; ++k) t += red[k];
                            inv = rsqrtf(t / N + eps); }
    __syncthreads();
    for (int i = threadIdx.x; i < N; i += BS) {
        const float h = (xr[i] - mean) * inv * bf(w, i) + bf(b, i);
        // exact GELU, matching nn.GELU()'s default (erf form, not tanh)
        yr[i] = 0.5f * h * (1.f + erff(h * 0.70710678118654752f));
    }
}

// Gather the 2x2 spatial block into one row so the Conv2d downsample becomes a gemm.
// out[n][c*4 + i*2 + j] = in[n*4 + i*2 + j][c], which is exactly the view/permute pair in
// Glm5NextVisionModel.forward followed by a kernel==stride conv.
__global__ void k_downsample_gather(float* __restrict__ out, const float* __restrict__ in,
                                    int n_out, int H) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)n_out * H * 4) return;
    const int p = idx % 4;                       // i*2 + j
    const int c = (idx / 4) % H;
    const int n = idx / (4 * (size_t)H);
    out[idx] = in[((size_t)n * 4 + p) * H + c];
}

__global__ void k_add(float* __restrict__ a, const float* __restrict__ b, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

size_t vision_workspace_floats(int n_patch) {
    const size_t S = n_patch;
    return S * VIS_HIDDEN * 4                    // h, resid, norm scratch, attn out
         + S * 3 * VIS_HIDDEN                    // qkv
         + S * VIS_HIDDEN * 3                    // q, k, v
         + S * VIS_INTER * 2                     // gate, up
         + (S / 4) * VIS_OUT_HIDDEN * 3          // downsample gather + merger scratch
         + (S / 4) * VIS_PROJ_INTER * 2;         // merger gate/up
}

// GLM5_VIS_STOP=k runs only the first k blocks and leaves the result in ws[0..S*HIDDEN), so the
// gate can bisect the tower instead of reporting one number for 24 blocks.
static int vis_stop() { const char* e = getenv("GLM5_VIS_STOP"); return e ? atoi(e) : -1; }

void vision_forward(const float* x, const float* cos, const float* sin,
                    const VisionWeights& W, int S, float* out, float* ws, cudaStream_t s) {
    const int STOP = vis_stop();
    if (S % (VIS_MERGE * VIS_MERGE)) {
        fprintf(stderr, "vision_forward: n_patch %d is not a multiple of %d\n",
                S, VIS_MERGE * VIS_MERGE); abort();
    }
    float* h     = ws;                              // [S, 1024]
    float* resid = h     + (size_t)S * VIS_HIDDEN;
    float* nrm   = resid + (size_t)S * VIS_HIDDEN;
    float* attn  = nrm   + (size_t)S * VIS_HIDDEN;
    float* qkv   = attn  + (size_t)S * VIS_HIDDEN;
    float* q     = qkv   + (size_t)S * 3 * VIS_HIDDEN;
    float* k     = q     + (size_t)S * VIS_HIDDEN;
    float* v     = k     + (size_t)S * VIS_HIDDEN;
    float* gate  = v     + (size_t)S * VIS_HIDDEN;
    float* up    = gate  + (size_t)S * VIS_INTER;
    float* dsx   = up    + (size_t)S * VIS_INTER;   // [S/4, 4096]
    const int NO = S / (VIS_MERGE * VIS_MERGE);
    float* m1    = dsx   + (size_t)NO * VIS_OUT_HIDDEN;
    float* m2    = m1    + (size_t)NO * VIS_OUT_HIDDEN;
    float* mg    = m2    + (size_t)NO * VIS_OUT_HIDDEN;
    float* mu    = mg    + (size_t)NO * VIS_PROJ_INTER;

    const int TPB = 256;
    auto grid = [&](size_t n) { return (int)((n + TPB - 1) / TPB); };

    // patch_embed: kernel == stride, so the Conv3d is a plain linear over the flattened patch.
    gemm(h, W.patch_embed, x, S, VIS_HIDDEN, VIS_IN_DIM, W.dtype, s);
    k_add_bias<<<grid((size_t)S * VIS_HIDDEN), TPB, 0, s>>>(h, W.patch_embed_b, S, VIS_HIDDEN);
    KCHK("k_add_bias");

    if (STOP == 0) return;
    const float scale = rsqrtf((float)VIS_HEAD_DIM);
    for (int b = 0; b < VIS_DEPTH; ++b) {
        const VisionBlockWeights& B = W.blocks[b];
        CU(cudaMemcpyAsync(resid, h, (size_t)S * VIS_HIDDEN * 4, cudaMemcpyDeviceToDevice, s));

        k_rmsnorm_rows<256><<<S, 256, 0, s>>>(nrm, h, B.norm1, VIS_HIDDEN, VIS_EPS);
        KCHK("k_rmsnorm_rows");
        gemm(qkv, B.qkv, nrm, S, 3 * VIS_HIDDEN, VIS_HIDDEN, W.dtype, s);
        k_add_bias<<<grid((size_t)S * 3 * VIS_HIDDEN), TPB, 0, s>>>(qkv, B.qkv_b, S, 3 * VIS_HIDDEN);
        k_split_qkv<<<grid((size_t)S * VIS_HIDDEN), TPB, 0, s>>>(q, k, v, qkv, S, VIS_HEADS, VIS_HEAD_DIM);
        KCHK("k_split_qkv");
        k_rmsnorm_heads<<<S * VIS_HEADS, 32, 0, s>>>(q, B.q_norm, VIS_HEAD_DIM);
        k_rmsnorm_heads<<<S * VIS_HEADS, 32, 0, s>>>(k, B.k_norm, VIS_HEAD_DIM);
        KCHK("k_rmsnorm_heads");
        k_rope_vision<<<grid((size_t)S * VIS_HEADS * (VIS_HEAD_DIM / 2)), TPB, 0, s>>>(
            q, k, cos, sin, S, VIS_HEADS, VIS_HEAD_DIM);
        KCHK("k_rope_vision");
        k_vis_attn<256><<<dim3(S, VIS_HEADS), 256, (VIS_HEAD_DIM + S) * sizeof(float), s>>>(
            attn, q, k, v, S, VIS_HEADS, VIS_HEAD_DIM, scale);
        KCHK("k_vis_attn");
        gemm(h, B.proj, attn, S, VIS_HIDDEN, VIS_HIDDEN, W.dtype, s);
        k_add_bias<<<grid((size_t)S * VIS_HIDDEN), TPB, 0, s>>>(h, B.proj_b, S, VIS_HIDDEN);
        k_add<<<grid((size_t)S * VIS_HIDDEN), TPB, 0, s>>>(h, resid, (size_t)S * VIS_HIDDEN);
        KCHK("k_add");

        CU(cudaMemcpyAsync(resid, h, (size_t)S * VIS_HIDDEN * 4, cudaMemcpyDeviceToDevice, s));
        k_rmsnorm_rows<256><<<S, 256, 0, s>>>(nrm, h, B.norm2, VIS_HIDDEN, VIS_EPS);
        gemm(gate, B.gate, nrm, S, VIS_INTER, VIS_HIDDEN, W.dtype, s);
        k_add_bias<<<grid((size_t)S * VIS_INTER), TPB, 0, s>>>(gate, B.gate_b, S, VIS_INTER);
        gemm(up, B.up, nrm, S, VIS_INTER, VIS_HIDDEN, W.dtype, s);
        k_add_bias<<<grid((size_t)S * VIS_INTER), TPB, 0, s>>>(up, B.up_b, S, VIS_INTER);
        k_swiglu_clamp<<<grid((size_t)S * VIS_INTER), TPB, 0, s>>>(
            gate, gate, up, VIS_SWIGLU_LIMIT, (size_t)S * VIS_INTER);
        KCHK("k_swiglu_clamp");
        gemm(h, B.down, gate, S, VIS_HIDDEN, VIS_INTER, W.dtype, s);
        k_add_bias<<<grid((size_t)S * VIS_HIDDEN), TPB, 0, s>>>(h, B.down_b, S, VIS_HIDDEN);
        k_add<<<grid((size_t)S * VIS_HIDDEN), TPB, 0, s>>>(h, resid, (size_t)S * VIS_HIDDEN);
        if (STOP == b + 1) return;
    }

    k_rmsnorm_rows<256><<<S, 256, 0, s>>>(nrm, h, W.post_layernorm, VIS_HIDDEN, VIS_EPS);
    KCHK("post_layernorm");

    // 2x2 spatial downsample: gather then gemm, which is what a kernel==stride Conv2d is.
    k_downsample_gather<<<grid((size_t)NO * VIS_HIDDEN * 4), TPB, 0, s>>>(dsx, nrm, NO, VIS_HIDDEN);
    KCHK("k_downsample_gather");
    gemm(m1, W.downsample, dsx, NO, VIS_OUT_HIDDEN, VIS_HIDDEN * 4, W.dtype, s);
    k_add_bias<<<grid((size_t)NO * VIS_OUT_HIDDEN), TPB, 0, s>>>(m1, W.downsample_b, NO, VIS_OUT_HIDDEN);

    // merger
    gemm(m2, W.mg_proj, m1, NO, VIS_OUT_HIDDEN, VIS_OUT_HIDDEN, W.dtype, s);
    k_layernorm_gelu<256><<<NO, 256, 0, s>>>(m2, m2, W.mg_ln_w, W.mg_ln_b, VIS_OUT_HIDDEN, 1e-5f);
    KCHK("k_layernorm_gelu");
    gemm(mg, W.mg_gate, m2, NO, VIS_PROJ_INTER, VIS_OUT_HIDDEN, W.dtype, s);
    gemm(mu, W.mg_up,   m2, NO, VIS_PROJ_INTER, VIS_OUT_HIDDEN, W.dtype, s);
    k_swiglu_clamp<<<grid((size_t)NO * VIS_PROJ_INTER), TPB, 0, s>>>(
        mg, mg, mu, VIS_SWIGLU_LIMIT, (size_t)NO * VIS_PROJ_INTER);
    gemm(out, W.mg_down, mg, NO, VIS_OUT_HIDDEN, VIS_PROJ_INTER, W.dtype, s);
}


// Position ids for a (t, h, w) patch grid, matching transformers' get_vision_position_ids.
//
// NOT raster order. Patches are visited in spatial_merge_size x spatial_merge_size blocks, because
// the downsample later folds each 2x2 neighbourhood into one output row -- so the rope ids have to
// agree with the order the merge expects, or every image is spatially scrambled in a way that
// still produces fluent-looking captions.
void vision_position_ids(int t, int h, int w, int m, int* pos) {
    int o = 0;
    for (int f = 0; f < t; ++f)
        for (int hb = 0; hb < h / m; ++hb)
            for (int wb = 0; wb < w / m; ++wb)
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < m; ++j) {
                        pos[2 * o + 0] = hb * m + i;
                        pos[2 * o + 1] = wb * m + j;
                        ++o;
                    }
}

// cos/sin tables. rotary dim is head_dim/2 = 32 split across the two axes, so inv_freq has 16
// entries and each axis contributes 16 -> concat gives 32 -> cat(rot, rot) gives head_dim.
void vision_rope_tables(const int* pos, int n_patch, float* cos_t, float* sin_t) {
    const int half = VIS_HEAD_DIM / 2;          // 32
    const int per  = half / 2;                  // 16 frequencies per axis
    for (int r = 0; r < n_patch; ++r) {
        for (int a = 0; a < 2; ++a)
            for (int i = 0; i < per; ++i) {
                const float inv = 1.0f / powf(10000.0f, (float)(2 * i) / (float)half);
                const float ang = (float)pos[2 * r + a] * inv;
                const int idx = a * per + i;
                cos_t[(size_t)r * VIS_HEAD_DIM + idx] = cosf(ang);
                sin_t[(size_t)r * VIS_HEAD_DIM + idx] = sinf(ang);
                cos_t[(size_t)r * VIS_HEAD_DIM + idx + half] = cosf(ang);
                sin_t[(size_t)r * VIS_HEAD_DIM + idx + half] = sinf(ang);
            }
    }
}

}  // namespace glm5
