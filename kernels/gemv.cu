// gemv.cu — batch-1 matrix-vector products, the single most bandwidth-critical shape in decode.
//
// y[n] = sum_k W[n, k] * x[k],  W row-major [N, K].
//
// At decode this kernel IS the model: 76% of B_tok is dense weights streamed through here exactly
// once per token (ROOFLINE.md §1). The only thing that matters is reading W at close to peak
// bandwidth; the arithmetic is one FMA per element and never binds.
//
// Layout: one BLOCK per output row, blockDim.x threads striding K. Rows are contiguous in memory
// and every thread in the block reads a different 16-byte chunk of the same row, so the row is
// consumed by fully-coalesced 128-byte transactions. x[] is tiny (16 KB at HIDDEN=4096) and hits
// L2 on every row after the first.
//
// ALIGNMENT. The 16-byte vector path is only legal when the weight pointer is 16-byte aligned,
// and CHECKPOINT TENSORS ARE NOT: safetensors aligns to 4 bytes, so roughly half the tensors in
// this model sit at offset 4 mod 8 and a float4 load on them faults with "misaligned address".
// Buffers from cudaMalloc are always fine, which is exactly why this survives a synthetic gate and
// dies on real weights. gemv() therefore measures alignment at launch and dispatches to a scalar
// variant when the vector path would be illegal - never assumes.
#include "gemv.h"
#include <cuda_bf16.h>
#include <cstdint>

namespace glm5 {

template <int BS>
__global__ void k_gemv_f32(float* __restrict__ y, const float* __restrict__ W,
                           const float* __restrict__ x, int N, int K) {
    const int n = blockIdx.x;
    if (n >= N) return;
    const float4* Wr = reinterpret_cast<const float4*>(W + (size_t)n * K);
    const float4* xv = reinterpret_cast<const float4*>(x);
    const int K4 = K >> 2;
    float acc = 0.f;
    for (int i = threadIdx.x; i < K4; i += BS) {
        float4 w = Wr[i], xx = xv[i];
        acc += w.x * xx.x + w.y * xx.y + w.z * xx.z + w.w * xx.w;
    }
    // tail (K is a multiple of 4 for every shape in this model, but do not assume it)
    for (int i = (K4 << 2) + threadIdx.x; i < K; i += BS) acc += W[(size_t)n * K + i] * x[i];

    __shared__ float red[BS / 32];
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (lane < BS / 32) ? red[lane] : 0.f;
        for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if (lane == 0) y[n] = acc;
    }
}

template <int BS>
__global__ void k_gemv_bf16(float* __restrict__ y, const __nv_bfloat16* __restrict__ W,
                            const float* __restrict__ x, int N, int K) {
    const int n = blockIdx.x;
    if (n >= N) return;
    // 8 bf16 = 16 bytes: the widest load the ISA gives us, and what makes this kernel
    // bandwidth-bound rather than issue-bound.
    const float4* Wr = reinterpret_cast<const float4*>(W + (size_t)n * K);
    const int K8 = K >> 3;
    float acc = 0.f;
    for (int i = threadIdx.x; i < K8; i += BS) {
        float4 raw = Wr[i];
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
        const float* xp = x + (i << 3);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __bfloat1622float2(h[j]);
            acc += f.x * xp[j * 2] + f.y * xp[j * 2 + 1];
        }
    }
    for (int i = (K8 << 3) + threadIdx.x; i < K; i += BS)
        acc += __bfloat162float(W[(size_t)n * K + i]) * x[i];

    __shared__ float red[BS / 32];
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (lane < BS / 32) ? red[lane] : 0.f;
        for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if (lane == 0) y[n] = acc;
    }
}

// Scalar variants: legal for any alignment the element type itself allows.
template <int BS>
__global__ void k_gemv_f32_scalar(float* __restrict__ y, const float* __restrict__ W,
                                  const float* __restrict__ x, int N, int K) {
    const int n = blockIdx.x;
    const float* Wr = W + (size_t)n * K;
    float acc = 0.f;
    for (int i = threadIdx.x; i < K; i += BS) acc += Wr[i] * x[i];
    __shared__ float red[BS / 32];
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (lane < BS / 32) ? red[lane] : 0.f;
        for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if (lane == 0) y[n] = acc;
    }
}

template <int BS>
__global__ void k_gemv_bf16_scalar(float* __restrict__ y, const __nv_bfloat16* __restrict__ W,
                                   const float* __restrict__ x, int N, int K) {
    const int n = blockIdx.x;
    const __nv_bfloat16* Wr = W + (size_t)n * K;
    float acc = 0.f;
    for (int i = threadIdx.x; i < K; i += BS) acc += __bfloat162float(Wr[i]) * x[i];
    __shared__ float red[BS / 32];
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (lane < BS / 32) ? red[lane] : 0.f;
        for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if (lane == 0) y[n] = acc;
    }
}

// True only when EVERY row start is 16-byte aligned: the base must be, and so must the row stride,
// or row 1 lands misaligned even though row 0 is fine.
static inline bool vec16_ok(const void* p, int K, int elem_bytes) {
    return (((uintptr_t)p & 15) == 0) && ((((size_t)K * elem_bytes) & 15) == 0);
}

void gemv(float* y, const void* W, const float* x, int N, int K, int dtype, cudaStream_t s) {
    constexpr int BS = 256;
    if (dtype == GEMV_F32) {
        if (vec16_ok(W, K, 4)) k_gemv_f32<BS><<<N, BS, 0, s>>>(y, (const float*)W, x, N, K);
        else                   k_gemv_f32_scalar<BS><<<N, BS, 0, s>>>(y, (const float*)W, x, N, K);
    } else {
        if (vec16_ok(W, K, 2)) k_gemv_bf16<BS><<<N, BS, 0, s>>>(y, (const __nv_bfloat16*)W, x, N, K);
        else                   k_gemv_bf16_scalar<BS><<<N, BS, 0, s>>>(y, (const __nv_bfloat16*)W, x, N, K);
    }
}

}  // namespace glm5

namespace glm5 {
__global__ void k_f32_to_bf16(__nv_bfloat16* d, const float* s, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = __float2bfloat16(s[i]);
}
void f32_to_bf16(void* dst, const float* src, size_t n, cudaStream_t s) {
    k_f32_to_bf16<<<(unsigned)((n + 255) / 256), 256, 0, s>>>((__nv_bfloat16*)dst, src, n);
}
}

namespace glm5 {
__global__ void k_bf16_to_f32(float* d, const __nv_bfloat16* s, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = __bfloat162float(s[i]);
}
void f32_from_bf16_dev(float* dst, const void* src, size_t n, cudaStream_t s) {
    k_bf16_to_f32<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(dst, (const __nv_bfloat16*)src, n);
}
}

// ---- batched: W read once for M rows of x ------------------------------------------------------
//
// Same loop order and same reduction tree as the batch-1 kernels above, so M=1 is bit-identical to
// gemv. That is a requirement, not a nicety: speculative verification compares a batched forward
// against what the sequential path would have produced, and "close" is not a comparison.
//
// MB is a COMPILE-TIME batch. A runtime-bounded loop over the accumulator array spills it to local
// memory and the kernel stops being bandwidth-bound, which is the entire reason this exists.
namespace glm5 {

template <int BS, int MB>
__global__ void k_gemm_f32(float* __restrict__ y, const float* __restrict__ W,
                           const float* __restrict__ x, int N, int K) {
    const int n = blockIdx.x;
    if (n >= N) return;
    const float4* Wr = reinterpret_cast<const float4*>(W + (size_t)n * K);
    const int K4 = K >> 2;
    float acc[MB];
    #pragma unroll
    for (int m = 0; m < MB; ++m) acc[m] = 0.f;
    for (int i = threadIdx.x; i < K4; i += BS) {
        const float4 w = Wr[i];
        #pragma unroll
        for (int m = 0; m < MB; ++m) {
            const float4 xx = reinterpret_cast<const float4*>(x + (size_t)m * K)[i];
            acc[m] += w.x * xx.x + w.y * xx.y + w.z * xx.z + w.w * xx.w;
        }
    }
    for (int i = (K4 << 2) + threadIdx.x; i < K; i += BS) {
        const float w = W[(size_t)n * K + i];
        #pragma unroll
        for (int m = 0; m < MB; ++m) acc[m] += w * x[(size_t)m * K + i];
    }
    __shared__ float red[BS / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int m = 0; m < MB; ++m) {
        float a = acc[m];
        for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
        if (lane == 0) red[warp] = a;
        __syncthreads();
        if (warp == 0) {
            a = (lane < BS / 32) ? red[lane] : 0.f;
            for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
            if (lane == 0) y[(size_t)m * N + n] = a;
        }
        __syncthreads();                      // red[] is reused by the next m
    }
}

template <int BS, int MB>
__global__ void k_gemm_bf16(float* __restrict__ y, const __nv_bfloat16* __restrict__ W,
                            const float* __restrict__ x, int N, int K) {
    const int n = blockIdx.x;
    if (n >= N) return;
    const float4* Wr = reinterpret_cast<const float4*>(W + (size_t)n * K);
    const int K8 = K >> 3;
    float acc[MB];
    #pragma unroll
    for (int m = 0; m < MB; ++m) acc[m] = 0.f;
    for (int i = threadIdx.x; i < K8; i += BS) {
        float4 raw = Wr[i];
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
        float wf[8];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(h[j]);
            wf[j * 2] = f.x; wf[j * 2 + 1] = f.y;
        }
        #pragma unroll
        for (int m = 0; m < MB; ++m) {
            const float* xp = x + (size_t)m * K + (i << 3);
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[m] += wf[j * 2] * xp[j * 2] + wf[j * 2 + 1] * xp[j * 2 + 1];
        }
    }
    for (int i = (K8 << 3) + threadIdx.x; i < K; i += BS) {
        const float w = __bfloat162float(W[(size_t)n * K + i]);
        #pragma unroll
        for (int m = 0; m < MB; ++m) acc[m] += w * x[(size_t)m * K + i];
    }
    __shared__ float red[BS / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int m = 0; m < MB; ++m) {
        float a = acc[m];
        for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
        if (lane == 0) red[warp] = a;
        __syncthreads();
        if (warp == 0) {
            a = (lane < BS / 32) ? red[lane] : 0.f;
            for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
            if (lane == 0) y[(size_t)m * N + n] = a;
        }
        __syncthreads();
    }
}

// Scalar variants for checkpoint tensors the 16-byte path cannot legally touch (see vec16_ok).
template <int BS, int MB>
__global__ void k_gemm_f32_scalar(float* __restrict__ y, const float* __restrict__ W,
                                  const float* __restrict__ x, int N, int K) {
    const int n = blockIdx.x;
    const float* Wr = W + (size_t)n * K;
    float acc[MB];
    #pragma unroll
    for (int m = 0; m < MB; ++m) acc[m] = 0.f;
    for (int i = threadIdx.x; i < K; i += BS) {
        const float w = Wr[i];
        #pragma unroll
        for (int m = 0; m < MB; ++m) acc[m] += w * x[(size_t)m * K + i];
    }
    __shared__ float red[BS / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int m = 0; m < MB; ++m) {
        float a = acc[m];
        for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
        if (lane == 0) red[warp] = a;
        __syncthreads();
        if (warp == 0) {
            a = (lane < BS / 32) ? red[lane] : 0.f;
            for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
            if (lane == 0) y[(size_t)m * N + n] = a;
        }
        __syncthreads();
    }
}

template <int BS, int MB>
__global__ void k_gemm_bf16_scalar(float* __restrict__ y, const __nv_bfloat16* __restrict__ W,
                                   const float* __restrict__ x, int N, int K) {
    const int n = blockIdx.x;
    const __nv_bfloat16* Wr = W + (size_t)n * K;
    float acc[MB];
    #pragma unroll
    for (int m = 0; m < MB; ++m) acc[m] = 0.f;
    for (int i = threadIdx.x; i < K; i += BS) {
        const float w = __bfloat162float(Wr[i]);
        #pragma unroll
        for (int m = 0; m < MB; ++m) acc[m] += w * x[(size_t)m * K + i];
    }
    __shared__ float red[BS / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int m = 0; m < MB; ++m) {
        float a = acc[m];
        for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
        if (lane == 0) red[warp] = a;
        __syncthreads();
        if (warp == 0) {
            a = (lane < BS / 32) ? red[lane] : 0.f;
            for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
            if (lane == 0) y[(size_t)m * N + n] = a;
        }
        __syncthreads();
    }
}

#define GEMM_LAUNCH(MB)                                                                       \
    do {                                                                                      \
        if (dtype == GEMV_F32) {                                                              \
            if (vok) k_gemm_f32<BS, MB><<<N, BS, 0, s>>>(yc, (const float*)W, xc, N, K);       \
            else     k_gemm_f32_scalar<BS, MB><<<N, BS, 0, s>>>(yc, (const float*)W, xc, N, K);\
        } else {                                                                              \
            if (vok) k_gemm_bf16<BS, MB><<<N, BS, 0, s>>>(yc, (const __nv_bfloat16*)W, xc, N, K); \
            else     k_gemm_bf16_scalar<BS, MB><<<N, BS, 0, s>>>(yc, (const __nv_bfloat16*)W, xc, N, K); \
        }                                                                                     \
    } while (0)

void gemm(float* y, const void* W, const float* x, int M, int N, int K, int dtype, cudaStream_t s) {
    constexpr int BS = 256;
    const bool vok = vec16_ok(W, K, dtype == GEMV_F32 ? 4 : 2);
    int done = 0;
    while (done < M) {
        const int rem = M - done;
        // Largest exact instantiation that fits. Anything not covered splits into these, which
        // re-reads W per chunk — the same cost the caller would have paid without batching.
        const int c = rem >= 32 ? 32 : rem >= 16 ? 16 : rem >= 8 ? 8 : rem;
        float* yc = y + (size_t)done * N;
        const float* xc = x + (size_t)done * K;
        switch (c) {
            case 1:  GEMM_LAUNCH(1);  break;
            case 2:  GEMM_LAUNCH(2);  break;
            case 3:  GEMM_LAUNCH(3);  break;
            case 4:  GEMM_LAUNCH(4);  break;
            case 5:  GEMM_LAUNCH(5);  break;
            case 6:  GEMM_LAUNCH(6);  break;
            case 7:  GEMM_LAUNCH(7);  break;
            case 8:  GEMM_LAUNCH(8);  break;
            case 16: GEMM_LAUNCH(16); break;
            default: GEMM_LAUNCH(32); break;
        }
        done += c;
    }
}
#undef GEMM_LAUNCH

}  // namespace glm5
