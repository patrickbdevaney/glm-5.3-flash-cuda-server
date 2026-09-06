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
#include "dprof.h"
#include "nvfp4.cuh"
#ifndef FP4_PROBE
#define FP4_PROBE 0
#endif
// Rows per block, MEASURED not reasoned (tools/bench_gemv sweeps it). 5 is the peak on this box
// and it is a sharp one: 2->115, 3->135, 4->150, 5->161, 6->135, 8->132, 16->100, 32->75 GB/s on
// the KDA projection. Too few rows and x traffic dominates again; too many and acc[R][MB] plus
// the staged x push the block off a register cliff.
#ifndef NVFP4_R
#define NVFP4_R 5
#endif
// Batched (M > 1) shape. A gemm reads W once for all M rows, but it reads M rows of x per
// weight, and x was already the binding term at M=1 -- so the batched kernel was EXACTLY as slow
// as M separate gemvs. Measured on the bf16 path too, which is where the prefill anomaly came
// from: weight bandwidth fell 210 -> 111 -> 56 -> 28.7 -> 14.2 GB/s for M = 1, 2, 4, 8, 16,
// precisely 1/M, and prefill at width 16 (84.0 ms/tok) was SLOWER than at width 1 (81.7).
//
// So M is chunked and rows are tiled against each other. Per weight the cost is
//     0.5625 * (M / MCHUNK)        weights, re-read once per chunk
//   + 4 * MCHUNK / R               activations, divided by the rows sharing them
// under a register budget of roughly R * MCHUNK, which is why RM is expressed as that product.
// MCHUNK=4, R=8 measured best (18.2 GB/s on kda o_proj at M=16 against 7.7 for the old MB=16,
// R=2 shape); R=12 and R=16 both lose to register pressure at 93 registers already.
#ifndef NVFP4_RM
#define NVFP4_RM 32
#endif
#ifndef NVFP4_MCHUNK
#define NVFP4_MCHUNK 4
#endif
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace glm5 {

// One float4 of W folded into the accumulator, in EXACTLY the source order of the original
// rolled loop. These exist so the 4-deep unroll below can issue four independent loads before
// consuming any of them without perturbing a single addition: `acc` is threaded through, so the
// sequence of `acc +=` operations is identical to the rolled version and the result is
// bit-identical -- which matters because forward_batch at M=1 must equal decode (gate_batch).
//
// Why unroll at all: the rolled loop consumes each load immediately, which is ILP=1. The 0731
// engine measured that pattern at 110-132 GB/s on this box against 224-237 GB/s at ILP>=2, and
// dprof has mla:o_proj (a [4096, 16384] bf16, the single widest gemv in the model) at 150 GB/s
// while its narrower neighbours reach 200.
__device__ __forceinline__ float acc4_f32(float acc, float4 w, const float4 xx) {
    acc += w.x * xx.x + w.y * xx.y + w.z * xx.z + w.w * xx.w;
    return acc;
}
__device__ __forceinline__ float acc8_bf16(float acc, float4 raw, const float* __restrict__ xp) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 f = __bfloat1622float2(h[j]);
        acc += f.x * xp[j * 2] + f.y * xp[j * 2 + 1];
    }
    return acc;
}

template <int BS>
__global__ void k_gemv_f32(float* __restrict__ y, const float* __restrict__ W,
                           const float* __restrict__ x, int N, int K) {
    const int n = blockIdx.x;
    if (n >= N) return;
    const float4* Wr = reinterpret_cast<const float4*>(W + (size_t)n * K);
    const float4* xv = reinterpret_cast<const float4*>(x);
    const int K4 = K >> 2;
    float acc = 0.f;
    int i = threadIdx.x;
    for (; i + 3 * BS < K4; i += 4 * BS) {
        const float4 w0 = Wr[i], w1 = Wr[i + BS], w2 = Wr[i + 2 * BS], w3 = Wr[i + 3 * BS];
        acc = acc4_f32(acc, w0, xv[i]);
        acc = acc4_f32(acc, w1, xv[i + BS]);
        acc = acc4_f32(acc, w2, xv[i + 2 * BS]);
        acc = acc4_f32(acc, w3, xv[i + 3 * BS]);
    }
    for (; i < K4; i += BS) acc = acc4_f32(acc, Wr[i], xv[i]);
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
    int i = threadIdx.x;
    for (; i + 3 * BS < K8; i += 4 * BS) {
        const float4 r0 = Wr[i], r1 = Wr[i + BS], r2 = Wr[i + 2 * BS], r3 = Wr[i + 3 * BS];
        acc = acc8_bf16(acc, r0, x + ((size_t)i << 3));
        acc = acc8_bf16(acc, r1, x + ((size_t)(i + BS) << 3));
        acc = acc8_bf16(acc, r2, x + ((size_t)(i + 2 * BS) << 3));
        acc = acc8_bf16(acc, r3, x + ((size_t)(i + 3 * BS) << 3));
    }
    for (; i < K8; i += BS) acc = acc8_bf16(acc, Wr[i], x + ((size_t)i << 3));
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

// Block size as a function of K. A fixed 256 leaves most of a block idle on the short shapes:
// the KDA gate projections are [8192, 128], where a 256-thread block gave 224 of its threads
// nothing to do and the launch degenerated to pure latency (dprof measured kda:gates at 37% of
// achievable while the gemvs on either side of it ran at 85%).
//
// gemv and gemm MUST agree on this for a given K. The reduction tree's shape is BS/32, so a
// disagreement would silently break the invariant that forward_batch at M=1 is bit-identical to
// decode -- which gate_batch asserts and speculative verification depends on.
static inline int gemv_bs(int K) { return K <= 128 ? 32 : K <= 512 ? 64 : 256; }

// ---- NVFP4 ---------------------------------------------------------------------------------
//
// ROOFLINE §3's lever: 13.91 GiB of bf16 dense weights on the AR path become 3.91 GiB here.
//
// TWO EARLIER VERSIONS OF THIS KERNEL WERE WRONG, AND BOTH WERE WRONG ABOUT THE SAME THING.
// The first read one uint32 per iteration and took its scale through the MoE's shared LUT: it
// removed 50.6% of B_tok and bought 1.4% (8.36 -> 8.48 tok/s). The second read uint4 and used the
// hardware e4m3 converter, and was 3.4x SLOWER still. Stubbing the inner loop one term at a time
// (tools/bench_gemv, -DFP4_PROBE) settled it in one run:
//
//   stub the FP4 unpack   -> 17.1 GB/s   (no change; cvt.rn.f16x2.e2m1x2 is free)
//   stub the e4m3 scale   -> 17.1 GB/s   (no change)
//   stub the x reads      -> 323.7 GB/s  (19x)
//
// THE ACTIVATIONS ARE THE COST, NOT THE WEIGHTS. A gemv reads 4 bytes of x per weight. Against
// bf16 that is 2 bytes of x per byte of weight; against NVFP4 it is 7.1. bf16 sustains ~420 GB/s
// of x out of L2 and sits at 88-96% of streaming DRAM — it is at the right wall. NVFP4 asking for
// 1.7 TB/s of x to reach the same weight bandwidth is not, and no amount of unpack cleverness
// changes it. Halving the weight bytes cannot help while x is 7x the weight traffic.
//
// So the kernel reuses x. Each block owns R output ROWS and reads x ONCE for all of them, which
// divides activation traffic by R: at R=8 it is 0.5 bytes per weight, back under the weights
// themselves, and the kernel is bandwidth-bound on the thing that was actually removed.
//
// Granularity is uint32, deliberately, NOT the uint4 that looked wider. The packed layout ties
// 16 contiguous bytes to 32 contiguous weights, so a uint4 makes lane t read x[32t..32t+31] — at
// a fixed offset the warp's 32 lanes then touch 32 DIFFERENT 128-byte lines and use 4 bytes of
// each. uint32 gives x[8t..8t+7], the same stride the bf16 kernel uses, and the warp covers one
// contiguous 1 KB span.
//
// ONE kernel serves both gemv and gemm. The bf16 path has separate ones and pays for it with a
// standing obligation to keep two reduction trees in step; here `gemv` is literally
// k_gemm_nvfp4<BS, 1, R>, so forward_batch at M=1 being bit-identical to decode is structural
// rather than a property that has to be re-argued after every edit. gate_batch still checks it.

template <int BS, int MB, int R>
__global__ void k_gemm_nvfp4(float* __restrict__ y, const uint8_t* __restrict__ P,
                             const uint8_t* __restrict__ Sc, const float* __restrict__ gs,
                             const float* __restrict__ x, int N, int K) {
    const int n0 = blockIdx.x * R;
    const int K8 = K >> 3;                      // one uint32 = 8 weights
    const size_t prow = (size_t)K >> 1, srow = (size_t)K >> 4;

    float acc[R][MB];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int m = 0; m < MB; ++m) acc[r][m] = 0.f;

    const int rows = (N - n0) < R ? (N - n0) : R;      // last group may be short
    for (int i = threadIdx.x; i < K8; i += BS) {
        // x once, reused by all R rows. This load is the whole point of the kernel.
        float xr[MB][8];
        #pragma unroll
        for (int m = 0; m < MB; ++m)
            #pragma unroll
            for (int j = 0; j < 8; ++j)
#if FP4_PROBE == 3                      // stub the x reads -- this is the term that mattered
                xr[m][j] = 1.f;
#else
                xr[m][j] = x[(size_t)m * K + ((size_t)i << 3) + j];
#endif
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r >= rows) break;
            const unsigned pv =
                reinterpret_cast<const unsigned*>(P + (size_t)(n0 + r) * prow)[i];
#if FP4_PROBE == 2                      // stub the e4m3 scale converter
            const float sc = (float)Sc[(size_t)(n0 + r) * srow + (i >> 1)];
#else
            const float sc = fp8e4m3(Sc[(size_t)(n0 + r) * srow + (i >> 1)]);
#endif
            float w[8];
#if FP4_PROBE == 1                      // stub the e2m1 unpack
            #pragma unroll
            for (int j = 0; j < 8; ++j) w[j] = (float)((pv >> j) & 1);
#else
            e2m1x8(w, pv);
#endif
            #pragma unroll
            for (int m = 0; m < MB; ++m) {
                float t = 0.f;
                #pragma unroll
                for (int j = 0; j < 8; ++j) t = fmaf(w[j], xr[m][j], t);
                acc[r][m] = fmaf(t, sc, acc[r][m]);
            }
        }
    }

    // The global scale is a per-tensor constant, so it comes out of the loop and applies once.
    const float ig = 1.f / gs[0];
    __shared__ float red[BS / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        if (r >= rows) break;
        #pragma unroll
        for (int m = 0; m < MB; ++m) {
            float a = acc[r][m];
            for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
            if (lane == 0) red[warp] = a;
            __syncthreads();
            if (warp == 0) {
                a = (lane < BS / 32) ? red[lane] : 0.f;
                for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
                if (lane == 0) y[(size_t)m * N + n0 + r] = a * ig;
            }
            __syncthreads();                    // red[] is reused by the next (r, m)
        }
    }
}

// Rows per block. x traffic per weight is 4*MB/R bytes, so R has to grow with MB to hold it
// under the 0.5625 bytes the weights themselves cost — but acc[R][MB] and xr[MB][8] both live in
// registers, so R*MB is the budget and 8 is where it lands.
template <int MB> struct Rows { static constexpr int v = MB <= 1 ? NVFP4_R : NVFP4_RM / MB; };

#define NVFP4_LAUNCH_BS(BS, MB)                                                        \
    k_gemm_nvfp4<BS, MB, Rows<MB>::v><<<(N + Rows<MB>::v - 1) / Rows<MB>::v, BS, 0, s>>>( \
        yc, W.packed, W.scale, W.gscale, xc, N, K)
// Block size for the NVFP4 path, which has its own work granularity: the loop trip count is
// K/8, not K/4 or K/8-of-a-float4, so gemv_bs()'s thresholds leave threads idle on the narrow
// shapes (K=1536 gave 192 units of work to 256 threads). gemv and gemm both come through
// gemm_nvfp4, so they cannot disagree with each other — which is the invariant that matters.
static inline int nvfp4_bs(int K) {
    const int units = K >> 3;
    return units <= 128 ? 32 : units <= 256 ? 64 : units <= 1024 ? 128 : 256;
}

#define NVFP4_LAUNCH(MB)                                     \
    do {                                                     \
        switch (nvfp4_bs(K)) {                               \
            case 32:  NVFP4_LAUNCH_BS(32,  MB); break;       \
            case 64:  NVFP4_LAUNCH_BS(64,  MB); break;       \
            case 128: NVFP4_LAUNCH_BS(128, MB); break;       \
            default:  NVFP4_LAUNCH_BS(256, MB); break;       \
        }                                                    \
    } while (0)

static void gemm_nvfp4(float* y, const WRef& W, const float* x, int M, int N, int K,
                       cudaStream_t s) {
    // K must be a multiple of 16 (the NVFP4 group). tools/requant_dense_nvfp4.py refuses to emit
    // anything else, so a violation here means the overlay and the engine disagree about a shape.
    if (K & 15) { fprintf(stderr, "gemm_nvfp4: K=%d is not a multiple of 16\n", K); abort(); }
    // W is streamed once per chunk of the M loop, NOT once per call and NOT once per token. That
    // pass count is the whole reason the old per-token model mispriced prefill, so it is counted
    // here, where the chunking actually happens, rather than modelled anywhere else.
    dprof_bytes((double)((M + NVFP4_MCHUNK - 1) / NVFP4_MCHUNK) * N * K * 0.5625);
    int done = 0;
    while (done < M) {
        const int rem = M - done;
        const int c = rem >= NVFP4_MCHUNK ? NVFP4_MCHUNK : rem;
        float* yc = y + (size_t)done * N;
        const float* xc = x + (size_t)done * K;
        switch (c) {
            case 1:  NVFP4_LAUNCH(1);  break;
            case 2:  NVFP4_LAUNCH(2);  break;
            case 3:  NVFP4_LAUNCH(3);  break;
            case 4:  NVFP4_LAUNCH(4);  break;
            case 5:  NVFP4_LAUNCH(5);  break;
            case 6:  NVFP4_LAUNCH(6);  break;
            case 7:  NVFP4_LAUNCH(7);  break;
            default: NVFP4_LAUNCH(8);  break;
        }
        done += c;
    }
}
#undef NVFP4_LAUNCH
#undef NVFP4_LAUNCH_BS

#define GEMV_LAUNCH(BS)                                                                          \
    do {                                                                                         \
        if (dtype == GEMV_F32) {                                                                 \
            if (vec16_ok(W, K, 4)) k_gemv_f32<BS><<<N, BS, 0, s>>>(y, (const float*)W, x, N, K); \
            else k_gemv_f32_scalar<BS><<<N, BS, 0, s>>>(y, (const float*)W, x, N, K);            \
        } else {                                                                                 \
            if (vec16_ok(W, K, 2))                                                               \
                k_gemv_bf16<BS><<<N, BS, 0, s>>>(y, (const __nv_bfloat16*)W, x, N, K);            \
            else k_gemv_bf16_scalar<BS><<<N, BS, 0, s>>>(y, (const __nv_bfloat16*)W, x, N, K);    \
        }                                                                                        \
    } while (0)

// Bytes on the wire per weight. NVFP4 is 0.5 for the packed nibble plus one f8 scale per group of
// 16, so 0.5625 -- the same figure ROOFLINE §3 uses.
static inline double wbytes(const WRef& W, int dtype) {
    return W.nvfp4() ? 0.5625 : (dtype == GEMV_F32 ? 4.0 : 2.0);
}

void gemv(float* y, const WRef& Wr, const float* x, int N, int K, int dtype, cudaStream_t s) {
    if (Wr.nvfp4()) { gemm(y, Wr, x, 1, N, K, dtype, s); return; }   // gemm does the accounting
    dprof_bytes((double)N * K * wbytes(Wr, dtype));
    const void* W = Wr.p;
    switch (gemv_bs(K)) {
        case 32:  GEMV_LAUNCH(32);  break;
        case 64:  GEMV_LAUNCH(64);  break;
        default:  GEMV_LAUNCH(256); break;
    }
}
#undef GEMV_LAUNCH

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

#define GEMM_LAUNCH_BS(BS, MB)                                                                       \
    do {                                                                                      \
        if (dtype == GEMV_F32) {                                                              \
            if (vok) k_gemm_f32<BS, MB><<<N, BS, 0, s>>>(yc, (const float*)W, xc, N, K);       \
            else     k_gemm_f32_scalar<BS, MB><<<N, BS, 0, s>>>(yc, (const float*)W, xc, N, K);\
        } else {                                                                              \
            if (vok) k_gemm_bf16<BS, MB><<<N, BS, 0, s>>>(yc, (const __nv_bfloat16*)W, xc, N, K); \
            else     k_gemm_bf16_scalar<BS, MB><<<N, BS, 0, s>>>(yc, (const __nv_bfloat16*)W, xc, N, K); \
        }                                                                                     \
    } while (0)

// Mirrors gemv_bs(K) exactly -- see the note there on why they must not diverge.
#define GEMM_LAUNCH(MB)                                    \
    do {                                                   \
        switch (gemv_bs(K)) {                              \
            case 32:  GEMM_LAUNCH_BS(32,  MB); break;      \
            case 64:  GEMM_LAUNCH_BS(64,  MB); break;      \
            default:  GEMM_LAUNCH_BS(256, MB); break;      \
        }                                                  \
    } while (0)

void gemm(float* y, const WRef& Wr, const float* x, int M, int N, int K, int dtype, cudaStream_t s) {
    if (Wr.nvfp4()) { gemm_nvfp4(y, Wr, x, M, N, K, s); return; }
    const void* W = Wr.p;
    const bool vok = vec16_ok(W, K, dtype == GEMV_F32 ? 4 : 2);
    {   // one pass over W per chunk; mirror the chunk sizes chosen below exactly
        int d = 0, passes = 0;
        while (d < M) { const int r = M - d; d += r >= 32 ? 32 : r >= 16 ? 16 : r >= 8 ? 8 : r; ++passes; }
        dprof_bytes((double)passes * N * K * wbytes(Wr, dtype));
    }
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
#undef GEMM_LAUNCH_BS

}  // namespace glm5
