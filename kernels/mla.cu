// mla.cu — Multi-head Latent Attention, decode step, for the 11 full-attention layers.
//
// 13.1% of B_tok. Two things make this model's MLA simpler than DeepSeek-V4's:
//   * qk_rope_head_dim == 0 and mla_use_nope == true, so there is NO rotary embedding in the main
//     attention path at all — no YaRN, no rope cache, no interleave. Position information reaches
//     these layers through the 34 KDA layers beneath them.
//   * there is no o_lora / o_groups factorisation; o_proj is one [4096, 16384] matrix.
//
// The decode path uses the ABSORBED form. Instead of expanding the 512-wide latent into 64 heads
// of (256 key + 256 value) per cached token, W_k is folded into the query once:
//
//     qa[h] = W_k[h]^T q[h]                    [512]      64 x (256x512), read kv_b once
//     s[h][t] = qa[h] . C[t] * scaling                    attention runs on the latent directly
//     ctx[h] = sum_t a[h][t] C[t]              [512]
//     o[h]   = W_v[h] ctx[h]                   [256]      expand only once, at the end
//
// This is algebraically identical to expanding (ref/gen_mla.py measures the difference at ~1e-6)
// and it is why only the 512-wide latent needs caching: 88 MiB at 8k context across all 11 layers,
// against 1408 MiB for the expanded form.
#include "mla.h"
#include "mla_context.cuh"
#include "dprof.h"
#include "indexer.h"
#include "gemv.h"
#include "layer.h"
#include "glm5_config.h"
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

namespace glm5 {

#define CU(x) do { cudaError_t e_=(x); if(e_){ fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(e_)); abort(); } } while(0)

// A LAUNCH THAT FAILS ON RESOURCES DOES NOTHING AND COSTS NOTHING, AND THAT LOOKS LIKE A WIN.
// Sweeping k_context's HG (heads per block) reported HG=32 as 6.3x faster than HG=16 and HG=64
// faster still -- 12x off the trend line the other points sat on. Both were
// "too many resources requested for launch": acc[HG] at 512 threads exceeds the per-thread
// register budget, the kernel never ran, and dprof honestly timed an empty stream slot. Only
// gate_mla caught it. cudaGetLastError is a host-side thread-local read with no sync, so this is
// affordable per launch, and the alternative is believing a number that is 12x too good.
#define KCHK(name) do { cudaError_t e_ = cudaGetLastError(); if (e_) { \
    fprintf(stderr, "cuda launch %s (%s:%d): %s\n", name, __FILE__, __LINE__, \
            cudaGetErrorString(e_)); abort(); } } while(0)

// Hh and Lk come from mla_context.cuh, which the k_context family needs to be self-contained
// for tools/bench_context.cu.
// Latent-cache traffic: `reads` full passes over n rows of 512 fp32. The sparse twins' count
// lives on the device, so above DENSE_CTX_LIMIT this uses the selector's cap -- exact whenever it
// saturates, which is the regime that matters.
#define DPCACHE(n, reads) dprof_bytes((double)(reads) * (double)(n) * MLA_KV_LORA * 4.0)

static constexpr int Dq = MLA_QK_NOPE;    // 256
static constexpr int Dv = MLA_V_HEAD;     // 256
static constexpr int ROW = Dq + Dv;       // 512 rows of kv_b per head

// Tokens per block in the batched absorb/expand kernels. Every weight byte is read once per chunk,
// so kv_b traffic per chunk falls by min(M, MLA_MB). 8 keeps acc[] in registers at 512 threads;
// larger spills, which costs more than the extra reuse buys.
#ifndef MLA_MB
#define MLA_MB 8
#endif

// Heads per block in k_context. Lowering it multiplies BOTH the block count and the cache traffic
// by the same factor, so sweeping it separates an occupancy-bound kernel from a traffic-bound one
// with a known sign in each direction. See OPTIMIZATION_LOG #15.
// Swept on the real kernel by tools/bench_context.cu; see OPTIMIZATION_LOG #16 for the grid.
// The result inverts the obvious model: HG=4 reads the latent cache 16 times per call and beats
// HG=16, which reads it 4 times, by 6.5x. The cache is L2-resident at these sizes (the winning
// point runs at 610 GB/s against a 237 GB/s streaming read), so re-reads are nearly free and the
// binding constraint is per-SM occupancy, which acc[HG] destroys. Below HG=4 the traffic finally
// bites -- HG=1 saturates L2 at ~1290 GB/s and loses.
#ifndef MLA_HG
#define MLA_HG 4
#endif

// t-tiles per k_context launch. Splitting the CACHED-TOKEN axis is how this kernel gets blocks
// without paying for them: each tile owns a disjoint slice of the latent cache, so NT tiles read
// the cache once between them, where NT head-groups would each read all of it. That is the whole
// difference from the HG knob -- lowering HG bought parallelism at 1 extra full cache read per
// block, and hit the L2 ceiling at ~419 GB/s; NT buys the same parallelism at zero extra reads,
// and costs only the partial buffer it writes and the reduce pass that sums it.
//
// NT is a COMPILE-TIME CONSTANT and grid.y is always NT, including for the sparse twin whose
// count lives on the device. Tiles that fall past the end write zeros rather than being skipped
// on the host, which is what lets the dense and sparse paths partition t identically and stay
// bit-exact against each other (tests/gate_mla_sparse.cu).
#ifndef MLA_NT
#define MLA_NT 16
#endif

// workspace: q_resid[1536] | q[16384] | c_new[512] | qa[64*512] | scores[64*max_ctx] |
//            ctx[64*512] | heads[16384]
size_t mla_workspace_floats(int max_ctx) {
    return MLA_Q_LORA + MLA_Q_DIM + Lk + (size_t)Hh * Lk + (size_t)Hh * max_ctx
         + (size_t)Hh * Lk + (size_t)Hh * Dv + (size_t)MLA_NT * Hh * Lk;
}

// qa AND ctx are now per-token, because absorb_q and expand_v batch over M (see
// k_absorb_q_batch). That costs M*64*512 floats each -- 4 MiB apiece at M=32 -- to stop kv_b
// being streamed M times per layer. Only `scores` stays single-token: the score/softmax/context
// chain is still serial over tokens, so one [64, max_ctx] scratch is reused.
size_t mla_batch_workspace_floats(int max_ctx, int M) {
    return (size_t)M * (MLA_Q_LORA + MLA_Q_DIM + Lk + (size_t)Hh * Dv)
         + (size_t)M * Hh * Lk + (size_t)Hh * max_ctx + (size_t)M * Hh * Lk
         + (size_t)MLA_NT * Hh * Lk;          // k_context partials, reused across the M loop
}

// qa[h][l] = sum_{d<256} q[h][d] * kv_b[(h*512 + d)*512 + l]
// One block per head, 512 threads, thread l owns output column l. For a fixed d all 512 threads
// read 512 consecutive bf16 — one fully-coalesced 1 KB row per step.
__global__ void k_absorb_q(float* __restrict__ qa, const float* __restrict__ q,
                           const __nv_bfloat16* __restrict__ kv_b) {
    const int h = blockIdx.x, l = threadIdx.x;
    const __nv_bfloat16* Wk = kv_b + (size_t)h * ROW * Lk;
    float acc = 0.f;
    for (int d = 0; d < Dq; ++d) acc += q[(size_t)h * Dq + d] * __bfloat162float(Wk[(size_t)d * Lk + l]);
    qa[(size_t)h * Lk + l] = acc;
}

// Batched twin. THIS IS THE POINT OF THE WHOLE CHANGE: k_absorb_q streams the entire W_k half of
// kv_b (bf16 [32768, 512], 33.55 MB per layer) for ONE token, and the prefill loop called it once
// per token. With the dprof sub-phase marks in place the call counts said so outright -- absorb_q
// and expand_v at 6336 calls (198 x 32) against 198 for the projections -- and together they were
// 0.738 G/token, 7.6% of B_tok, where ROOFLINE §1 prices one read at 0.344 G.
//
// A block now owns (head, chunk of MB tokens) and reads each weight ONCE for all MB of them. It is
// the same insight as the MoE expert gathering and the row-tiled NVFP4 gemv: the weight does not
// care who reads it, so the fix is always to widen the consumer, never to speed up the read.
//
// The `if (i < nm)` inside an unrolled loop over a compile-time bound is deliberate -- a runtime
// bound would push acc[] out of registers and into local memory, which costs more than the tail
// block saves.
template <int MB>
__global__ void k_absorb_q_batch(float* __restrict__ qa, const float* __restrict__ q,
                                 const __nv_bfloat16* __restrict__ kv_b, int M) {
    const int h = blockIdx.x, l = threadIdx.x;
    const int m0 = blockIdx.y * MB;
    const int nm = (M - m0) < MB ? (M - m0) : MB;
    const __nv_bfloat16* Wk = kv_b + (size_t)h * ROW * Lk;
    float acc[MB];
    #pragma unroll
    for (int i = 0; i < MB; ++i) acc[i] = 0.f;
    for (int d = 0; d < Dq; ++d) {
        const float w = __bfloat162float(Wk[(size_t)d * Lk + l]);   // one coalesced 1 KB row
        #pragma unroll
        for (int i = 0; i < MB; ++i)
            if (i < nm) acc[i] += q[(size_t)(m0 + i) * MLA_Q_DIM + (size_t)h * Dq + d] * w;
    }
    #pragma unroll
    for (int i = 0; i < MB; ++i)
        if (i < nm) qa[((size_t)(m0 + i) * Hh + h) * Lk + l] = acc[i];
}

// s[h][t] = qa[h] . C[t] * scaling
//
// One block per tile of 8 cached tokens, looping all 64 heads. The LATENT CACHE IS READ ONCE per
// block — the obvious layout (one block per (head, token)) would re-read it 64 times, which at 8k
// context is 268 MB per layer per token instead of 16 MB. qa is only 128 KB and stays in L2.
template <int TT>
__global__ void k_scores(float* __restrict__ scores, const float* __restrict__ qa,
                         const float* __restrict__ cache, int n_tok, int max_ctx, float scaling) {
    __shared__ float Ct[TT][Lk];
    const int t0 = blockIdx.x * TT;
    for (int i = threadIdx.x; i < TT * Lk; i += blockDim.x) {
        const int tt = i / Lk, l = i - tt * Lk;
        Ct[tt][l] = (t0 + tt < n_tok) ? cache[(size_t)(t0 + tt) * Lk + l] : 0.f;
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;   // TT warps, one per token
    if (warp >= TT || t0 + warp >= n_tok) return;
    for (int h = 0; h < Hh; ++h) {
        float acc = 0.f;
        for (int l = lane; l < Lk; l += 32) acc += qa[(size_t)h * Lk + l] * Ct[warp][l];
        for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if (lane == 0) scores[(size_t)h * max_ctx + t0 + warp] = acc * scaling;
    }
}

// Sparse twin of k_scores: slot i attends key sel[i] instead of key i. `sel` is ASCENDING (see
// k_select_emit), so the cache is still read front-to-back — the gather is sequential, not random.
template <int TT>
__global__ void k_scores_sel(float* __restrict__ scores, const float* __restrict__ qa,
                             const float* __restrict__ cache, const int32_t* __restrict__ sel,
                             const int32_t* __restrict__ n_ptr, int max_ctx, float scaling) {
    __shared__ float Ct[TT][Lk];
    // The count lives on the DEVICE and the grid is sized for the worst case. Reading it back to
    // pick a launch size would mean a stream sync per full-attention layer per token — 11 pipeline
    // stalls a step, to save blocks that exit in nanoseconds.
    const int n_sel = *n_ptr;
    const int i0 = blockIdx.x * TT;
    if (i0 >= n_sel) return;
    for (int i = threadIdx.x; i < TT * Lk; i += blockDim.x) {
        const int tt = i / Lk, l = i - tt * Lk;
        const int t = (i0 + tt < n_sel) ? sel[i0 + tt] : -1;
        Ct[tt][l] = (t >= 0) ? cache[(size_t)t * Lk + l] : 0.f;
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp >= TT || i0 + warp >= n_sel) return;
    for (int h = 0; h < Hh; ++h) {
        float acc = 0.f;
        for (int l = lane; l < Lk; l += 32) acc += qa[(size_t)h * Lk + l] * Ct[warp][l];
        for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if (lane == 0) scores[(size_t)h * max_ctx + i0 + warp] = acc * scaling;
    }
}

// softmax over the n_tok visible positions, one block per head.
template <int BS>
__global__ void k_softmax(float* __restrict__ s, int n_tok, int max_ctx) {
    const int h = blockIdx.x;
    float* row = s + (size_t)h * max_ctx;
    __shared__ float rm[BS / 32], rs[BS / 32];
    float m = -1e30f;
    for (int t = threadIdx.x; t < n_tok; t += BS) m = fmaxf(m, row[t]);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, o));
    if ((threadIdx.x & 31) == 0) rm[threadIdx.x >> 5] = m;
    __syncthreads();
    __shared__ float mx;
    if (threadIdx.x == 0) { float v = -1e30f; for (int k = 0; k < BS / 32; ++k) v = fmaxf(v, rm[k]); mx = v; }
    __syncthreads();
    float sum = 0.f;
    for (int t = threadIdx.x; t < n_tok; t += BS) { const float e = __expf(row[t] - mx); row[t] = e; sum += e; }
    for (int o = 16; o; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((threadIdx.x & 31) == 0) rs[threadIdx.x >> 5] = sum;
    __syncthreads();
    __shared__ float tot;
    if (threadIdx.x == 0) { float v = 0; for (int k = 0; k < BS / 32; ++k) v += rs[k]; tot = v; }
    __syncthreads();
    for (int t = threadIdx.x; t < n_tok; t += BS) row[t] /= tot;
}

// Same softmax, count read from device memory. Duplicated rather than templated on a predicate so
// the dense path keeps its compile-time bound and nothing about it changes.
template <int BS>
__global__ void k_softmax_dev(float* __restrict__ s, const int32_t* __restrict__ n_ptr, int max_ctx) {
    const int n_tok = *n_ptr;
    const int h = blockIdx.x;
    float* row = s + (size_t)h * max_ctx;
    __shared__ float rm[BS / 32], rs[BS / 32];
    float m = -1e30f;
    for (int t = threadIdx.x; t < n_tok; t += BS) m = fmaxf(m, row[t]);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, o));
    if ((threadIdx.x & 31) == 0) rm[threadIdx.x >> 5] = m;
    __syncthreads();
    __shared__ float mx;
    if (threadIdx.x == 0) { float v = -1e30f; for (int k = 0; k < BS / 32; ++k) v = fmaxf(v, rm[k]); mx = v; }
    __syncthreads();
    float sum = 0.f;
    for (int t = threadIdx.x; t < n_tok; t += BS) { const float e = __expf(row[t] - mx); row[t] = e; sum += e; }
    for (int o = 16; o; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((threadIdx.x & 31) == 0) rs[threadIdx.x >> 5] = sum;
    __syncthreads();
    __shared__ float tot;
    if (threadIdx.x == 0) { float v = 0; for (int k = 0; k < BS / 32; ++k) v += rs[k]; tot = v; }
    __syncthreads();
    for (int t = threadIdx.x; t < n_tok; t += BS) row[t] /= tot;
}

// o[h][d] = sum_l ctx[h][l] * kv_b[(h*512 + 256 + d)*512 + l]
// One block per (head, output dim); each block streams one contiguous 512-element bf16 row.
template <int BS>
__global__ void k_expand_v(float* __restrict__ out, const float* __restrict__ ctx,
                           const __nv_bfloat16* __restrict__ kv_b) {
    const int h = blockIdx.x / Dv, d = blockIdx.x - h * Dv;
    const __nv_bfloat16* Wv = kv_b + ((size_t)h * ROW + Dq + d) * Lk;
    const float* c = ctx + (size_t)h * Lk;
    float acc = 0.f;
    for (int l = threadIdx.x; l < Lk; l += BS) acc += c[l] * __bfloat162float(Wv[l]);
    __shared__ float red[BS / 32];
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) { float v = 0; for (int k = 0; k < BS / 32; ++k) v += red[k]; out[(size_t)h * Dv + d] = v; }
}

// Batched twin of k_expand_v: same block-per-(head, output dim), same reduction order, but the
// 512-element W_v row is read once for MB tokens instead of once per token. Requires ctx to have
// been kept for all M tokens, which is why the workspace grew.
template <int BS, int MB>
__global__ void k_expand_v_batch(float* __restrict__ out, const float* __restrict__ ctx,
                                 const __nv_bfloat16* __restrict__ kv_b, int M) {
    const int hd = blockIdx.x, h = hd / Dv, d = hd - h * Dv;
    const int m0 = blockIdx.y * MB;
    const int nm = (M - m0) < MB ? (M - m0) : MB;
    const __nv_bfloat16* Wv = kv_b + ((size_t)h * ROW + Dq + d) * Lk;
    float acc[MB];
    #pragma unroll
    for (int i = 0; i < MB; ++i) acc[i] = 0.f;
    for (int l = threadIdx.x; l < Lk; l += BS) {
        const float w = __bfloat162float(Wv[l]);
        #pragma unroll
        for (int i = 0; i < MB; ++i)
            if (i < nm) acc[i] += ctx[((size_t)(m0 + i) * Hh + h) * Lk + l] * w;
    }
    __shared__ float red[MB][BS / 32];
    #pragma unroll
    for (int i = 0; i < MB; ++i) {
        float a = acc[i];
        for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
        if ((threadIdx.x & 31) == 0) red[i][threadIdx.x >> 5] = a;
    }
    __syncthreads();
    if ((int)threadIdx.x < nm) {
        float v = 0;                                          // same order as k_expand_v
        for (int k = 0; k < BS / 32; ++k) v += red[threadIdx.x][k];
        out[((size_t)(m0 + threadIdx.x) * Hh + h) * Dv + d] = v;
    }
}

__global__ void k_store_latent(float* __restrict__ cache, const float* __restrict__ c_new, int t) {
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l < Lk) cache[(size_t)t * Lk + l] = c_new[l];
}

void mla_decode_step(const float* x, const MlaWeights& W, float* cache, int t, int max_ctx,
                     float* y, float* ws, cudaStream_t s) {
    float* q_resid = ws;
    float* q       = ws + MLA_Q_LORA;
    float* c_new   = q + MLA_Q_DIM;
    float* qa      = c_new + Lk;
    float* scores  = qa + (size_t)Hh * Lk;
    float* ctx     = scores + (size_t)Hh * max_ctx;
    float* heads   = ctx + (size_t)Hh * Lk;
    float* cpart   = heads + (size_t)Hh * Dv;                   // [NT, 64, 512]
    const int n_tok = t + 1;
    const float scaling = rsqrtf((float)(MLA_QK_NOPE + MLA_QK_ROPE));   // 1/16

    dprof_begin(DP_M_QPROJ, s);
    gemv(q_resid, W.q_a, x, MLA_Q_LORA, HIDDEN, W.dtype, s);
    rmsnorm(q_resid, q_resid, W.q_a_norm, W.dtype, MLA_Q_LORA, s);
    gemv(q, W.q_b, q_resid, MLA_Q_DIM, MLA_Q_LORA, W.dtype, s);
    dprof_end(DP_M_QPROJ, s);

    dprof_begin(DP_M_KV, s);
    gemv(c_new, W.kv_a, x, Lk, HIDDEN, W.dtype, s);
    rmsnorm(c_new, c_new, W.kv_a_norm, W.dtype, Lk, s);
    k_store_latent<<<(Lk + 255) / 256, 256, 0, s>>>(cache, c_new, t);
    dprof_end(DP_M_KV, s);

    dprof_bytes((double)Hh * Dq * Lk * 2.0);
    k_absorb_q<<<Hh, Lk, 0, s>>>(qa, q, (const __nv_bfloat16*)W.kv_b);
    constexpr int TT = 8;
    DPCACHE(n_tok, 1);
        k_scores<TT><<<(n_tok + TT - 1) / TT, TT * 32, 0, s>>>(scores, qa, cache, n_tok, max_ctx, scaling);
    k_softmax<256><<<Hh, 256, 0, s>>>(scores, n_tok, max_ctx);
    constexpr int HG = MLA_HG;
    constexpr int NT = MLA_NT;
    DPCACHE(n_tok, Hh / HG);
        k_context_part<HG, NT><<<dim3(Hh / HG, NT), Lk, 0, s>>>(cpart, scores, cache, n_tok, max_ctx);
    KCHK("k_context_part");
    k_context_reduce<NT><<<Hh * Lk / 256, 256, 0, s>>>(ctx, cpart);
    KCHK("k_context_reduce");
    dprof_bytes((double)Hh * Dv * Lk * 2.0);
    k_expand_v<128><<<Hh * Dv, 128, 0, s>>>(heads, ctx, (const __nv_bfloat16*)W.kv_b);
    gemv(y, W.o_proj, heads, HIDDEN, Hh * Dv, W.dtype, s);
}

void mla_batch_step(const float* x, const MlaWeights& W, float* cache, int pos0, int M,
                    int max_ctx, float* y, float* ws, cudaStream_t s) {
    float* q_resid = ws;                                        // [M, 1536]
    float* q       = q_resid + (size_t)M * MLA_Q_LORA;          // [M, 16384]
    float* c_new   = q + (size_t)M * MLA_Q_DIM;                 // [M, 512]
    float* heads   = c_new + (size_t)M * Lk;                    // [M, 16384]
    float* qa      = heads + (size_t)M * Hh * Dv;               // [M, 64, 512]
    float* scores  = qa + (size_t)M * Hh * Lk;                  // [64, max_ctx]  one token at a time
    float* ctx     = scores + (size_t)Hh * max_ctx;             // [M, 64, 512]
    float* cpart   = ctx + (size_t)M * Hh * Lk;                 // [NT, 64, 512], reused per token
    const float scaling = rsqrtf((float)(MLA_QK_NOPE + MLA_QK_ROPE));   // 1/16
    const int MG = (M + MLA_MB - 1) / MLA_MB;

    gemm(q_resid, W.q_a, x, M, MLA_Q_LORA, HIDDEN, W.dtype, s);
    for (int m = 0; m < M; ++m)
        rmsnorm(q_resid + (size_t)m * MLA_Q_LORA, q_resid + (size_t)m * MLA_Q_LORA,
                W.q_a_norm, W.dtype, MLA_Q_LORA, s);
    gemm(q, W.q_b, q_resid, M, MLA_Q_DIM, MLA_Q_LORA, W.dtype, s);

    gemm(c_new, W.kv_a, x, M, Lk, HIDDEN, W.dtype, s);
    for (int m = 0; m < M; ++m)
        rmsnorm(c_new + (size_t)m * Lk, c_new + (size_t)m * Lk, W.kv_a_norm, W.dtype, Lk, s);

    // Every latent lands in the cache BEFORE any attention reads it — see the header note on
    // causality within the batch.
    for (int m = 0; m < M; ++m)
        k_store_latent<<<(Lk + 255) / 256, 256, 0, s>>>(cache, c_new + (size_t)m * Lk, pos0 + m);

    // Absorb for ALL M tokens first: it depends only on q and kv_b, not on the cache or on any
    // other token's attention, so hoisting it out of the loop is free and reads kv_b MG times
    // instead of M.
    dprof_bytes((double)MG * Hh * Dq * Lk * 2.0);              // W_k half of kv_b, once per chunk
    k_absorb_q_batch<MLA_MB><<<dim3(Hh, MG), Lk, 0, s>>>(qa, q, (const __nv_bfloat16*)W.kv_b, M);

    for (int m = 0; m < M; ++m) {
        const int n_tok = pos0 + m + 1;
        float* qa_m  = qa  + (size_t)m * Hh * Lk;
        float* ctx_m = ctx + (size_t)m * Hh * Lk;
        constexpr int TT = 8;
        DPCACHE(n_tok, 1);
        k_scores<TT><<<(n_tok + TT - 1) / TT, TT * 32, 0, s>>>(scores, qa_m, cache, n_tok, max_ctx, scaling);
        k_softmax<256><<<Hh, 256, 0, s>>>(scores, n_tok, max_ctx);
        constexpr int HG = MLA_HG;
        constexpr int NT = MLA_NT;
        DPCACHE(n_tok, Hh / HG);
        k_context_part<HG, NT><<<dim3(Hh / HG, NT), Lk, 0, s>>>(cpart, scores, cache, n_tok, max_ctx);
        KCHK("k_context_part");
        k_context_reduce<NT><<<Hh * Lk / 256, 256, 0, s>>>(ctx_m, cpart);
        KCHK("k_context_reduce");
    }

    dprof_bytes((double)MG * Hh * Dv * Lk * 2.0);              // W_v half of kv_b
    k_expand_v_batch<128, MLA_MB><<<dim3(Hh * Dv, MG), 128, 0, s>>>(
        heads, ctx, (const __nv_bfloat16*)W.kv_b, M);
    gemm(y, W.o_proj, heads, M, HIDDEN, Hh * Dv, W.dtype, s);
}

// DSA decode: run the indexer, then attend only the keys it selected.
//
// Below DENSE_CTX_LIMIT this is a strictly redundant path — the indexer selects every visible key,
// the ascending emit makes the accumulation order identical, and the result is BIT-IDENTICAL to
// mla_decode_step. tests/gate_mla_sparse.cu asserts exactly that, which is the only end-to-end
// check available for the sparse path: above the limit there is no dense answer to compare to.
void mla_decode_step_dsa(const float* x, const MlaWeights& W, const IndexerWeights& IW,
                         IndexerState& IS, float* cache, int t, int max_ctx,
                         int32_t* sel, int32_t* nsel, float* y, float* ws, float* iws,
                         cudaStream_t s, bool force_sparse) {
    float* q_resid = ws;
    float* q       = ws + MLA_Q_LORA;
    float* c_new   = q + MLA_Q_DIM;
    float* qa      = c_new + Lk;
    float* scores  = qa + (size_t)Hh * Lk;
    float* ctx     = scores + (size_t)Hh * max_ctx;
    float* heads   = ctx + (size_t)Hh * Lk;
    float* cpart   = heads + (size_t)Hh * Dv;                   // [NT, 64, 512]
    const int n_tok = t + 1;
    const float scaling = rsqrtf((float)(MLA_QK_NOPE + MLA_QK_ROPE));

    dprof_begin(DP_M_QPROJ, s);
    gemv(q_resid, W.q_a, x, MLA_Q_LORA, HIDDEN, W.dtype, s);
    rmsnorm(q_resid, q_resid, W.q_a_norm, W.dtype, MLA_Q_LORA, s);
    gemv(q, W.q_b, q_resid, MLA_Q_DIM, MLA_Q_LORA, W.dtype, s);
    dprof_end(DP_M_QPROJ, s);

    dprof_begin(DP_M_KV, s);
    gemv(c_new, W.kv_a, x, Lk, HIDDEN, W.dtype, s);
    rmsnorm(c_new, c_new, W.kv_a_norm, W.dtype, Lk, s);
    k_store_latent<<<(Lk + 255) / 256, 256, 0, s>>>(cache, c_new, t);
    dprof_end(DP_M_KV, s);

    // THE POOL STATE MUST BE MAINTAINED FROM TOKEN 0, EVEN WHILE ATTENTION IS STILL DENSE.
    // Pool keys are built incrementally as each group of 4 tokens completes, so a run that only
    // started the indexer once the context crossed 2051 would have no pool keys for the first
    // 512 pools — which is most of the context, and exactly the part it then has to score.
    dprof_begin(DP_M_INDEXER, s);
    indexer_keys(x, IW, IS, t, iws, s);
    dprof_end(DP_M_INDEXER, s);

    dprof_begin(DP_M_ABSORB, s);
    dprof_bytes((double)Hh * Dq * Lk * 2.0);
    k_absorb_q<<<Hh, Lk, 0, s>>>(qa, q, (const __nv_bfloat16*)W.kv_b);
    dprof_end(DP_M_ABSORB, s);
    constexpr int TT = 8;
    constexpr int HG = MLA_HG;
    constexpr int NT = MLA_NT;

    dprof_begin(DP_M_SDPA, s);
    if (n_tok <= DENSE_CTX_LIMIT && !force_sparse) {
        // Below the limit the indexer provably selects every visible key (ref/gen_indexer.py), so
        // scoring and selecting would burn a top-k over every pool to arrive at "all of them".
        // Dense attention is the same answer for less work — bit-identically, which
        // tests/gate_mla_sparse.cu asserts for all 2051 steps.
        dprof_begin(DP_S_SCORES, s);
        DPCACHE(n_tok, 1);
        k_scores<TT><<<(n_tok + TT - 1) / TT, TT * 32, 0, s>>>(scores, qa, cache, n_tok, max_ctx, scaling);
        dprof_end(DP_S_SCORES, s);
        dprof_begin(DP_S_SOFTMAX, s);
        k_softmax<256><<<Hh, 256, 0, s>>>(scores, n_tok, max_ctx);
        dprof_end(DP_S_SOFTMAX, s);
        dprof_begin(DP_S_CONTEXT, s);
        DPCACHE(n_tok, Hh / HG);
        k_context_part<HG, NT><<<dim3(Hh / HG, NT), Lk, 0, s>>>(cpart, scores, cache, n_tok, max_ctx);
        KCHK("k_context_part");
        k_context_reduce<NT><<<Hh * Lk / 256, 256, 0, s>>>(ctx, cpart);
        KCHK("k_context_reduce");
        dprof_end(DP_S_CONTEXT, s);
    } else {
        // The indexer consumes the SAME q_resid the attention does — it is the q-side LoRA output,
        // not a separate projection. Computing it twice would waste a gemv and give the two a way
        // to drift apart.
        indexer_select(x, q_resid, IW, IS, t, sel, nsel, iws, s);
        constexpr int NB = (IDX_OUT_WIDTH + TT - 1) / TT;        // worst-case grid, count on device
        dprof_begin(DP_S_SCORES, s);
        DPCACHE(n_tok < IDX_OUT_WIDTH ? n_tok : IDX_OUT_WIDTH, 1);
        k_scores_sel<TT><<<NB, TT * 32, 0, s>>>(scores, qa, cache, sel, nsel, max_ctx, scaling);
        dprof_end(DP_S_SCORES, s);
        dprof_begin(DP_S_SOFTMAX, s);
        k_softmax_dev<256><<<Hh, 256, 0, s>>>(scores, nsel, max_ctx);
        dprof_end(DP_S_SOFTMAX, s);
        dprof_begin(DP_S_CONTEXT, s);
        DPCACHE(n_tok < IDX_OUT_WIDTH ? n_tok : IDX_OUT_WIDTH, Hh / HG);
        k_context_sel_part<HG, NT><<<dim3(Hh / HG, NT), Lk, 0, s>>>(cpart, scores, cache, sel, nsel, max_ctx);
        KCHK("k_context_sel_part");
        k_context_reduce<NT><<<Hh * Lk / 256, 256, 0, s>>>(ctx, cpart);
        KCHK("k_context_reduce");
        dprof_end(DP_S_CONTEXT, s);
    }
    dprof_end(DP_M_SDPA, s);

    dprof_begin(DP_M_OPROJ, s);
    dprof_bytes((double)Hh * Dv * Lk * 2.0);
    k_expand_v<128><<<Hh * Dv, 128, 0, s>>>(heads, ctx, (const __nv_bfloat16*)W.kv_b);
    gemv(y, W.o_proj, heads, HIDDEN, Hh * Dv, W.dtype, s);
    dprof_end(DP_M_OPROJ, s);
}

void mla_batch_step_dsa(const float* x, const MlaWeights& W, const IndexerWeights& IW,
                        IndexerState& IS, float* cache, int pos0, int M, int max_ctx,
                        int32_t* sel, int32_t* nsel, float* y, float* ws, float* iws,
                        cudaStream_t s, bool force_sparse) {
    float* q_resid = ws;
    float* q       = q_resid + (size_t)M * MLA_Q_LORA;
    float* c_new   = q + (size_t)M * MLA_Q_DIM;
    float* heads   = c_new + (size_t)M * Lk;
    float* qa      = heads + (size_t)M * Hh * Dv;               // [M, 64, 512]
    float* scores  = qa + (size_t)M * Hh * Lk;                  // [64, max_ctx]
    float* ctx     = scores + (size_t)Hh * max_ctx;             // [M, 64, 512]
    float* cpart   = ctx + (size_t)M * Hh * Lk;                 // [NT, 64, 512], reused per token
    const float scaling = rsqrtf((float)(MLA_QK_NOPE + MLA_QK_ROPE));
    const int MG = (M + MLA_MB - 1) / MLA_MB;

    // The sub-phase marks mirror mla_decode_step_dsa's EXACTLY, including which kernels go in
    // which bucket, so the prefill and decode tables can be read against each other. That is the
    // whole reason to have them: they are what showed absorb_q and expand_v running M times per
    // layer, each streaming the whole of kv_b, which is what the two batched kernels below fix.
    //
    // Marks around a per-token kernel still open and close M times and dprof sums them, so the
    // `calls` column remains the measurement: a row whose call count scales with M is a row that
    // did not get batched. After this change only indexer and sdpa should read M.
    dprof_begin(DP_M_QPROJ, s);
    gemm(q_resid, W.q_a, x, M, MLA_Q_LORA, HIDDEN, W.dtype, s);
    for (int m = 0; m < M; ++m)
        rmsnorm(q_resid + (size_t)m * MLA_Q_LORA, q_resid + (size_t)m * MLA_Q_LORA,
                W.q_a_norm, W.dtype, MLA_Q_LORA, s);
    gemm(q, W.q_b, q_resid, M, MLA_Q_DIM, MLA_Q_LORA, W.dtype, s);
    dprof_end(DP_M_QPROJ, s);

    dprof_begin(DP_M_KV, s);
    gemm(c_new, W.kv_a, x, M, Lk, HIDDEN, W.dtype, s);
    for (int m = 0; m < M; ++m)
        rmsnorm(c_new + (size_t)m * Lk, c_new + (size_t)m * Lk, W.kv_a_norm, W.dtype, Lk, s);
    for (int m = 0; m < M; ++m)
        k_store_latent<<<(Lk + 255) / 256, 256, 0, s>>>(cache, c_new + (size_t)m * Lk, pos0 + m);
    dprof_end(DP_M_KV, s);

    // Absorb for all M at once. Safe to hoist above the indexer: k_absorb_q reads only q and kv_b,
    // and the indexer's pool state is untouched by it.
    dprof_begin(DP_M_ABSORB, s);
    dprof_bytes((double)MG * Hh * Dq * Lk * 2.0);              // W_k half of kv_b, once per chunk
    k_absorb_q_batch<MLA_MB><<<dim3(Hh, MG), Lk, 0, s>>>(qa, q, (const __nv_bfloat16*)W.kv_b, M);
    dprof_end(DP_M_ABSORB, s);

    constexpr int TT = 8;
    constexpr int HG = MLA_HG;
    constexpr int NT = MLA_NT;
    for (int m = 0; m < M; ++m) {
        const int t = pos0 + m, n_tok = t + 1;
        float* qa_m  = qa  + (size_t)m * Hh * Lk;
        float* ctx_m = ctx + (size_t)m * Hh * Lk;
        // Pool state is incremental and must advance for EVERY token, dense branch or not.
        dprof_begin(DP_M_INDEXER, s);
        indexer_keys(x + (size_t)m * HIDDEN, IW, IS, t, iws, s);
        dprof_end(DP_M_INDEXER, s);

        dprof_begin(DP_M_SDPA, s);
        if (n_tok <= DENSE_CTX_LIMIT && !force_sparse) {
            dprof_begin(DP_S_SCORES, s);
            DPCACHE(n_tok, 1);
        k_scores<TT><<<(n_tok + TT - 1) / TT, TT * 32, 0, s>>>(scores, qa_m, cache, n_tok, max_ctx, scaling);
            dprof_end(DP_S_SCORES, s);
            dprof_begin(DP_S_SOFTMAX, s);
            k_softmax<256><<<Hh, 256, 0, s>>>(scores, n_tok, max_ctx);
            dprof_end(DP_S_SOFTMAX, s);
            dprof_begin(DP_S_CONTEXT, s);
            DPCACHE(n_tok, Hh / HG);
        k_context_part<HG, NT><<<dim3(Hh / HG, NT), Lk, 0, s>>>(cpart, scores, cache, n_tok, max_ctx);
            KCHK("k_context_part");
            k_context_reduce<NT><<<Hh * Lk / 256, 256, 0, s>>>(ctx_m, cpart);
            KCHK("k_context_reduce");
            dprof_end(DP_S_CONTEXT, s);
        } else {
            indexer_select(x + (size_t)m * HIDDEN, q_resid + (size_t)m * MLA_Q_LORA, IW, IS, t,
                           sel, nsel, iws, s);
            constexpr int NB = (IDX_OUT_WIDTH + TT - 1) / TT;
            dprof_begin(DP_S_SCORES, s);
            DPCACHE(n_tok < IDX_OUT_WIDTH ? n_tok : IDX_OUT_WIDTH, 1);
        k_scores_sel<TT><<<NB, TT * 32, 0, s>>>(scores, qa_m, cache, sel, nsel, max_ctx, scaling);
            dprof_end(DP_S_SCORES, s);
            dprof_begin(DP_S_SOFTMAX, s);
            k_softmax_dev<256><<<Hh, 256, 0, s>>>(scores, nsel, max_ctx);
            dprof_end(DP_S_SOFTMAX, s);
            dprof_begin(DP_S_CONTEXT, s);
            DPCACHE(n_tok < IDX_OUT_WIDTH ? n_tok : IDX_OUT_WIDTH, Hh / HG);
        k_context_sel_part<HG, NT><<<dim3(Hh / HG, NT), Lk, 0, s>>>(cpart, scores, cache, sel, nsel, max_ctx);
            KCHK("k_context_sel_part");
            k_context_reduce<NT><<<Hh * Lk / 256, 256, 0, s>>>(ctx_m, cpart);
            KCHK("k_context_reduce");
            dprof_end(DP_S_CONTEXT, s);
        }
        dprof_end(DP_M_SDPA, s);
    }

    // expand_v sits with o_proj because that is where mla_decode_step_dsa puts it.
    dprof_begin(DP_M_OPROJ, s);
    dprof_bytes((double)MG * Hh * Dv * Lk * 2.0);              // W_v half of kv_b
    k_expand_v_batch<128, MLA_MB><<<dim3(Hh * Dv, MG), 128, 0, s>>>(
        heads, ctx, (const __nv_bfloat16*)W.kv_b, M);
    gemm(y, W.o_proj, heads, M, HIDDEN, Hh * Dv, W.dtype, s);
    dprof_end(DP_M_OPROJ, s);
}

}  // namespace glm5
