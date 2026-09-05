// moe.cu — MoE block for GLM-5.3-Flash. 144 routed experts (REAP-50), 8 per token, 1 shared.
//
// 28.9% of B_tok (routed 24.1 + shared 3.0 + router 1.8). Unlike the KDA layers this is *sparse*
// traffic: 9 of 145 expert matrices are touched per token, so the kernel's job is to read those
// nine at full bandwidth and never materialise a dequantised copy of anything.
//
// Routing follows Glm5NextTextTopkRouter exactly, including two details that are easy to get
// subtly wrong and would show up only as a slow quality drift:
//   * selection ranks by (sigmoid(logits) + e_score_correction_bias), but the WEIGHTS gathered
//     are the UNBIASED sigmoid scores;
//   * n_group == topk_group == 1, so the group mask selects everything and is a no-op here.
//
// ---------------------------------------------------------------------------------------------
// KERNEL SHAPE (rewritten after OPTIMIZATION_LOG #9 profiled this block at 13% of achievable
// bandwidth while holding 66.5% of the decode step). Three things were wrong with the first
// version, and all three are the same mistake in different clothes: not enough bytes in flight.
//
//   1. BLOCK-per-output-row with a block-wide reduction. At hid=4096 and BS=128 each thread ran
//      TWO loop iterations and then paid a 5-step shuffle, a shared-memory round trip and a
//      __syncthreads. The reduction cost more than the work it reduced.
//   2. A __constant__ memory LUT for the e2m1 nibble. Constant memory broadcasts only when every
//      lane in the warp reads the SAME address; here every lane reads a different one, so each of
//      the 16 lookups per group serialised up to 8 ways. This, not the loads, was the stall.
//   3. ILP=1. Each load was consumed immediately. The 0731 engine measured that pattern at
//      110-132 GB/s on this box against 224-237 GB/s at ILP>=2 (its OPTIMIZATION_LOG #7).
//
// The rewrite is WARP-per-output-row (shuffle reduce only, no shared, no syncthreads), the
// hardware `cvt.rn.f16x2.e2m1x2` unpack via __nv_cvt_fp4x2_to_halfraw2, and a 4-deep unroll that
// issues eight independent weight loads before consuming any of them.
//
// NUMERICS ARE UNCHANGED, deliberately. The hardware unpack returns exactly the same eight values
// the LUT held ({0,.5,1,1.5,2,3,4,6}, all exact in fp16 and in fp32), and the accumulation stays
// fp32 against fp32 activations. The 0731 engine's equivalent change also switched to half2
// accumulation and bought 2.59x -- but it cost that engine its draft-head acceptance (3.12 -> 1.00
// tokens/verify, its OPTIMIZATION_LOG #9). We have an MTP head at 72.3% acceptance to protect, so
// the fp32 accumulate stays and only the decode path changes.
#include "moe.h"
#include "dprof.h"
#include "gemv.h"
#include "glm5_config.h"
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace glm5 {

static constexpr int H  = HIDDEN;
static constexpr int I  = MOE_INTER;
static constexpr int KS = N_EXPERT_PER_TOK;      // 8 routed slots
static constexpr int NS = KS + 1;                // + the shared expert, as slot 8

// ---- fp8-e4m3 -> float. Bias 7, 3 mantissa bits, no infinities. ------------------------------
// Still here for the 256-entry LUT build; it is evaluated once per block, not per element.
__device__ __forceinline__ float fp8e4m3(uint8_t v) {
    const int s = v >> 7, e = (v >> 3) & 0xF, m = v & 0x7;
    const float sign = s ? -1.f : 1.f;
    if (e == 0) return sign * (float)m * (1.f / 8.f) * (1.f / 64.f);       // subnormal, 2^-6
    // 2^(e-7) built exactly from the exponent field rather than through exp2f
    return sign * (1.f + (float)m * (1.f / 8.f)) * __int_as_float((e - 7 + 127) << 23);
}

// ---- e2m1 (fp4) x2 -> two floats, in ONE hardware instruction --------------------------------
// cvt.rn.f16x2.e2m1x2. Low nibble becomes .x, high nibble .y -- the same order the checkpoint
// packs in, and the same order the old scalar LUT consumed. Probed working on sm_110a
// (dspark-cuda-reap-finetune/DECODE_GAP_RESEARCH.md: the FP4 *unpack* is exposed on Thor even
// though the FP4 *mma* is not).
__device__ __forceinline__ float2 e2m1x2(unsigned char b) {
    __half2_raw r = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b, __NV_E2M1);
    __half2 h = *reinterpret_cast<__half2*>(&r);
    return make_float2(__low2float(h), __high2float(h));
}

// 8 packed e2m1 codes (one uint32) against 8 fp32 activations (TWO float4 loads, not eight
// scalars -- the activation is fp32 here and at 4 bytes each it is the wider operand).
__device__ __forceinline__ float dot8(const float* __restrict__ a, unsigned pv) {
    const float4 x0 = *(const float4*)(a), x1 = *(const float4*)(a + 4);
    const float2 w0 = e2m1x2((pv      ) & 0xff), w1 = e2m1x2((pv >>  8) & 0xff);
    const float2 w2 = e2m1x2((pv >> 16) & 0xff), w3 = e2m1x2((pv >> 24) & 0xff);
    float s = 0.f;
    s = fmaf(x0.x, w0.x, s); s = fmaf(x0.y, w0.y, s);
    s = fmaf(x0.z, w1.x, s); s = fmaf(x0.w, w1.y, s);
    s = fmaf(x1.x, w2.x, s); s = fmaf(x1.y, w2.y, s);
    s = fmaf(x1.z, w3.x, s); s = fmaf(x1.w, w3.y, s);
    return s;
}

// The same dot, with the activation held as fp16 and the MACs done as __hfma2.
//
// HALF THE INSTRUCTIONS, NOT HALF THE BYTES. This kernel is instruction-bound (see the LUT note
// above): per 8 codes the fp32 path issues 4 unpacks, 8 half->float conversions, 2 shared float4
// loads and 8 fmaf, and the fp16 path issues 4 unpacks, 1 shared float4 load and 4 __hfma2 --
// the e2m1 unpack already PRODUCES a half2, so consuming it as one removes the conversion
// entirely, and the staged activation halves in size so its bank conflicts halve too.
//
// It is OFF by default because it is not the same function. The accumulate inside each group of
// 8 becomes fp16 (fp32 across groups), which the 0731 engine measured at cosine 0.9999999 /
// rms_rel 4.04e-04 -- and which cost that engine its draft-head acceptance, 3.12 -> 1.00
// tokens/verify (its OPTIMIZATION_LOG #9). We have an MTP head at 72.3% acceptance that has
// never been fine-tuned against a perturbed target, so this stays behind GLM5_MOE_HALF=1 until
// somebody measures acceptance with it on. Base-model quality is not the risk here; the draft is.
__device__ __forceinline__ float dot8h(const __half* __restrict__ a, unsigned pv) {
    const float4 raw = *(const float4*)(a);              // 8 halves = one 16-byte shared load
    const __half2* xh = reinterpret_cast<const __half2*>(&raw);
    __half2 acc = __float2half2_rn(0.f);
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        __half2_raw wr = __nv_cvt_fp4x2_to_halfraw2(
            (__nv_fp4x2_storage_t)((pv >> (b * 8)) & 0xff), __NV_E2M1);
        acc = __hfma2(*reinterpret_cast<__half2*>(&wr), xh[b], acc);
    }
    return __half2float(__low2half(acc)) + __half2float(__high2half(acc));
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// ---- routing ---------------------------------------------------------------------------------
// Sigmoid in parallel, selection on one thread. At E=144, k=8 the selection is 1152 comparisons —
// utterly free next to the [144, 4096] router GEMV that precedes it — and doing it serially makes
// it exactly reproducible, including tie-breaking, which a parallel argmax reduction is not.
__global__ void k_route(int32_t* __restrict__ sel, float* __restrict__ wts,
                        const float* __restrict__ logits, const float* __restrict__ bias,
                        int E, int K, float scaling, bool norm_prob) {
    extern __shared__ float sm[];
    float* score  = sm;             // [E] unbiased sigmoid — these become the weights
    float* choice = sm + E;         // [E] biased — these decide the ranking
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        const float s = 1.f / (1.f + __expf(-logits[i]));
        score[i] = s;
        choice[i] = s + bias[i];
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    for (int k = 0; k < K; ++k) {
        int bi = -1; float bv = -1e30f;
        for (int i = 0; i < E; ++i) if (choice[i] > bv) { bv = choice[i]; bi = i; }
        sel[k] = bi;
        wts[k] = score[bi];
        choice[bi] = -1e30f;
    }
    float d = 1.f;
    if (norm_prob) { d = 1e-20f; for (int k = 0; k < K; ++k) d += wts[k]; }
    for (int k = 0; k < K; ++k) wts[k] = wts[k] / d * scaling;
}

// ---- expert gate/up + SwiGLU, fused ----------------------------------------------------------
// One WARP per (intermediate row, slot). Both gate_proj and up_proj rows for the same output index
// are consumed here, so the SwiGLU product never round-trips through memory AND the two rows'
// loads interleave -- eight independent misses in flight per unrolled step instead of one.
//
// Clamp semantics are asymmetric, straight from the reference:
//   gate = min(gate, +limit)          (no lower clamp)
//   up   = clamp(up, -limit, +limit)
//
// ALIGNMENT (this cost a fault, and would have cost a silent one on a different shard layout).
// Weights are read as uint32, never uint2/uint4: safetensors aligns tensors to 4 bytes, not 8, and
// in this checkpoint 777 of 1671 `weight_packed` tensors per shard sit at offset 4 mod 8 (e.g.
// layer 45 expert 0 down_proj at blob offset 2691195540). A uint2 load on those faults with
// "misaligned address". Row strides (hid/2 = 2048 B, inter/2 = 1024 B) and the lane offset
// (lane*8 codes = lane*4 bytes) are all multiples of 4, so uint32 is exactly the widest legal
// load. nvfp4_check_align() asserts the 4-byte floor at load time so a future checkpoint that
// breaks even that is caught at startup, not mid-request.
template <int WPB, bool HALF>
__global__ void k_expert_act(float* __restrict__ act, const float* __restrict__ x,
                             const Nvfp4Mat* __restrict__ experts,
                             const Nvfp4Mat* __restrict__ shared,
                             const int32_t* __restrict__ sel, int inter, int hid, float limit) {
    // The activation is staged in shared ONCE per block. Without this, each of the WPB warps
    // re-reads all of x from L1 for both its gate and its up row: at fp32 that is 32 bytes of
    // activation fetched per 4 bytes of weight, and it was what held this kernel to 134 GB/s
    // after the first rewrite while the KDA gemvs next door ran at 200. Staging is numerically
    // free -- same values, same order, same fp32 accumulate.
    // The scale LUT stays in SHARED, not inlined. Decoding e4m3 arithmetically per use looked
    // like the obvious win -- six ALU ops and no memory, against a table whose index is the scale
    // BYTE VALUE and therefore effectively random across a warp. Measured, it lost: w13+act went
    // 535 -> 637 ms. This kernel is instruction-bound, not shared-bandwidth-bound, so trading a
    // conflicted LDS for eight more ALU ops is the wrong direction.
    __shared__ float lut[256];      // e4m3 code -> float, built once per block
    extern __shared__ __align__(16) char smem_act[];
    float*  xs = reinterpret_cast<float*>(smem_act);       // [hid]      when !HALF
    __half* xh = reinterpret_cast<__half*>(smem_act);      // [hid]      when  HALF
    for (int i = threadIdx.x; i < 256; i += WPB * 32) lut[i] = fp8e4m3((uint8_t)i);
    if constexpr (HALF)
        for (int i = threadIdx.x; i < hid; i += WPB * 32) xh[i] = __float2half(x[i]);
    else
        for (int i = threadIdx.x; i < (hid >> 2); i += WPB * 32)
            reinterpret_cast<float4*>(xs)[i] = reinterpret_cast<const float4*>(x)[i];
    __syncthreads();

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int o = blockIdx.x * WPB + warp, slot = blockIdx.y;
    if (o >= inter) return;
    const Nvfp4Mat* M = (slot < KS) ? (experts + (size_t)sel[slot] * 3) : shared;

    const uint8_t* Pg = M[0].packed + (size_t)o * (hid >> 1);
    const uint8_t* Sg = M[0].scale  + (size_t)o * (hid >> 4);
    const uint8_t* Pu = M[1].packed + (size_t)o * (hid >> 1);
    const uint8_t* Su = M[1].scale  + (size_t)o * (hid >> 4);
    const float igg = 1.f / M[0].gscale[0], igu = 1.f / M[1].gscale[0];

    float ag = 0.f, au = 0.f;
    // 1024 K-values per iteration = 4 steps x 32 lanes x 8 codes. hid is 4096 here; the tail loop
    // below covers any hid that is merely a multiple of 256.
    const int K4 = hid & ~1023;
    int base = 0;
    for (; base < K4; base += 1024) {
        int k[4]; unsigned pg[4], pu[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            k[u] = base + (u << 8) + (lane << 3);
            pg[u] = __ldcs((const unsigned*)(Pg + (k[u] >> 1)));   // streaming: evict first,
            pu[u] = __ldcs((const unsigned*)(Pu + (k[u] >> 1)));   // the weights are never reused
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float dg = HALF ? dot8h(xh + k[u], pg[u]) : dot8(xs + k[u], pg[u]);
            const float du = HALF ? dot8h(xh + k[u], pu[u]) : dot8(xs + k[u], pu[u]);
            ag = fmaf(dg, lut[Sg[k[u] >> 4]] * igg, ag);
            au = fmaf(du, lut[Su[k[u] >> 4]] * igu, au);
        }
    }
    for (; base < hid; base += 256) {
        const int k0 = base + (lane << 3);
        const unsigned wg = __ldcs((const unsigned*)(Pg + (k0 >> 1)));
        const unsigned wu = __ldcs((const unsigned*)(Pu + (k0 >> 1)));
        const float dg = HALF ? dot8h(xh + k0, wg) : dot8(xs + k0, wg);
        const float du = HALF ? dot8h(xh + k0, wu) : dot8(xs + k0, wu);
        ag = fmaf(dg, lut[Sg[k0 >> 4]] * igg, ag);
        au = fmaf(du, lut[Su[k0 >> 4]] * igu, au);
    }

    float g = warp_sum(ag), u = warp_sum(au);
    if (lane == 0) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
        act[(size_t)slot * inter + o] = (g / (1.f + __expf(-g))) * u;
    }
}

// ---- expert down_proj -------------------------------------------------------------------------
// One WARP per (hidden element, slot), writing a per-slot partial. The first version looped the 9
// slots INSIDE the block with a __syncthreads between each, which serialised nine dependent
// reduction chains and left the machine with 4096 blocks of work where it wanted 36864 warps.
// Splitting the slot into the grid and reducing afterwards costs one extra 147 KB round trip and
// buys 9x the memory-level parallelism.
//
// Determinism is preserved: the reduce below sums slots in a fixed order with no atomics, so two
// identical requests still produce identical logits.
template <int WPB, bool HALF>
__global__ void k_expert_down(float* __restrict__ part, const float* __restrict__ act,
                              const Nvfp4Mat* __restrict__ experts,
                              const Nvfp4Mat* __restrict__ shared,
                              const int32_t* __restrict__ sel, int inter, int hid) {
    __shared__ float lut[256];
    extern __shared__ __align__(16) char smem_down[];
    float*  a  = reinterpret_cast<float*>(smem_down);      // [inter]  this block's slot slice
    __half* ah = reinterpret_cast<__half*>(smem_down);
    const int slot = blockIdx.y;
    const float* asrc = act + (size_t)slot * inter;
    for (int i = threadIdx.x; i < 256; i += WPB * 32) lut[i] = fp8e4m3((uint8_t)i);
    if constexpr (HALF)
        for (int i = threadIdx.x; i < inter; i += WPB * 32) ah[i] = __float2half(asrc[i]);
    else
        for (int i = threadIdx.x; i < (inter >> 2); i += WPB * 32)
            reinterpret_cast<float4*>(a)[i] = reinterpret_cast<const float4*>(asrc)[i];
    __syncthreads();

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int h = blockIdx.x * WPB + warp;
    if (h >= hid) return;
    const Nvfp4Mat* M = (slot < KS) ? (experts + (size_t)sel[slot] * 3) : shared;

    const uint8_t* P = M[2].packed + (size_t)h * (inter >> 1);
    const uint8_t* S = M[2].scale  + (size_t)h * (inter >> 4);
    const float ig = 1.f / M[2].gscale[0];

    float acc = 0.f;
    const int K4 = inter & ~1023;
    int base = 0;
    for (; base < K4; base += 1024) {
        int k[4]; unsigned pv[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            k[u] = base + (u << 8) + (lane << 3);
            pv[u] = __ldcs((const unsigned*)(P + (k[u] >> 1)));
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float d = HALF ? dot8h(ah + k[u], pv[u]) : dot8(a + k[u], pv[u]);
            acc = fmaf(d, lut[S[k[u] >> 4]] * ig, acc);
        }
    }
    for (; base < inter; base += 256) {
        const int k0 = base + (lane << 3);
        const unsigned wv = __ldcs((const unsigned*)(P + (k0 >> 1)));
        const float d = HALF ? dot8h(ah + k0, wv) : dot8(a + k0, wv);
        acc = fmaf(d, lut[S[k0 >> 4]] * ig, acc);
    }

    acc = warp_sum(acc);
    if (lane == 0) part[(size_t)slot * hid + h] = acc;
}

// Fixed slot order, no atomics: bit-identical across runs.
__global__ void k_down_combine(float* __restrict__ y, const float* __restrict__ part,
                               const float* __restrict__ wts, int hid) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hid) return;
    float t = 0.f;
    #pragma unroll
    for (int slot = 0; slot < NS; ++slot)
        t += part[(size_t)slot * hid + h] * ((slot < KS) ? wts[slot] : 1.f);
    y[h] = t;
}

// ---- entry points ------------------------------------------------------------------------------
// logits[E] + act[NS*I] + part[NS*H]. The `part` buffer is the price of splitting the down_proj
// slot loop into the grid; at 147 KB it is noise next to the 5.4 GB/tok this block streams.
size_t moe_workspace_floats() {
    return (size_t)N_ROUTED_EXPERT + (size_t)NS * I + (size_t)NS * H;
}

// Fail loudly at load time rather than with a misaligned-address fault mid-request.
bool nvfp4_check_align(const Nvfp4Mat& m, const char* what) {
    if (((uintptr_t)m.packed & 3) || ((uintptr_t)m.gscale & 3)) {
        fprintf(stderr, "nvfp4: %s is not 4-byte aligned (packed=%p gscale=%p); "
                        "the row-dot kernel requires it\n", what, (const void*)m.packed,
                        (const void*)m.gscale);
        return false;
    }
    return true;
}

void moe_route(const float* x, const MoeLayer& L, int32_t* sel, float* wts, float* logits,
               cudaStream_t s) {
    // The reference computes router logits in fp32 from fp32-upcast weights; gemv does exactly that.
    gemv(logits, L.router_w, x, L.n_expert, HIDDEN, GEMV_BF16, s);
    k_route<<<1, 256, 2 * L.n_expert * sizeof(float), s>>>(
        sel, wts, logits, L.router_bias, L.n_expert, L.topk, ROUTED_SCALE, NORM_TOPK_PROB);
}

void moe_forward(const float* x, const MoeLayer& L, float* y, int32_t* sel, float* wts,
                 float* ws, cudaStream_t s) {
    constexpr int WPB = 8;                       // 8 warps = 256 threads per block
    // Read once: getenv in a per-layer hot path would cost more than the kernel it selects.
    static const bool HALF = [] {
        const char* e = getenv("GLM5_MOE_HALF");
        return e && *e == '1';
    }();
    float* logits = ws;
    float* act    = ws + N_ROUTED_EXPERT;
    float* part   = act + (size_t)NS * I;
    dprof_begin(DP_E_ROUTER, s);
    moe_route(x, L, sel, wts, logits, s);
    dprof_end(DP_E_ROUTER, s);
    dprof_begin(DP_E_ACT, s);
    if (HALF)
        k_expert_act<WPB, true><<<dim3((I + WPB - 1) / WPB, NS), WPB * 32, H * sizeof(__half), s>>>(
            act, x, L.experts, L.shared, sel, I, H, SWIGLU_LIMIT);
    else
        k_expert_act<WPB, false><<<dim3((I + WPB - 1) / WPB, NS), WPB * 32, H * sizeof(float), s>>>(
            act, x, L.experts, L.shared, sel, I, H, SWIGLU_LIMIT);
    dprof_end(DP_E_ACT, s);
    dprof_begin(DP_E_DOWN, s);
    if (HALF)
        k_expert_down<WPB, true><<<dim3((H + WPB - 1) / WPB, NS), WPB * 32, I * sizeof(__half), s>>>(
            part, act, L.experts, L.shared, sel, I, H);
    else
        k_expert_down<WPB, false><<<dim3((H + WPB - 1) / WPB, NS), WPB * 32, I * sizeof(float), s>>>(
            part, act, L.experts, L.shared, sel, I, H);
    k_down_combine<<<(H + 255) / 256, 256, 0, s>>>(y, part, wts, H);
    dprof_end(DP_E_DOWN, s);
}

}  // namespace glm5
