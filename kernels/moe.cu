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
#include "moe.h"
#include "gemv.h"
#include "glm5_config.h"
#include <cstdio>
#include <cstdint>

namespace glm5 {

static constexpr int H  = HIDDEN;
static constexpr int I  = MOE_INTER;
static constexpr int KS = N_EXPERT_PER_TOK;      // 8 routed slots
static constexpr int NS = KS + 1;                // + the shared expert, as slot 8

// ---- fp8-e4m3 -> float. Bias 7, 3 mantissa bits, no infinities. ------------------------------
__device__ __forceinline__ float fp8e4m3(uint8_t v) {
    const int s = v >> 7, e = (v >> 3) & 0xF, m = v & 0x7;
    const float sign = s ? -1.f : 1.f;
    if (e == 0) return sign * (float)m * (1.f / 8.f) * (1.f / 64.f);       // subnormal, 2^-6
    // 2^(e-7) built exactly from the exponent field rather than through exp2f
    return sign * (1.f + (float)m * (1.f / 8.f)) * __int_as_float((e - 7 + 127) << 23);
}

// ---- e2m1 (fp4) magnitude LUT. Low 3 bits index it; bit 3 is the sign. ------------------------
__device__ __constant__ float kE2M1[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
__device__ __forceinline__ float fp4(int nib) {
    return kE2M1[nib & 7] * ((nib & 8) ? -1.f : 1.f);
}

// Dot one NVFP4 row of `in` elements with x. Block-wide; the result is valid in thread 0.
//
// Each thread owns whole 16-element groups, so one fp8 scale covers exactly one 8-byte span and
// no thread ever straddles a scale boundary.
//
// ALIGNMENT (this cost a fault, and would have cost a silent one on a different shard layout).
// The group is 8 bytes, but it is read as TWO uint32 loads, not one uint2. safetensors aligns
// tensors to 4 bytes, not 8: in this checkpoint 777 of 1671 `weight_packed` tensors per shard sit
// at offset 4 mod 8 (e.g. layer 45 expert 0 down_proj at blob offset 2691195540). A uint2 load on
// those faults with "misaligned address". nvfp4_check_align() asserts the 4-byte floor at load
// time so a future checkpoint that breaks even that is caught at startup, not mid-request.
template <int BS>
__device__ __forceinline__ float nvfp4_row_dot(const uint8_t* __restrict__ packed,
                                               const uint8_t* __restrict__ scale,
                                               float inv_gs, const float* __restrict__ x,
                                               int in, float* red) {
    const int ngroup = in >> 4;
    const uint32_t* pk = reinterpret_cast<const uint32_t*>(packed);
    float acc = 0.f;
    for (int g = threadIdx.x; g < ngroup; g += BS) {
        const uint32_t raw_x = pk[g * 2], raw_y = pk[g * 2 + 1];
        const float sc = fp8e4m3(scale[g]) * inv_gs;
        const float* xp = x + (g << 4);
        float d = 0.f;
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int byte = (raw_x >> (b * 8)) & 0xFF;
            d += fp4(byte & 0xF) * xp[b * 2] + fp4(byte >> 4) * xp[b * 2 + 1];
        }
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int byte = (raw_y >> (b * 8)) & 0xFF;
            d += fp4(byte & 0xF) * xp[8 + b * 2] + fp4(byte >> 4) * xp[8 + b * 2 + 1];
        }
        acc += d * sc;
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    float t = 0.f;
    if (threadIdx.x == 0)
        for (int w = 0; w < BS / 32; ++w) t += red[w];
    return t;
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
// One block per (intermediate row, slot). Both gate_proj and up_proj rows for the same output
// index are consumed here, so the SwiGLU product never round-trips through memory.
//
// Clamp semantics are asymmetric, straight from the reference:
//   gate = min(gate, +limit)          (no lower clamp)
//   up   = clamp(up, -limit, +limit)
template <int BS>
__global__ void k_expert_act(float* __restrict__ act, const float* __restrict__ x,
                             const Nvfp4Mat* __restrict__ experts,
                             const Nvfp4Mat* __restrict__ shared,
                             const int32_t* __restrict__ sel, int inter, int hid, float limit) {
    __shared__ float red[BS / 32];
    const int o = blockIdx.x, slot = blockIdx.y;
    const Nvfp4Mat* M = (slot < KS) ? (experts + (size_t)sel[slot] * 3) : shared;

    const size_t prow = (size_t)o * (hid >> 1), srow = (size_t)o * (hid >> 4);
    float g = nvfp4_row_dot<BS>(M[0].packed + prow, M[0].scale + srow, 1.f / M[0].gscale[0], x, hid, red);
    __syncthreads();
    float u = nvfp4_row_dot<BS>(M[1].packed + prow, M[1].scale + srow, 1.f / M[1].gscale[0], x, hid, red);
    if (threadIdx.x == 0) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
        act[(size_t)slot * inter + o] = (g / (1.f + __expf(-g))) * u;
    }
}

// ---- expert down_proj, weighted accumulate ----------------------------------------------------
// One block per hidden element, looping the 9 slots. Deterministic: no atomics, and slots are
// summed in a fixed order every run, so two identical requests produce identical logits.
template <int BS>
__global__ void k_expert_down(float* __restrict__ y, const float* __restrict__ act,
                              const Nvfp4Mat* __restrict__ experts,
                              const Nvfp4Mat* __restrict__ shared,
                              const int32_t* __restrict__ sel, const float* __restrict__ wts,
                              int inter, int hid) {
    __shared__ float red[BS / 32];
    __shared__ float total;
    const int h = blockIdx.x;
    if (threadIdx.x == 0) total = 0.f;
    __syncthreads();
    for (int slot = 0; slot < NS; ++slot) {
        const Nvfp4Mat* M = (slot < KS) ? (experts + (size_t)sel[slot] * 3) : shared;
        const float v = nvfp4_row_dot<BS>(M[2].packed + (size_t)h * (inter >> 1),
                                          M[2].scale + (size_t)h * (inter >> 4),
                                          1.f / M[2].gscale[0], act + (size_t)slot * inter, inter, red);
        if (threadIdx.x == 0) total += v * ((slot < KS) ? wts[slot] : 1.f);
        __syncthreads();
    }
    if (threadIdx.x == 0) y[h] = total;
}

// ---- entry points ------------------------------------------------------------------------------
size_t moe_workspace_floats() { return (size_t)N_ROUTED_EXPERT + (size_t)NS * I; }

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
    constexpr int BS = 128;
    float* logits = ws;
    float* act    = ws + N_ROUTED_EXPERT;
    moe_route(x, L, sel, wts, logits, s);
    k_expert_act<BS><<<dim3(I, NS), BS, 0, s>>>(act, x, L.experts, L.shared, sel, I, H, SWIGLU_LIMIT);
    k_expert_down<BS><<<H, BS, 0, s>>>(y, act, L.experts, L.shared, sel, wts, I, H);
}

}  // namespace glm5
