// mla_context.cuh — the k_context family, shared by kernels/mla.cu and tools/bench_context.cu.
//
// EXTRACTED SO THE MICROBENCH MEASURES THE SHIPPED KERNEL. The (HG, NT) optimum was not
// predictable from the byte model -- HG=16/NT=8 has more blocks AND 4x less cache traffic than
// HG=4/NT=1 and is still slower -- and finding it by rebuilding the engine costs ~10 minutes per
// point, most of it loading 101 GiB. A microbench on synthetic buffers sweeps the whole grid in
// one run, but only if it is timing THIS code rather than a copy that can drift from it.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "glm5_config.h"

namespace glm5 {
namespace ctxk {
static constexpr int Hh = MLA_HEADS;      // 64
static constexpr int Lk = MLA_KV_LORA;    // 512
}

// ctx[h][l] = sum_t a[h][t] * C[t][l], computed in NT disjoint t-tiles.
//
// One block per (head group, t-tile), 512 threads (one per latent lane). C[t][l] is read once per
// head-group rather than once per head, and the t-tiles between them read it once in total.
// A tile whose range starts past the end writes zeros -- see the MLA_NT note on why that is
// deliberate rather than a host-side grid trim.
using ctxk::Hh; using ctxk::Lk;

template <int HG, int NT>
__global__ void k_context_part(float* __restrict__ part, const float* __restrict__ s,
                               const float* __restrict__ cache, int n_tok, int max_ctx) {
    const int h0 = blockIdx.x * HG, l = threadIdx.x, tile = blockIdx.y;
    const int per = (n_tok + NT - 1) / NT;
    const int t0 = tile * per;
    const int t1 = (n_tok < t0 + per) ? n_tok : t0 + per;
    float acc[HG];
    #pragma unroll
    for (int i = 0; i < HG; ++i) acc[i] = 0.f;
    for (int t = t0; t < t1; ++t) {
        const float c = cache[(size_t)t * Lk + l];       // coalesced across threads
        #pragma unroll
        for (int i = 0; i < HG; ++i) acc[i] += s[(size_t)(h0 + i) * max_ctx + t] * c;
    }
    #pragma unroll
    for (int i = 0; i < HG; ++i) part[((size_t)tile * Hh + h0 + i) * Lk + l] = acc[i];
}

// Sparse twin. Same partition arithmetic on the DEVICE-side count, so that below DENSE_CTX_LIMIT
// -- where n_sel == n_tok and sel[i] == i -- it tiles t exactly as the dense kernel does and the
// two remain bit-identical.
template <int HG, int NT>
__global__ void k_context_sel_part(float* __restrict__ part, const float* __restrict__ s,
                                   const float* __restrict__ cache, const int32_t* __restrict__ sel,
                                   const int32_t* __restrict__ n_ptr, int max_ctx) {
    const int n_sel = *n_ptr;
    const int h0 = blockIdx.x * HG, l = threadIdx.x, tile = blockIdx.y;
    const int per = (n_sel + NT - 1) / NT;
    const int i0 = tile * per;
    const int i1 = (n_sel < i0 + per) ? n_sel : i0 + per;
    float acc[HG];
    #pragma unroll
    for (int i = 0; i < HG; ++i) acc[i] = 0.f;
    for (int i = i0; i < i1; ++i) {
        const int t = sel[i];
        if (t < 0) continue;
        const float c = cache[(size_t)t * Lk + l];
        #pragma unroll
        for (int j = 0; j < HG; ++j) acc[j] += s[(size_t)(h0 + j) * max_ctx + i] * c;
    }
    #pragma unroll
    for (int i = 0; i < HG; ++i) part[((size_t)tile * Hh + h0 + i) * Lk + l] = acc[i];
}

// Sum the NT partials. 32768 outputs, so this is a rounding error next to the pass that made them.
template <int NT>
__global__ void k_context_reduce(float* __restrict__ ctx, const float* __restrict__ part) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.f;
    #pragma unroll
    for (int k = 0; k < NT; ++k) v += part[(size_t)k * Hh * Lk + idx];
    ctx[idx] = v;
}

}  // namespace glm5
