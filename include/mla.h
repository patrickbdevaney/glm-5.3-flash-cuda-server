// mla.h — MLA full-attention decode (11 of 45 layers). Pure NoPE: no rotary anywhere.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace glm5 {

struct MlaWeights {
    const void* q_a;         // bf16 [1536, 4096]
    const void* q_a_norm;    // bf16 [1536]
    const void* q_b;         // bf16 [16384, 1536]
    const void* kv_a;        // bf16 [512, 4096]
    const void* kv_a_norm;   // bf16 [512]
    const void* kv_b;        // bf16 [32768, 512]  rows: h*512 + (0..255 = W_k, 256..511 = W_v)
    const void* o_proj;      // bf16 [4096, 16384]
    int dtype;
};

size_t mla_workspace_floats(int max_ctx);

// One decode step. `cache` is the 512-wide MLA latent cache [max_ctx, 512] fp32; the new token's
// latent is appended at row `t` (0-based), and attention runs over rows 0..t inclusive.
//
// Dense causal attention. That is EXACT, not an approximation, while t+1 <= IDX_TOPK: the DSA
// indexer pools keys in groups of 4 and selects min(2048/4, n_pools) of them, so below 2048
// tokens every pool is selected and the tail is appended, covering every visible position.
// ref/gen_mla.py verifies this against the real indexer rather than asserting it.
void mla_decode_step(const float* x, const MlaWeights& W, float* cache, int t, int max_ctx,
                     float* y, float* ws, cudaStream_t s);

}  // namespace glm5
