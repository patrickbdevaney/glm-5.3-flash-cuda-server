// layer.h — hyper-connections, norms, dense MLP: everything a decoder layer needs besides
// attention and MoE.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "moe.h"

namespace glm5 {

// Manifold-constrained hyper-connections (mHC). Constants are identical to DeepSeek-V4-Flash:
// hc_mult 4, sinkhorn 20, fn [24, 16384], base [24], scale [3].
struct HcWeights {
    const void*  fn;      // bf16 [24, 16384]
    const float* base;    // fp32 [24]
    const float* scale;   // fp32 [3]
};

// streams [4, 4096] -> collapsed [4096], plus post [4] and comb [4,4] for the apply step.
void hc_compose(const float* streams, const HcWeights& W, float* collapsed, float* post,
                float* comb, float* ws, cudaStream_t s);

// streams = post[h] * sub[d] + (comb^T @ residual)[h][d]
void hc_apply(float* streams, const float* residual, const float* sub, const float* post,
              const float* comb, cudaStream_t s);

// Final stream collapse. GLM's HyperHead is an UNWEIGHTED MEAN (DeepSeek's was weighted).
void hc_head_mean(const float* streams, float* out, cudaStream_t s);

// Weighted RMSNorm over HIDDEN, fp32 accumulate.
void rmsnorm(float* y, const float* x, const void* w, int dtype, int n, cudaStream_t s);

// Dense MLP (layers 0..2): clamped SwiGLU, bf16 weights.
struct DenseMlp { WRef gate; WRef up; WRef down; int inter; int dtype; };
void dense_mlp(const float* x, const DenseMlp& M, float* y, float* ws, cudaStream_t s);
// x, y are [B, HIDDEN]. ws must be 2*B*inter floats.
void dense_mlp_batch(const float* x, const DenseMlp& M, float* y, float* ws, int B, cudaStream_t s);

size_t hc_workspace_floats();
size_t dense_mlp_workspace_floats();
size_t dense_mlp_batch_workspace_floats(int B);

}  // namespace glm5
