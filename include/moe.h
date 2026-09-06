// moe.h — MoE block: noaux_tc sigmoid router + NVFP4 experts + shared expert.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "gemv.h"

namespace glm5 {

// One NVFP4 matrix as the checkpoint stores it.
//   packed [out, in/2]  uint8, LOW nibble first
//   scale  [out, in/16] fp8-e4m3
//   gscale [1]          fp32;  w = kE2M1[nib&7] * (-1)^(nib>>3) * (fp8(scale) / gscale)
struct Nvfp4Mat {
    const uint8_t* packed;
    const uint8_t* scale;
    const float*   gscale;
};

struct MoeLayer {
    WRef         router_w;      // bf16 [E, HIDDEN]
    const float* router_bias;   // fp32 [E]  e_score_correction_bias
    const Nvfp4Mat* experts;    // device array, 3*E entries: [e*3+0]=gate, +1=up, +2=down
    const Nvfp4Mat* shared;     // device array, 3 entries: gate, up, down
    int n_expert;
    int topk;
};

size_t moe_workspace_floats();
bool nvfp4_check_align(const Nvfp4Mat& m, const char* what);

// x [HIDDEN] fp32 -> y [HIDDEN] fp32. `sel` [topk] int32 and `wts` [topk] fp32 are written out so
// the caller (and the gate) can see the routing decision.
void moe_forward(const float* x, const MoeLayer& L, float* y, int32_t* sel, float* wts,
                 float* ws, cudaStream_t s);
void moe_route(const float* x, const MoeLayer& L, int32_t* sel, float* wts, float* logits,
               cudaStream_t s);
}  // namespace glm5
