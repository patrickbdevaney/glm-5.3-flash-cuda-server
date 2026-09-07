// vision.h — the GLM-5.3 vision tower, 24-block ViT, in CUDA.
//
// The checkpoint ships a complete vision tower and this server was text-only: 347 tensors,
// model.visual.*, simply not loaded. The GGUF build has vision; this did not, and dropping vision
// capability is on the operator's escalate-first list, so it is a capability gap rather than a
// nice-to-have.
//
// SHAPE OF THE THING. One 448x448 image is 32x32 patches of 14 -> 1024 rows of
// 3*2*14*14 = 1176 (channels x temporal_patch x patch x patch, C-order). Those go through
// patch_embed to 1024-wide, 24 blocks, post_layernorm, a 2x2 spatial downsample to 256 rows of
// 4096, and the merger. The 256 rows of 4096 are what splice into the language model's token
// stream at the image_token_id positions.
//
// Everything here is bf16 weights on fp32 activations, matching the rest of the engine. The
// vision tower is NOT NVFP4 -- it is 0.5B params, 1.2 GiB, and quantising it would buy nothing
// on a path that runs once per image rather than once per token.
#pragma once
#include <cuda_runtime.h>
#include "gemv.h"

namespace glm5 {

// From config.json vision_config. Fixed at compile time because the tower's shape is a property
// of the checkpoint, not of a request.
constexpr int VIS_DEPTH      = 24;
constexpr int VIS_HIDDEN     = 1024;
constexpr int VIS_HEADS      = 16;
constexpr int VIS_HEAD_DIM   = VIS_HIDDEN / VIS_HEADS;      // 64
constexpr int VIS_INTER      = 4096;
constexpr int VIS_PATCH      = 14;
constexpr int VIS_TPATCH     = 2;
constexpr int VIS_CHAN       = 3;
constexpr int VIS_IN_DIM     = VIS_CHAN * VIS_TPATCH * VIS_PATCH * VIS_PATCH;   // 1176
constexpr int VIS_MERGE      = 2;
constexpr int VIS_OUT_HIDDEN = 4096;
constexpr int VIS_PROJ_INTER = 10240;
constexpr float VIS_EPS      = 1e-5f;
constexpr float VIS_SWIGLU_LIMIT = 10.0f;

struct VisionBlockWeights {
    const void* norm1;      // [1024]
    WRef        qkv;        // [3072, 1024]
    const void* qkv_b;      // [3072]
    const void* q_norm;     // [64]
    const void* k_norm;     // [64]
    WRef        proj;       // [1024, 1024]
    const void* proj_b;     // [1024]
    const void* norm2;      // [1024]
    WRef        gate;       const void* gate_b;   // [4096, 1024]
    WRef        up;         const void* up_b;
    WRef        down;       const void* down_b;   // [1024, 4096]
};

struct VisionWeights {
    WRef        patch_embed;      // [1024, 1176]
    const void* patch_embed_b;    // [1024]
    VisionBlockWeights blocks[VIS_DEPTH];
    const void* post_layernorm;   // [1024]
    WRef        downsample;       // Conv2d [4096, 1024, 2, 2] read as [4096, 4096]
    const void* downsample_b;     // [4096]
    // merger: proj -> LayerNorm(with bias) -> GELU -> clamped SwiGLU -> down
    WRef        mg_proj;          // [4096, 4096], no bias
    const void* mg_ln_w;          // [4096]
    const void* mg_ln_b;          // [4096]
    WRef        mg_gate, mg_up, mg_down;
    int dtype;
};

// Scratch for `n_patch` input rows. Sized for the worst case the caller will pass.
size_t vision_workspace_floats(int n_patch);

// Run the tower. `x` is [n_patch, 1176] fp32; `cos`/`sin` are [n_patch, 64] fp32 from the vision
// rope; `out` receives [n_patch/4, 4096] fp32 -- the embeddings that replace image tokens.
//
// `n_patch` must be a multiple of MERGE*MERGE = 4: the downsample folds 2x2 spatial neighbours,
// and a ragged tail would silently mix patches from different images.
void vision_forward(const float* x, const float* cos, const float* sin,
                    const VisionWeights& W, int n_patch, float* out, float* ws, cudaStream_t s);

// Position ids for a (t, h, w) grid, matching transformers' get_vision_position_ids: patches are
// visited in spatial_merge_size x spatial_merge_size blocks, so the ids are NOT raster order.
// Writes [n_patch, 2] (h, w) into `pos`, host-side.
void vision_position_ids(int t, int h, int w, int merge, int* pos);

// cos/sin for those ids. theta 10000, dim = head_dim/2 = 32 frequencies over the two axes.
void vision_rope_tables(const int* pos, int n_patch, float* cos, float* sin);

}  // namespace glm5
