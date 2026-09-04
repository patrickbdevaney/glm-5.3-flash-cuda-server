// kda.h — Kimi Delta Attention decode path (34 of 45 layers, 47.4% of B_tok).
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace glm5 {

// One decode step of one KDA layer, batch 1.
//
// Buffers are caller-owned; `ws` is scratch of at least kda_workspace_floats().
// `conv_state` [3*QKV][KDA_CONV_STATE] and `S` [H][Dk][Dv] are updated IN PLACE.
struct KdaWeights {
    // Stored exactly as the checkpoint stores them (row-major [out, in]).
    const void* q_proj;    // [QKV, HIDDEN]
    const void* k_proj;    // [QKV, HIDDEN]
    const void* v_proj;    // [QKV, HIDDEN]
    const void* o_proj;    // [HIDDEN, QKV]
    const void* conv1d;    // [3*QKV, 4]  -- q,k,v conv weights concatenated in that order
    const void* f_a;       // [GATE_RANK, HIDDEN]
    const void* f_b;       // [QKV, GATE_RANK]
    const float* dt_bias;  // [QKV]
    const float* A_log;    // [H]
    const void* b_proj;    // [H, HIDDEN]
    const void* g_a;       // [GATE_RANK, HIDDEN]
    const void* g_b;       // [QKV, GATE_RANK]
    const void* o_norm;    // [Dv]
    int dtype;             // 0 = fp32 (oracle/gate), 1 = bf16 (production)
};

size_t kda_workspace_floats();

// x [HIDDEN] fp32 -> y [HIDDEN] fp32
void kda_decode_step(const float* x, const KdaWeights& W,
                     float* conv_state, float* S, float* y, float* ws,
                     cudaStream_t stream);

// Individually gateable stages, exposed so tests/gate_kda.cu can bisect a failure to one stage
// instead of reporting "the layer is wrong".
void kda_qkv_conv(const float* x, const KdaWeights& W, float* conv_state, float* qkv, float* ws,
                  cudaStream_t stream);
void kda_gates(const float* x, const KdaWeights& W, float* g, float* beta, float* gate, float* ws,
               cudaStream_t stream);
void kda_norm_qk(const float* qkv, float* q_n, float* k_n, cudaStream_t stream);
void kda_recurrence(const float* q_n, const float* k_n, const float* v_in, const float* g,
                    const float* beta, float* S, float* core_out, cudaStream_t stream);
void kda_out_norm(const float* core, const float* gate, const void* o_norm_w, int dtype,
                  float* normed, cudaStream_t stream);

}  // namespace glm5
