// glm5_config.h — GLM-5.3-Flash-REAP50 geometry.
//
// Every constant here was read out of the checkpoint's config.json and cross-checked against the
// safetensors tensor shapes (tools/roofline.py prints both). Where the two could disagree the
// tensor shape wins, because the tensor is what we actually index.
#pragma once
#include <cstdint>
#include <cstddef>

// The CPU-only gates (tokenizer, api, stream) include this header from g++, not nvcc.
#ifdef __CUDACC__
#define GLM5_HD __host__ __device__
#else
#define GLM5_HD
#endif

namespace glm5 {

// ---- backbone ----
constexpr int HIDDEN            = 4096;
constexpr int N_LAYER           = 45;        // backbone layers 0..44
constexpr int MTP_LAYER         = 45;        // the multi-token-prediction block sits at index 45
constexpr int VOCAB             = 154880;
constexpr float RMS_EPS         = 1e-5f;

// EOS is plural. Dropping any of these makes the model talk to itself (see gguf-multi-eos-token-trap).
constexpr int EOS_IDS[]         = {154820, 154827, 154829};
constexpr int N_EOS             = 3;
constexpr int PAD_ID            = 154820;

// ---- layer typing ----
// 11 full-attention (MLA + DSA) layers, every 4th starting at 3; the other 34 are KDA.
constexpr int FULL_ATTN_LAYERS[] = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43};
constexpr int N_FULL_ATTN        = 11;
constexpr int N_KDA_LAYER        = 34;
constexpr int FIRST_K_DENSE      = 3;        // layers 0,1,2 use a dense MLP; 3..44 are MoE

inline constexpr GLM5_HD bool is_full_attn(int L) {
    return L >= FIRST_K_DENSE && ((L - 3) & 3) == 0 && L <= 43;
}
inline constexpr GLM5_HD bool is_kda(int L) { return L < N_LAYER && !is_full_attn(L); }
inline constexpr GLM5_HD bool is_moe(int L)  { return L >= FIRST_K_DENSE; }

// ---- KDA (Kimi Delta Attention) — 34 layers, 47.4% of B_tok ----
constexpr int KDA_HEADS         = 64;
constexpr int KDA_HEAD_DIM      = 128;       // k_dim == v_dim
constexpr int KDA_QKV_DIM       = KDA_HEADS * KDA_HEAD_DIM;   // 8192
constexpr int KDA_CONV_K        = 4;         // short depthwise conv
constexpr int KDA_CONV_STATE    = KDA_CONV_K - 1;             // 3 columns retained per channel
constexpr int KDA_GATE_RANK     = 128;       // f_a_proj / g_a_proj bottleneck
constexpr float KDA_LOWER_BOUND = -5.0f;     // linear_attn_config.gate_lower_bound
constexpr float KDA_L2_EPS      = 1e-6f;     // FLA convention: sqrt(sum + eps), NOT max(., eps)

// state per layer: [64 heads][128 k][128 v] fp32 = 4 MiB; x34 = 136 MiB, context-INDEPENDENT
constexpr size_t KDA_STATE_PER_LAYER = (size_t)KDA_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM;
constexpr size_t KDA_CONV_PER_LAYER  = (size_t)3 * KDA_QKV_DIM * KDA_CONV_STATE;

// ---- MLA (full-attention layers) — pure NoPE, qk_rope_head_dim == 0 ----
constexpr int MLA_HEADS         = 64;
constexpr int MLA_Q_LORA        = 1536;
constexpr int MLA_KV_LORA       = 512;
constexpr int MLA_QK_NOPE       = 256;
constexpr int MLA_QK_ROPE       = 0;         // NoPE: there is no rotary embedding in main attention
constexpr int MLA_V_HEAD        = 256;
constexpr int MLA_Q_DIM         = MLA_HEADS * MLA_QK_NOPE;    // 16384, = q_b_proj rows
constexpr int MLA_KV_B_OUT      = MLA_HEADS * (MLA_QK_NOPE + MLA_V_HEAD);  // 32768

// ---- DSA lightning indexer ----
constexpr int IDX_HEADS         = 32;
constexpr int IDX_HEAD_DIM      = 128;
constexpr int IDX_TOPK          = 2048;
constexpr int IDX_KPOOL         = 4;         // k-pooling with compress gate + APE (new vs 0731)

// ---- MoE ----
constexpr int N_ROUTED_EXPERT   = 144;       // REAP-50 pruned from 288
constexpr int N_EXPERT_PER_TOK  = 8;
constexpr int N_SHARED_EXPERT   = 1;
constexpr int MOE_INTER         = 2048;
constexpr int DENSE_INTER       = 12288;     // layers 0..2
constexpr float ROUTED_SCALE    = 2.5f;
constexpr bool NORM_TOPK_PROB   = true;
constexpr float SWIGLU_LIMIT    = 10.0f;
// scoring_func = sigmoid (NOT sqrtsoftplus as in DeepSeek-V4), topk_method = noaux_tc

// ---- manifold-constrained hyper-connections (mHC) ----
// Identical constants to DeepSeek-V4-Flash: kernels/hc.cu ports directly.
constexpr int HC_MULT           = 4;
constexpr int HC_SINKHORN_ITERS = 20;
constexpr float HC_EPS          = 1e-6f;
constexpr int HC_MIX            = (2 + HC_MULT) * HC_MULT;    // 24 = fn rows
constexpr int HC_HCD            = HC_MULT * HIDDEN;           // 16384 = fn cols
// NOTE: GLM's final HyperHead is an UNWEIGHTED MEAN over the 4 streams (DeepSeek's was weighted),
// and the MTP block at layer 45 carries NO hc_* tensors at all — plain pre-norm residual.

// ---- NVFP4 packing (MoE experts and shared experts only, in this checkpoint) ----
constexpr int NVFP4_GROUP       = 16;        // one fp8-e4m3 scale per 16 weights
// weight_packed  U8   [out, in/2]      two 4-bit values per byte, low nibble first
// weight_scale   F8   [out, in/16]
// weight_global_scale F32 [1]
// dequant: w[o,i] = fp4_lut[nibble] * (float(scale[o, i/16]) / global_scale[0])
// verified bit-identical to compressed-tensors NVFP4PackedCompressor.decompress
// (glm-5.3-reap/scripts/nvfp4_dequant_check.py, 0.000e+00 on every sampled tensor)

}  // namespace glm5
