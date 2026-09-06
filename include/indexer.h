// indexer.h — the DeepSeek Sparse Attention (DSA) indexer: which keys a query is allowed to see.
//
// Only needed above DENSE_CTX_LIMIT (2051). At or below it every pool is selected and the engine's
// dense MLA is EXACT — verified against the real module in ref/gen_indexer.py, not assumed.
//
// WHAT IT COMPUTES, per query, from the module (transformers has it; the config does not):
//   k       = LayerNorm_{w,b}(wk @ h)                          [128]   NOTE: LayerNorm, WITH BIAS
//   gate    = compress_gate @ h                                [128]
//   pool p  = tokens [4p, 4p+3]; complete pools only
//   pk[p]   = sum_j softmax_j(gate_j + ape_j) * k_j            [128]   per-CHANNEL softmax over j
//   q       = wq_b @ q_resid                                   [32, 128]
//   wts     = (weights_proj @ h) * 32^-0.5                     [32]
//   score[p]= sum_h wts[h] * relu( (q[h] . pk[p]) * 128^-0.5 )
//   select   top-k of score, k = min(index_topk/index_kpool, n_complete_pools) = min(512, floor(T/4))
//   output   the 4 raw token indices of each selected pool, then the incomplete TAIL raw,
//            padded with -1 to a CONSTANT width of index_topk + index_kpool - 1 = 2051
//
// FOUR THINGS THAT ARE NOT GUESSABLE AND EACH PRODUCE PLAUSIBLE-BUT-WRONG OUTPUT IF ASSUMED:
//   1. `k_norm` is a LayerNorm with a bias, eps 1e-6 — not the RMSNorm used everywhere else here.
//   2. The pool softmax is per CHANNEL over the 4 tokens, not per token over the channels.
//   3. A trailing INCOMPLETE pool is never selectable, but its tokens are appended raw. That is
//      why the dense limit is 2051 and not 2048.
//   4. `wts` can be NEGATIVE, so score is a signed sum of relu terms and routinely produces -0.0.
//      The top-k must canonicalise that (topk_radix.h does) or it silently picks different pools.
//
// Pool keys are FIXED once a pool's 4 tokens exist, so they are computed once on completion and
// cached — not recomputed 512-deep every decode step.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "glm5_config.h"
#include "gemv.h"

namespace glm5 {

struct IndexerWeights {
    WRef         wq_b;           // [IDX_HEADS*IDX_HEAD_DIM, MLA_Q_LORA]
    WRef         wk;             // [IDX_HEAD_DIM, HIDDEN]
    const void*  k_norm_w;       // [IDX_HEAD_DIM]
    const void*  k_norm_b;       // [IDX_HEAD_DIM]
    WRef         weights_proj;   // [IDX_HEADS, HIDDEN]
    const void*  compress_ape;   // [IDX_KPOOL, IDX_HEAD_DIM]
    WRef         compress_gate;  // [IDX_HEAD_DIM, HIDDEN]
    int dtype;                   // GEMV_F32 (gate) or GEMV_BF16 (production)
};

// Per-layer persistent state. `pool_keys` grows by one row every IDX_KPOOL tokens; `roll` holds the
// current incomplete pool's raw k and gate.
struct IndexerState {
    float* pool_keys;            // [max_pools, IDX_HEAD_DIM]
    float* roll_k;               // [IDX_KPOOL, IDX_HEAD_DIM]
    float* roll_gate;            // [IDX_KPOOL, IDX_HEAD_DIM]
};

inline int idx_max_pools(int max_ctx) { return (max_ctx + IDX_KPOOL - 1) / IDX_KPOOL; }
size_t indexer_state_floats(int max_ctx);
size_t indexer_workspace_floats(int max_ctx);

// One decode step at position `t` (0-based). Consumes h [HIDDEN] and q_resid [MLA_Q_LORA],
// advances the pool state, and writes the visible-key list.
//
// `out_idx` is [IDX_OUT_WIDTH] int32, padded with -1. `out_n` (device, 1 int) receives the count of
// real entries. Indices are NOT sorted — they are pool-score order then the tail, exactly as the
// reference emits them; attention treats them as a set.
void indexer_decode_step(const float* h, const float* q_resid, const IndexerWeights& W,
                         IndexerState& S, int t, int max_ctx,
                         int32_t* out_idx, int32_t* out_n, float* ws, cudaStream_t s);

// Individually gateable stages, so a failure localises to one of them.
void indexer_keys(const float* h, const IndexerWeights& W, IndexerState& S, int t,
                  float* ws, cudaStream_t s);                       // k, gate -> roll, maybe pool
void indexer_select(const float* h, const float* q_resid, const IndexerWeights& W,
                    IndexerState& S, int t, int32_t* out_idx, int32_t* out_n,
                    float* ws, cudaStream_t s);
void indexer_scores(const float* h, const float* q_resid, const IndexerWeights& W,
                    const IndexerState& S, int n_pools, float* scores, float* ws, cudaStream_t s);

}  // namespace glm5
