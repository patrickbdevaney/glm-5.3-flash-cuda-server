// engine.h — the GLM-5.3-Flash forward pass: embed -> 45 layers -> HyperHead mean -> norm -> lm_head.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include "glm5_config.h"
#include "sample.h"
#include "kda.h"
#include "mla.h"
#include "indexer.h"
#include "layer.h"
#include "moe.h"

namespace st { class WeightStore; }

namespace glm5 {

struct EngineConfig {
    std::string model_dir;
    int  max_ctx    = 2048;   // dense MLA is EXACT to 2048; beyond that the DSA indexer is required
    int  n_layer    = N_LAYER;
    // Widest multi-token forward the engine will run: the prefill chunk, and the ceiling on
    // speculative verify width. Sizes the batch activation buffers, so it must be set before load.
    // 32 is where prefill stops improving: 57.5 / 52.8 / 51.7 / 51.5 / 52.1 ms/tok at chunk
    // 4 / 16 / 32 / 64 / 128 (OPTIMIZATION_LOG #12). Wider costs buffers for nothing.
    int  max_batch  = 32;
    // Recurrent-state slots. 1 = ordinary autoregression, updated in place, no extra memory.
    // Speculation needs draft_width+1: token m of a verify writes slot m+1, so slot j holds the
    // state after exactly j tokens and rejecting K-j drafts is a pointer move rather than an
    // impossible rewind (SPEC_DECODE.md). Each slot is 145.56 MiB at the full 45 layers.
    int  state_slots = 1;
    bool verbose    = true;
};

// What a caller asks for. Defaults are generation_config.json's, not the habitual 0.7/0.9 —
// this checkpoint ships temperature 1.0 / top_p 0.95 and those are the numbers it was tuned at.
struct GenParams {
    SampleParams sampling;                  // temperature 1.0, top_p 0.95
    int  max_tokens = 512;
    bool has_seed   = false;                // without one, the seed comes from the clock
    uint64_t seed   = 0;
    std::vector<int> eos_ids;               // three of them for this model; empty = never stop early
};

struct GenStats {
    int prompt_tokens = 0;
    int cached_tokens = 0;                  // prompt tokens served from the resident prefix
    int completion_tokens = 0;
    double prefill_ms = 0, decode_ms = 0, tok_per_s = 0;
    bool hit_eos = false;
};

class Engine {
public:
    explicit Engine(const EngineConfig& cfg);
    ~Engine();

    // Prefill `ids`, then decode until EOS or max_tokens. `on_token` receives each generated id and
    // returns false to stop. EOS is NOT delivered to the callback — a server would have to filter
    // it out of every stream otherwise, and forgetting to is how a stop token ends up in the text.
    GenStats generate(const std::vector<int>& ids, const GenParams& p,
                      const std::function<bool(int)>& on_token);

    // Prefill only, leaving the sequence state at the end of `ids`. Returns the logits for the
    // last token in `logits_out` (host, VOCAB floats) if non-null.
    int prefill(const std::vector<int>& ids, float* logits_out = nullptr);

    // Prefill chunk width. Defaults to max_batch; settable so a bench can sweep it without a
    // reload, which is what makes a round-robin sweep possible at all on a 100 GiB model.
    void setChunk(int c) { chunk_ = c < 1 ? 1 : (c > cfg_.max_batch ? cfg_.max_batch : c); }

    // M tokens in ONE forward, at positions pos0 .. pos0+M-1.
    //
    // This is the kernel the whole repo turns on. Weights are read ONCE for all M — a K-wide
    // forward costs 15.005 G plus whichever routed experts the K tokens select, not K x 19.76 G
    // (ROOFLINE.md §4). It is what makes prefill affordable AND what makes speculative
    // verification cheaper than just decoding the tokens, which is the only reason a draft head
    // can pay for itself.
    //
    // Results are BIT-IDENTICAL to M sequential decode() calls — gate_batch.cu asserts equality,
    // not closeness, because a verify that merely approximates the AR path is not lossless.
    //
    // `logits` is [M, VOCAB] when all_logits, else [VOCAB] for the last token only; nullptr skips
    // lm_head entirely (6.4% of B_tok saved on every prefill chunk but the last).
    // `snapshot` requires state_slots > M: the recurrent state is then left one-per-position
    // instead of one-at-the-end, at no extra bandwidth. Follow it with commit_state_slot(j).
    void forward_batch(const int* tokens, int M, int pos0, float* logits, bool all_logits,
                       cudaStream_t s = 0, bool snapshot = false);

    // MULTIMODAL: rows of `emb` replace the token embedding at absolute positions
    // [pos0, pos0 + n). Set before prefill; cleared by reset(). The language model is pure NoPE --
    // there is no rotary anywhere in its attention -- so an image contributes nothing to position
    // encoding beyond occupying n consecutive slots, and splicing is exactly an embedding swap.
    // That is why there is no mrope here: position reaches the LLM through the KDA layers.
    void set_image_embeds(int pos0, int n, const float* dev_emb);
    void clear_image_embeds();

    // Make slot j the canonical state. This is how a speculative verify accepts j of K drafts.
    void commit_state_slot(int j, cudaStream_t s = 0);
    IndexerState idxState(int slot);
    int  stateSlots() const { return cfg_.state_slots; }

    // One decode step at position `pos` (0-based). Writes logits [VOCAB] fp32 to `logits`.
    // Advances the KDA recurrent state and the MLA latent cache in place.
    // Pass logits = nullptr to stop after the last layer (skipping the final norm and lm_head),
    // which is what the stack gate compares against.
    void decode(int token_id, int pos, float* logits, cudaStream_t s = 0);

    // The four mHC residual streams after the last layer: [HC_MULT, HIDDEN]. Debug/gate use.
    const float* streamsDev() const { return streams_; }

    // Reset all sequence state (KDA recurrent + conv windows, MLA cache). Weights stay resident.
    void reset(cudaStream_t s = 0);

    double residentGiB() const;
    int    maxCtx() const { return cfg_.max_ctx; }
    int    nLayer()  const { return cfg_.n_layer; }
    // The engine's own [max_batch, VOCAB] logits buffer. Exposed so a profiling driver can
    // drive decode() without allocating a second one and changing the memory footprint it
    // is trying to measure.
    float* logitsDev() { return logits_dev_; }
    int    seqLen()  const { return (int)seq_.size(); }     // tokens currently in the resident state

private:
    struct LayerW {
        bool kda = false, moe = false;
        HcWeights hc_attn{}, hc_ffn{};
        const void* ln_in = nullptr;
        const void* ln_post = nullptr;
        KdaWeights  kw{};
        MlaWeights  mw{};
        IndexerWeights iw{};
        DenseMlp    dense{};
        MoeLayer    ml{};
        int kda_slot = -1, mla_slot = -1;    // index into the state / cache arrays
    };

    EngineConfig cfg_;
    st::WeightStore* ws_ = nullptr;
    int chunk_ = 0;              // 0 = use cfg_.max_batch
    std::vector<LayerW> L_;

    const void* embed_ = nullptr;
    const void* final_norm_ = nullptr;
    WRef lm_head_{};
    bool nvfp4_dense_ = false;   // ROOFLINE §3 overlay is loaded and in use

    // activations
    float* streams_ = nullptr;    // [HC_MULT, HIDDEN]
    float* resid_   = nullptr;    // [HC_MULT, HIDDEN]
    float* coll_    = nullptr;
    float* normed_  = nullptr;
    float* sub_     = nullptr;
    float* post_    = nullptr;
    float* comb_    = nullptr;
    float* hcws_    = nullptr;
    float* ws_kda_  = nullptr;
    float* ws_mla_  = nullptr;
    float* ws_moe_  = nullptr;
    float* b_ws_moe_ = nullptr;   // batched MoE: M x (logits + act + partials)
    float* b_selw_  = nullptr;    // [max_batch, topk]
    int32_t* b_sel_ = nullptr;    // [max_batch, topk]
    float* ws_mlp_  = nullptr;
    float* pooled_  = nullptr;
    int32_t* sel_   = nullptr;
    float* selw_    = nullptr;

    // sequence state
    // batch activations, sized for cfg_.max_batch
    float* b_streams_ = nullptr;  // [B, HC_MULT, HIDDEN]
    float* b_resid_   = nullptr;
    float* b_coll_    = nullptr;  // [B, HIDDEN]
    float* b_normed_  = nullptr;
    float* b_sub_     = nullptr;
    float* b_pooled_  = nullptr;
    float* b_post_    = nullptr;  // [B, HC_MULT]
    float* b_comb_    = nullptr;  // [B, HC_MULT*HC_MULT]
    float* b_hcws_    = nullptr;
    float* b_ws_kda_  = nullptr;
    float* b_ws_mla_  = nullptr;
    float* b_ws_mlp_  = nullptr;

    float* kda_state_ = nullptr;  // [n_kda][64*128*128]
    float* kda_conv_  = nullptr;  // [n_kda][3*8192*3]
    float* mla_cache_ = nullptr;  // [n_full][max_ctx*512]
    float* idx_state_ = nullptr;  // [n_full][indexer_state_floats(max_ctx)]
    float* ws_idx_    = nullptr;
    int32_t* idx_sel_ = nullptr;  // [IDX_OUT_WIDTH]
    int32_t* idx_n_   = nullptr;
    int n_kda_ = 0, n_full_ = 0;

    // Host-side logits staging. Pinned, because at 620 KB per token an unpinned copy is a staged
    // pageable transfer that serialises against the next step's kernels.
    float* logits_host_ = nullptr;
    float* logits_dev_  = nullptr;

    // Everything the resident state has already consumed, prompt and generated alike. A request
    // whose ids begin with exactly this can skip straight to the tail — see generate().
    std::vector<int> seq_;

    // Image embeddings pending for this sequence: absolute position -> device row. Kept as a few
    // ranges rather than a per-token map because an image is always contiguous.
    struct ImgSpan { int pos0, n; const float* dev; };
    std::vector<ImgSpan> img_spans_;

    std::vector<std::pair<float,int>> scratch_;             // sampler workspace, reused

    std::vector<void*> owned_;
    double resident_ = 0;
};

}  // namespace glm5
