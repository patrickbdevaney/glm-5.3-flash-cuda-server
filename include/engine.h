// engine.h — the GLM-5.3-Flash forward pass: embed -> 45 layers -> HyperHead mean -> norm -> lm_head.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>
#include "glm5_config.h"
#include "kda.h"
#include "mla.h"
#include "layer.h"
#include "moe.h"

namespace st { class WeightStore; }

namespace glm5 {

struct EngineConfig {
    std::string model_dir;
    int  max_ctx    = 2048;   // dense MLA is EXACT to 2048; beyond that the DSA indexer is required
    int  n_layer    = N_LAYER;
    bool verbose    = true;
};

class Engine {
public:
    explicit Engine(const EngineConfig& cfg);
    ~Engine();

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

private:
    struct LayerW {
        bool kda = false, moe = false;
        HcWeights hc_attn{}, hc_ffn{};
        const void* ln_in = nullptr;
        const void* ln_post = nullptr;
        KdaWeights  kw{};
        MlaWeights  mw{};
        DenseMlp    dense{};
        MoeLayer    ml{};
        int kda_slot = -1, mla_slot = -1;    // index into the state / cache arrays
    };

    EngineConfig cfg_;
    st::WeightStore* ws_ = nullptr;
    std::vector<LayerW> L_;

    const void* embed_ = nullptr;
    const void* final_norm_ = nullptr;
    const void* lm_head_ = nullptr;

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
    float* ws_mlp_  = nullptr;
    float* pooled_  = nullptr;
    int32_t* sel_   = nullptr;
    float* selw_    = nullptr;

    // sequence state
    float* kda_state_ = nullptr;  // [n_kda][64*128*128]
    float* kda_conv_  = nullptr;  // [n_kda][3*8192*3]
    float* mla_cache_ = nullptr;  // [n_full][max_ctx*512]
    int n_kda_ = 0, n_full_ = 0;

    std::vector<void*> owned_;
    double resident_ = 0;
};

}  // namespace glm5
