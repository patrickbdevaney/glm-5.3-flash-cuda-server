// engine.cu — the forward pass, wired from gated kernels.
//
//   embed -> [45 x decoder layer] -> HyperHead mean -> final RMSNorm -> lm_head
//
// The residual is FOUR streams of 4096 the whole way down (mHC), collapsed to one only at the very
// end and by an UNWEIGHTED mean — GLM's HyperHead, unlike DeepSeek-V4's weighted collapse.
//
// Layer typing, from the checkpoint: 34 KDA linear-attention layers and 11 MLA/DSA full-attention
// layers at 3, 7, 11, ..., 43; layers 0-2 have a dense MLP, 3-44 are MoE.
//
// Context limit: MLA runs DENSE causal here, which is exact up to 2048 tokens because the DSA
// indexer selects every pool below that (verified in ref/gen_mla.py, see OPTIMIZATION_LOG #3).
// Above 2048 the indexer is required and the engine refuses rather than quietly returning
// attention over the wrong set of keys.
#include "engine.h"
#include "vision.h"
#include "vision_preproc.h"
#include "dprof.h"
#include "gemv.h"
#include "weight_store.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <sys/stat.h>

namespace glm5 {

#define CU(x) do { cudaError_t e_=(x); if(e_){ fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(e_)); abort(); } } while(0)

static float* dalloc(std::vector<void*>& owned, size_t n_float, double* acc = nullptr) {
    void* p; CU(cudaMalloc(&p, n_float * 4));
    CU(cudaMemset(p, 0, n_float * 4));
    owned.push_back(p);
    if (acc) *acc += n_float * 4.0 / 1073741824.0;
    return (float*)p;
}

Engine::Engine(const EngineConfig& cfg) : cfg_(cfg) {
    // No context limit any more. At or below DENSE_CTX_LIMIT (2051) the attention runs dense,
    // which is EXACT because the indexer selects every visible key there; above it the indexer
    // picks the top 512 pools and attention runs sparse over them. The pool state is maintained
    // from token 0 either way, because it is built incrementally.

    // Load only what the requested layer count needs. On a box that cannot currently hold the
    // whole 98 GiB checkpoint this is the difference between a smoke test and an OOM.
    std::string pats;
    if (cfg_.n_layer < N_LAYER) {
        for (int i = 0; i < cfg_.n_layer; ++i) pats += "layers." + std::to_string(i) + ".,";
        pats += "embed_tokens,lm_head,language_model.norm";
    }
    // ROOFLINE §3: the NVFP4 dense-weight overlay. Present -> used, absent -> the engine runs
    // exactly as it did before, on bf16. GLM5_DENSE_NVFP4=0 forces bf16 with the overlay resident,
    // which is what makes an A/B a restart rather than a reload.
    std::vector<std::string> overlays;
    {
        const char* off = getenv("GLM5_DENSE_NVFP4");
        const char* od  = getenv("GLM5_NVFP4_OVERLAY");
        std::string dir = od ? od : (cfg_.model_dir + "/../glm-5.3-flash-dense-nvfp4-overlay");
        struct stat sb;
        const bool have = (stat((dir + "/model.safetensors.index.json").c_str(), &sb) == 0);
        nvfp4_dense_ = have && !(off && std::string(off) == "0");
        if (nvfp4_dense_) overlays.push_back(dir);
        else if (cfg_.verbose && have) printf("engine: dense NVFP4 overlay present but DISABLED\n");
    }
    ws_ = new st::WeightStore(cfg_.model_dir, nullptr, pats.empty() ? nullptr : pats.c_str(),
                              overlays);
    resident_ = ws_->loadedGiB();
    // Probe now, not on the first image: hasVision() is what the server checks to decide whether
    // to accept an image part at all, and a lazily-set flag reads false until it is too late.
    has_vision_ = ws_->has("model.visual.patch_embed.proj.weight") &&
                  ws_->has("model.visual.merger.down_proj.weight");
    if (cfg_.verbose)
        printf("engine: vision tower %s\n", has_vision_ ? "resident" : "ABSENT (reduced --n-layer)");
    dprof_set_nvfp4_dense(nvfp4_dense_);
    if (cfg_.verbose && nvfp4_dense_) printf("engine: dense NVFP4 overlay ACTIVE\n");
    if (cfg_.verbose)
        printf("engine: %zu tensors, %.2f GiB resident, %d layers, max_ctx %d\n",
               ws_->count(), resident_, cfg_.n_layer, cfg_.max_ctx);

    embed_      = ws_->get("model.language_model.embed_tokens.weight").dev;
    final_norm_ = ws_->get("model.language_model.norm.weight").dev;
    lm_head_ = (nvfp4_dense_ && ws_->has("lm_head.weight_packed"))
                   ? WRef(ws_->dev<uint8_t>("lm_head.weight_packed"),
                          ws_->dev<uint8_t>("lm_head.weight_scale"),
                          ws_->dev<float>("lm_head.weight_global_scale"))
                   : WRef(ws_->get("lm_head.weight").dev);

    L_.resize(cfg_.n_layer);
    std::vector<Nvfp4Mat> hostE(3 * N_ROUTED_EXPERT), hostS(3);

    for (int i = 0; i < cfg_.n_layer; ++i) {
        LayerW& l = L_[i];
        const std::string P = "model.language_model.layers." + std::to_string(i) + ".";
        auto D = [&](const std::string& s) { return ws_->get(P + s).dev; };
        auto F = [&](const std::string& s) { return ws_->dev<float>(P + s); };
        // Prefer the NVFP4 overlay wherever it exists, fall back to bf16 where it does not.
        // `stem` is the tensor name WITHOUT ".weight": the overlay stores the triple under it.
        auto Q = [&](const std::string& stem) -> WRef {
            const std::string b = P + stem;
            if (nvfp4_dense_ && ws_->has(b + ".weight_packed"))
                return WRef(ws_->dev<uint8_t>(b + ".weight_packed"),
                            ws_->dev<uint8_t>(b + ".weight_scale"),
                            ws_->dev<float>(b + ".weight_global_scale"));
            // Not every tensor is named "<stem>.weight" — index_kpool_compress_gate is a bare
            // parameter, and the overlay stores it under its own name.
            return WRef(ws_->get(ws_->has(b + ".weight") ? b + ".weight" : b).dev);
        };

        l.kda = is_kda(i);
        l.moe = is_moe(i);
        l.hc_attn = {D("hc_attn_fn"), F("hc_attn_base"), F("hc_attn_scale")};
        l.hc_ffn  = {D("hc_ffn_fn"),  F("hc_ffn_base"),  F("hc_ffn_scale")};
        l.ln_in   = D("input_layernorm.weight");
        l.ln_post = D("post_attention_layernorm.weight");

        if (l.kda) {
            l.kda_slot = n_kda_++;
            // The checkpoint splits the fused conv1d into q_/k_/v_conv1d; concatenate in q,k,v
            // order and widen to fp32 (393 KB per layer — too small for a bf16 path to matter).
            const size_t per = (size_t)KDA_QKV_DIM * KDA_CONV_K;
            float* cw = dalloc(owned_, 3 * per, &resident_);
            const char* nm[3] = {"self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight",
                                 "self_attn.v_conv1d.weight"};
            for (int j = 0; j < 3; ++j) f32_from_bf16_dev(cw + j * per, D(nm[j]), per, 0);
            l.kw.dtype   = GEMV_BF16;
            l.kw.q_proj  = Q("self_attn.q_proj");
            l.kw.k_proj  = Q("self_attn.k_proj");
            l.kw.v_proj  = Q("self_attn.v_proj");
            l.kw.o_proj  = Q("self_attn.o_proj");
            l.kw.conv1d  = cw;
            l.kw.f_a     = Q("self_attn.f_a_proj");
            l.kw.f_b     = Q("self_attn.f_b_proj");
            l.kw.dt_bias = F("self_attn.dt_bias");
            l.kw.A_log   = F("self_attn.A_log");
            l.kw.b_proj  = Q("self_attn.b_proj");
            l.kw.g_a     = Q("self_attn.g_a_proj");
            l.kw.g_b     = Q("self_attn.g_b_proj");
            l.kw.o_norm  = D("self_attn.o_norm.weight");
        } else {
            l.mla_slot = n_full_++;
            l.mw.dtype     = GEMV_BF16;
            l.mw.q_a       = Q("self_attn.q_a_proj");
            l.mw.q_a_norm  = D("self_attn.q_a_layernorm.weight");
            l.mw.q_b       = Q("self_attn.q_b_proj");
            l.mw.kv_a      = Q("self_attn.kv_a_proj_with_mqa");
            l.mw.kv_a_norm = D("self_attn.kv_a_layernorm.weight");
            l.mw.kv_b      = D("self_attn.kv_b_proj.weight");
            l.mw.o_proj    = Q("self_attn.o_proj");
            l.iw.dtype         = GEMV_BF16;
            l.iw.wq_b          = Q("self_attn.indexer.wq_b");
            l.iw.wk            = Q("self_attn.indexer.wk");
            l.iw.k_norm_w      = D("self_attn.indexer.k_norm.weight");
            l.iw.k_norm_b      = D("self_attn.indexer.k_norm.bias");
            l.iw.weights_proj  = Q("self_attn.indexer.weights_proj");
            l.iw.compress_ape  = D("self_attn.indexer.index_kpool_compress_ape");
            l.iw.compress_gate = Q("self_attn.indexer.index_kpool_compress_gate");
        }

        if (!l.moe) {
            l.dense = {Q("mlp.gate_proj"), Q("mlp.up_proj"), Q("mlp.down_proj"),
                       DENSE_INTER, GEMV_BF16};
        } else {
            auto mat = [&](const std::string& b) {
                Nvfp4Mat m{ws_->dev<uint8_t>(P + b + ".weight_packed"),
                           ws_->dev<uint8_t>(P + b + ".weight_scale"),
                           ws_->dev<float>(P + b + ".weight_global_scale")};
                if (!nvfp4_check_align(m, (P + b).c_str())) abort();
                return m;
            };
            for (int e = 0; e < N_ROUTED_EXPERT; ++e) {
                const std::string b = "mlp.experts." + std::to_string(e) + ".";
                hostE[e * 3 + 0] = mat(b + "gate_proj");
                hostE[e * 3 + 1] = mat(b + "up_proj");
                hostE[e * 3 + 2] = mat(b + "down_proj");
            }
            hostS[0] = mat("mlp.shared_experts.gate_proj");
            hostS[1] = mat("mlp.shared_experts.up_proj");
            hostS[2] = mat("mlp.shared_experts.down_proj");
            Nvfp4Mat *de, *ds;
            CU(cudaMalloc(&de, hostE.size() * sizeof(Nvfp4Mat))); owned_.push_back(de);
            CU(cudaMalloc(&ds, 3 * sizeof(Nvfp4Mat)));            owned_.push_back(ds);
            CU(cudaMemcpy(de, hostE.data(), hostE.size() * sizeof(Nvfp4Mat), cudaMemcpyHostToDevice));
            CU(cudaMemcpy(ds, hostS.data(), 3 * sizeof(Nvfp4Mat), cudaMemcpyHostToDevice));
            l.ml.router_w    = D("mlp.gate.weight");
            l.ml.router_bias = F("mlp.gate.e_score_correction_bias");
            l.ml.experts     = de;
            l.ml.shared      = ds;
            l.ml.n_expert    = N_ROUTED_EXPERT;
            l.ml.topk        = N_EXPERT_PER_TOK;
        }
    }

    // activations + sequence state
    streams_ = dalloc(owned_, (size_t)HC_MULT * HIDDEN);
    resid_   = dalloc(owned_, (size_t)HC_MULT * HIDDEN);
    coll_    = dalloc(owned_, HIDDEN);
    normed_  = dalloc(owned_, HIDDEN);
    sub_     = dalloc(owned_, HIDDEN);
    pooled_  = dalloc(owned_, HIDDEN);
    post_    = dalloc(owned_, HC_MULT);
    comb_    = dalloc(owned_, HC_MULT * HC_MULT);
    hcws_    = dalloc(owned_, hc_workspace_floats());
    ws_kda_  = dalloc(owned_, kda_workspace_floats());
    ws_mla_  = dalloc(owned_, mla_workspace_floats(cfg_.max_ctx));
    ws_moe_  = dalloc(owned_, moe_workspace_floats());
    b_ws_moe_ = dalloc(owned_, moe_batch_workspace_floats(cfg_.max_batch));
    b_selw_   = dalloc(owned_, (size_t)cfg_.max_batch * N_EXPERT_PER_TOK);
    { void* p; CU(cudaMalloc(&p, (size_t)cfg_.max_batch * N_EXPERT_PER_TOK * 4));
      owned_.push_back(p); b_sel_ = (int32_t*)p; }
    ws_mlp_  = dalloc(owned_, dense_mlp_workspace_floats());
    selw_    = dalloc(owned_, N_EXPERT_PER_TOK);
    // Batch activations. At max_batch 16 these total well under 200 MB, which is nothing against
    // the weights — and they are what let one forward serve 16 tokens.
    {
        const int B = cfg_.max_batch < 1 ? 1 : cfg_.max_batch;
        b_streams_ = dalloc(owned_, (size_t)B * HC_MULT * HIDDEN);
        b_resid_   = dalloc(owned_, (size_t)B * HC_MULT * HIDDEN);
        b_coll_    = dalloc(owned_, (size_t)B * HIDDEN);
        b_normed_  = dalloc(owned_, (size_t)B * HIDDEN);
        b_sub_     = dalloc(owned_, (size_t)B * HIDDEN);
        b_pooled_  = dalloc(owned_, (size_t)B * HIDDEN);
        b_post_    = dalloc(owned_, (size_t)B * HC_MULT);
        b_comb_    = dalloc(owned_, (size_t)B * HC_MULT * HC_MULT);
        b_hcws_    = dalloc(owned_, (size_t)B * hc_workspace_floats());
        b_ws_kda_  = dalloc(owned_, kda_batch_workspace_floats(B));
        b_ws_mla_  = dalloc(owned_, mla_batch_workspace_floats(cfg_.max_ctx, B));
        b_ws_mlp_  = dalloc(owned_, dense_mlp_batch_workspace_floats(B));
    }
    logits_dev_ = dalloc(owned_, (size_t)(cfg_.max_batch < 1 ? 1 : cfg_.max_batch) * VOCAB, &resident_);
    CU(cudaHostAlloc(&logits_host_, (size_t)VOCAB * 4, cudaHostAllocDefault));
    { void* p; CU(cudaMalloc(&p, N_EXPERT_PER_TOK * 4)); owned_.push_back(p); sel_ = (int32_t*)p; }

    if (cfg_.state_slots < 1) cfg_.state_slots = 1;
    kda_state_ = dalloc(owned_, (size_t)cfg_.state_slots * n_kda_ * KDA_STATE_PER_LAYER, &resident_);
    kda_conv_  = dalloc(owned_, (size_t)cfg_.state_slots * n_kda_ * KDA_CONV_PER_LAYER, &resident_);
    mla_cache_ = dalloc(owned_, (size_t)n_full_ * cfg_.max_ctx * MLA_KV_LORA, &resident_);
    idx_state_ = dalloc(owned_, (size_t)n_full_ * indexer_state_floats(cfg_.max_ctx), &resident_);
    ws_idx_    = dalloc(owned_, indexer_workspace_floats(cfg_.max_ctx));
    { void* p; CU(cudaMalloc(&p, IDX_OUT_WIDTH * 4)); owned_.push_back(p); idx_sel_ = (int32_t*)p;
      CU(cudaMalloc(&p, 4)); owned_.push_back(p); idx_n_ = (int32_t*)p; }
    if (cfg_.verbose)
        printf("engine: %d KDA layers (%.2f MiB state, context-independent), %d full-attn layers "
               "(%.2f MiB latent cache at %d ctx)\n",
               n_kda_, n_kda_ * (KDA_STATE_PER_LAYER + KDA_CONV_PER_LAYER) * 4.0 / 1048576.0,
               n_full_, n_full_ * (double)cfg_.max_ctx * MLA_KV_LORA * 4.0 / 1048576.0, cfg_.max_ctx);
}

Engine::~Engine() {
    if (logits_host_) cudaFreeHost(logits_host_);
    for (void* p : owned_) cudaFree(p);
    delete ws_;
}

double Engine::residentGiB() const { return resident_; }

void Engine::reset(cudaStream_t s) {
    CU(cudaMemsetAsync(kda_state_, 0, (size_t)cfg_.state_slots * n_kda_ * KDA_STATE_PER_LAYER * 4, s));
    CU(cudaMemsetAsync(kda_conv_, 0, (size_t)cfg_.state_slots * n_kda_ * KDA_CONV_PER_LAYER * 4, s));
    CU(cudaMemsetAsync(mla_cache_, 0, (size_t)n_full_ * cfg_.max_ctx * MLA_KV_LORA * 4, s));
    CU(cudaMemsetAsync(idx_state_, 0, (size_t)n_full_ * indexer_state_floats(cfg_.max_ctx) * 4, s));
    seq_.clear();
    // NOT img_spans_. generate() calls reset() for any request that cannot reuse the resident
    // prefix, which is every first turn -- clearing here wiped the image embeddings the caller had
    // just registered and the model politely reported that no image was attached. The spans are
    // request-scoped and the caller owns them (Engine::clear_image_embeds).
}

// streams[h][d] = embed[token][d] for every h — the model broadcasts one embedding across all
// four residual streams (inputs_embeds.unsqueeze(2).expand(-1,-1,hc_mult,-1)).
__global__ void k_embed_broadcast(float* __restrict__ streams, const __nv_bfloat16* __restrict__ emb,
                                  int token) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= HIDDEN) return;
    const float v = __bfloat162float(emb[(size_t)token * HIDDEN + d]);
    for (int h = 0; h < HC_MULT; ++h) streams[(size_t)h * HIDDEN + d] = v;
}

// Multimodal twin: the row comes from a precomputed fp32 embedding instead of the token table,
// and is broadcast into all HC_MULT hyper-connection streams exactly as k_embed_broadcast does.
__global__ void k_embed_rows(float* __restrict__ streams, const float* __restrict__ row) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= HIDDEN) return;
    const float v = row[d];
    for (int h = 0; h < HC_MULT; ++h) streams[(size_t)h * HIDDEN + d] = v;
}

// M tokens, one forward. See engine.h for why this exists.
// The per-layer view into the shared indexer state block.
IndexerState Engine::idxState(int slot) {
    IndexerState S{};
    float* base = idx_state_ + (size_t)slot * indexer_state_floats(cfg_.max_ctx);
    S.pool_keys = base;
    S.roll_k    = base + (size_t)idx_max_pools(cfg_.max_ctx) * IDX_HEAD_DIM;
    S.roll_gate = S.roll_k + IDX_KPOOL * IDX_HEAD_DIM;
    return S;
}

// Build VisionWeights out of the resident store. Lazy: an engine that never sees an image never
// pays for the lookup, and one built with a reduced --n-layer never has the tensors at all.
static bool build_vision(st::WeightStore* ws, VisionWeights& W) {
    auto has = [&](const std::string& n) { return ws->has("model.visual." + n); };
    if (!has("patch_embed.proj.weight") || !has("merger.down_proj.weight")) return false;
    auto T = [&](const std::string& n) { return ws->get("model.visual." + n).dev; };
    W.dtype = GEMV_BF16;
    W.patch_embed = WRef(T("patch_embed.proj.weight"));
    W.patch_embed_b = T("patch_embed.proj.bias");
    for (int b = 0; b < VIS_DEPTH; ++b) {
        const std::string p = "blocks." + std::to_string(b) + ".";
        auto& B = W.blocks[b];
        B.norm1 = T(p + "norm1.weight");
        B.qkv = WRef(T(p + "attn.qkv.weight"));       B.qkv_b = T(p + "attn.qkv.bias");
        B.q_norm = T(p + "attn.q_norm.weight");       B.k_norm = T(p + "attn.k_norm.weight");
        B.proj = WRef(T(p + "attn.proj.weight"));     B.proj_b = T(p + "attn.proj.bias");
        B.norm2 = T(p + "norm2.weight");
        B.gate = WRef(T(p + "mlp.gate_proj.weight")); B.gate_b = T(p + "mlp.gate_proj.bias");
        B.up   = WRef(T(p + "mlp.up_proj.weight"));   B.up_b   = T(p + "mlp.up_proj.bias");
        B.down = WRef(T(p + "mlp.down_proj.weight")); B.down_b = T(p + "mlp.down_proj.bias");
    }
    W.post_layernorm = T("post_layernorm.weight");
    W.downsample = WRef(T("downsample.weight"));  W.downsample_b = T("downsample.bias");
    W.mg_proj = WRef(T("merger.proj.weight"));
    W.mg_ln_w = T("merger.post_projection_norm.weight");
    W.mg_ln_b = T("merger.post_projection_norm.bias");
    W.mg_gate = WRef(T("merger.gate_proj.weight"));
    W.mg_up   = WRef(T("merger.up_proj.weight"));
    W.mg_down = WRef(T("merger.down_proj.weight"));
    return true;
}

int Engine::encodeImage(const uint8_t* rgb, int h, int w, const float** dev, int max_image_tokens) {
    if (!has_vision_) return 0;
    if (!vw_) {
        vw_ = new VisionWeights();
        if (!build_vision(ws_, *vw_)) { delete vw_; vw_ = nullptr; has_vision_ = false; return 0; }
    }
    PreprocResult P = vision_preprocess_rgb(rgb, h, w, max_image_tokens);
    const int S = P.grid_h * P.grid_w;
    if (S <= 0 || (S & 3)) return 0;
    const int NO = S / 4;

    auto grow = [&](float** p, size_t* have, size_t want) {
        if (*have >= want) return;
        if (*p) cudaFree(*p);
        CU(cudaMalloc(p, want * 4)); *have = want;
    };
    grow(&vis_in_,  &vis_in_n_,  (size_t)S * VIS_IN_DIM);
    grow(&vis_ws_,  &vis_ws_n_,  vision_workspace_floats(S));
    grow(&vis_out_, &vis_out_n_, (size_t)NO * VIS_OUT_HIDDEN);
    grow(&vis_cos_, &vis_cos_n_, (size_t)S * VIS_HEAD_DIM);
    grow(&vis_sin_, &vis_sin_n_, (size_t)S * VIS_HEAD_DIM);

    std::vector<int> pos(2 * (size_t)S);
    vision_position_ids(1, P.grid_h, P.grid_w, VIS_MERGE, pos.data());
    std::vector<float> hc((size_t)S * VIS_HEAD_DIM), hs((size_t)S * VIS_HEAD_DIM);
    vision_rope_tables(pos.data(), S, hc.data(), hs.data());

    CU(cudaMemcpy(vis_in_, P.patches.data(), P.patches.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(vis_cos_, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(vis_sin_, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
    vision_forward(vis_in_, vis_cos_, vis_sin_, *vw_, S, vis_out_, vis_ws_, 0);
    CU(cudaDeviceSynchronize());
    *dev = vis_out_;
    return NO;
}

void Engine::set_image_embeds(int pos0, int n, const float* dev_emb) {
    img_spans_.push_back({pos0, n, dev_emb});
}
void Engine::clear_image_embeds() { img_spans_.clear(); }

void Engine::commit_state_slot(int j, cudaStream_t s) {
    if (j < 0 || j >= cfg_.state_slots) {
        fprintf(stderr, "engine: slot %d outside 0..%d\n", j, cfg_.state_slots - 1); abort(); }
    if (j == 0) return;
    const size_t sstride = (size_t)n_kda_ * KDA_STATE_PER_LAYER;
    const size_t cstride = (size_t)n_kda_ * KDA_CONV_PER_LAYER;
    CU(cudaMemcpyAsync(kda_state_, kda_state_ + (size_t)j * sstride, sstride * 4,
                       cudaMemcpyDeviceToDevice, s));
    CU(cudaMemcpyAsync(kda_conv_, kda_conv_ + (size_t)j * cstride, cstride * 4,
                       cudaMemcpyDeviceToDevice, s));
}

void Engine::forward_batch(const int* tokens, int M, int pos0, float* logits, bool all_logits,
                           cudaStream_t s, bool snapshot) {
    if (snapshot && cfg_.state_slots <= M) {
        fprintf(stderr, "engine: snapshot of %d tokens needs state_slots > %d, have %d\n",
                M, M, cfg_.state_slots); abort(); }
    if (M < 1 || M > cfg_.max_batch) {
        fprintf(stderr, "engine: batch %d outside 1..%d\n", M, cfg_.max_batch); abort(); }
    if (pos0 + M > cfg_.max_ctx) {
        fprintf(stderr, "engine: pos %d+%d >= max_ctx %d\n", pos0, M, cfg_.max_ctx); abort(); }
    for (int m = 0; m < M; ++m)
        if (tokens[m] < 0 || tokens[m] >= VOCAB) {
            fprintf(stderr, "engine: token %d out of range\n", tokens[m]); abort(); }

    dprof_begin(DP_EMBED, s);
    for (int m = 0; m < M; ++m)
        k_embed_broadcast<<<(HIDDEN + 255) / 256, 256, 0, s>>>(
            b_streams_ + (size_t)m * HC_MULT * HIDDEN, (const __nv_bfloat16*)embed_, tokens[m]);
    // Multimodal: overwrite the broadcast embedding wherever an image covers this position. Done
    // AFTER the broadcast rather than instead of it so the hyper-connection streams are already
    // laid out; the image row replaces all HC_MULT copies, exactly as an embedding would.
    for (const auto& sp : img_spans_)
        for (int m = 0; m < M; ++m) {
            const int p = pos0 + m;
            if (p >= sp.pos0 && p < sp.pos0 + sp.n)
                k_embed_rows<<<(HIDDEN + 255) / 256, 256, 0, s>>>(
                    b_streams_ + (size_t)m * HC_MULT * HIDDEN,
                    sp.dev + (size_t)(p - sp.pos0) * HIDDEN);
        }
    dprof_end(DP_EMBED, s);

    for (int i = 0; i < cfg_.n_layer; ++i) {
        LayerW& l = L_[i];

        // ---- attention site ----
        CU(cudaMemcpyAsync(b_resid_, b_streams_, (size_t)M * HC_MULT * HIDDEN * 4,
                           cudaMemcpyDeviceToDevice, s));
        // hc runs per token: it is 0.4% of B_tok and its own Sinkhorn is per-token state, so
        // looping costs M x 786 KB per site (~2% of a 4-wide forward). Batching k_hc_mix would
        // remove that; it is not the largest term and has not been done yet.
        dprof_begin(DP_HC_PRE_ATTN, s);
        for (int m = 0; m < M; ++m)
            hc_compose(b_streams_ + (size_t)m * HC_MULT * HIDDEN, l.hc_attn,
                       b_coll_ + (size_t)m * HIDDEN, b_post_ + (size_t)m * HC_MULT,
                       b_comb_ + (size_t)m * HC_MULT * HC_MULT,
                       b_hcws_ + (size_t)m * hc_workspace_floats(), s);
        dprof_end(DP_HC_PRE_ATTN, s);
        dprof_begin(DP_NORM_ATTN, s);
        for (int m = 0; m < M; ++m)
            rmsnorm(b_normed_ + (size_t)m * HIDDEN, b_coll_ + (size_t)m * HIDDEN,
                    l.ln_in, GEMV_BF16, HIDDEN, s);
        dprof_end(DP_NORM_ATTN, s);
        dprof_begin(DP_ATTN, s);
        if (l.kda) {
            dprof_begin(DP_KDA, s);
            // Slot stride is the whole per-slot state, so layer l's slot m sits at
            // base + m*stride + l*per_layer. Stride 0 is the in-place autoregressive case.
            kda_batch_step_slots(b_normed_, l.kw,
                           kda_conv_ + (size_t)l.kda_slot * KDA_CONV_PER_LAYER,
                           snapshot ? (size_t)n_kda_ * KDA_CONV_PER_LAYER : 0,
                           kda_state_ + (size_t)l.kda_slot * KDA_STATE_PER_LAYER,
                           snapshot ? (size_t)n_kda_ * KDA_STATE_PER_LAYER : 0,
                           b_sub_, b_ws_kda_, M, s);
            dprof_end(DP_KDA, s);
        } else {
            dprof_begin(DP_MLA, s);
            IndexerState IS = idxState(l.mla_slot);
            mla_batch_step_dsa(b_normed_, l.mw, l.iw, IS,
                               mla_cache_ + (size_t)l.mla_slot * cfg_.max_ctx * MLA_KV_LORA,
                               pos0, M, cfg_.max_ctx, idx_sel_, idx_n_, b_sub_, b_ws_mla_,
                               ws_idx_, s);
            dprof_end(DP_MLA, s);
        }
        dprof_end(DP_ATTN, s);
        dprof_begin(DP_HC_POST_ATTN, s);
        for (int m = 0; m < M; ++m)
            hc_apply(b_streams_ + (size_t)m * HC_MULT * HIDDEN, b_resid_ + (size_t)m * HC_MULT * HIDDEN,
                     b_sub_ + (size_t)m * HIDDEN, b_post_ + (size_t)m * HC_MULT,
                     b_comb_ + (size_t)m * HC_MULT * HC_MULT, s);
        dprof_end(DP_HC_POST_ATTN, s);

        // ---- MLP site ----
        CU(cudaMemcpyAsync(b_resid_, b_streams_, (size_t)M * HC_MULT * HIDDEN * 4,
                           cudaMemcpyDeviceToDevice, s));
        dprof_begin(DP_HC_PRE_FFN, s);
        for (int m = 0; m < M; ++m)
            hc_compose(b_streams_ + (size_t)m * HC_MULT * HIDDEN, l.hc_ffn,
                       b_coll_ + (size_t)m * HIDDEN, b_post_ + (size_t)m * HC_MULT,
                       b_comb_ + (size_t)m * HC_MULT * HC_MULT,
                       b_hcws_ + (size_t)m * hc_workspace_floats(), s);
        dprof_end(DP_HC_PRE_FFN, s);
        dprof_begin(DP_NORM_FFN, s);
        for (int m = 0; m < M; ++m)
            rmsnorm(b_normed_ + (size_t)m * HIDDEN, b_coll_ + (size_t)m * HIDDEN,
                    l.ln_post, GEMV_BF16, HIDDEN, s);
        dprof_end(DP_NORM_FFN, s);
        dprof_begin(DP_FFN, s);
        if (l.moe) {
            dprof_begin(DP_MOE, s);
            // The M tokens go through in ONE set of launches (token = grid.z). Serialising them
            // was most of why a wide prefill chunk bought nothing: prefill at width 16 measured
            // 84.0 ms/tok against 81.7 at width 1, and ffn:moe cost the same 35.1 ms per token at
            // both. The expert weights do not care who reads them, and at M=16 about 42 of the
            // 128 selections are repeats whose second read now comes out of L2.
            moe_forward_batch(b_normed_, l.ml, b_sub_, b_sel_, b_selw_, b_ws_moe_, M, s);
            dprof_end(DP_MOE, s);
        } else {
            dprof_begin(DP_DENSE, s);
            dense_mlp_batch(b_normed_, l.dense, b_sub_, b_ws_mlp_, M, s);
            dprof_end(DP_DENSE, s);
        }
        dprof_end(DP_FFN, s);
        dprof_begin(DP_HC_POST_FFN, s);
        for (int m = 0; m < M; ++m)
            hc_apply(b_streams_ + (size_t)m * HC_MULT * HIDDEN, b_resid_ + (size_t)m * HC_MULT * HIDDEN,
                     b_sub_ + (size_t)m * HIDDEN, b_post_ + (size_t)m * HC_MULT,
                     b_comb_ + (size_t)m * HC_MULT * HC_MULT, s);
        dprof_end(DP_HC_POST_FFN, s);
    }

    // Pool ALWAYS, even when the caller wants no logits: b_pooled_ is what pooledDev() hands to
    // /v1/embeddings, and folding it into the lm_head skip meant every embedding came back as
    // 4096 zeros -- which normalises to zero and yields a cosine of 0.0000 between every pair,
    // a result that looks like a broken model rather than an unwritten buffer.
    // Only the last token is pooled when logits are skipped, so this costs one head_mean.
    const int first = (logits && all_logits) ? 0 : M - 1;
    pooled_row_ = M - 1;                       // what pooledDev() must point at
    dprof_begin(DP_HEAD_MEAN, s);
    for (int m = first; m < M; ++m) {
        hc_head_mean(b_streams_ + (size_t)m * HC_MULT * HIDDEN, b_pooled_ + (size_t)m * HIDDEN, s);
        rmsnorm(b_pooled_ + (size_t)m * HIDDEN, b_pooled_ + (size_t)m * HIDDEN,
                final_norm_, GEMV_BF16, HIDDEN, s);
    }
    dprof_end(DP_HEAD_MEAN, s);
    if (!logits) return;                       // pooled is valid; lm_head is the part being skipped
    dprof_begin(DP_LM_HEAD, s);
    gemm(logits, lm_head_, b_pooled_ + (size_t)first * HIDDEN, M - first, VOCAB, HIDDEN, GEMV_BF16, s);
    dprof_end(DP_LM_HEAD, s);
}

void Engine::decode(int token_id, int pos, float* logits, cudaStream_t s) {
    if (pos >= cfg_.max_ctx) { fprintf(stderr, "engine: pos %d >= max_ctx %d\n", pos, cfg_.max_ctx); abort(); }
    if (token_id < 0 || token_id >= VOCAB) { fprintf(stderr, "engine: token %d out of range\n", token_id); abort(); }

    dprof_begin(DP_EMBED, s);
    k_embed_broadcast<<<(HIDDEN + 255) / 256, 256, 0, s>>>(streams_, (const __nv_bfloat16*)embed_, token_id);
    for (const auto& sp : img_spans_)
        if (pos >= sp.pos0 && pos < sp.pos0 + sp.n)
            k_embed_rows<<<(HIDDEN + 255) / 256, 256, 0, s>>>(
                streams_, sp.dev + (size_t)(pos - sp.pos0) * HIDDEN);
    dprof_end(DP_EMBED, s);

    // PING-PONG, not a copy. hc_apply reads the pre-site streams as `residual` and writes the
    // post-site streams, so the two never alias -- the memcpy that used to snapshot them into
    // resid_ was moving 64 KB twice per layer, 5.9 MB and 90 launches per token, to produce a
    // buffer the very next kernel could have read in place. 90 swaps is even, so `st` is back at
    // streams_ by the end and streamsDev() still names the live buffer.
    float* st = streams_, *alt = resid_;

    for (int i = 0; i < cfg_.n_layer; ++i) {
        LayerW& l = L_[i];

        // ---- attention site ----
        dprof_begin(DP_HC_PRE_ATTN, s);
        hc_compose(st, l.hc_attn, coll_, post_, comb_, hcws_, s);
        dprof_end(DP_HC_PRE_ATTN, s);
        dprof_begin(DP_NORM_ATTN, s);
        rmsnorm(normed_, coll_, l.ln_in, GEMV_BF16, HIDDEN, s);
        dprof_end(DP_NORM_ATTN, s);
        dprof_begin(DP_ATTN, s);
        if (l.kda) {
            dprof_begin(DP_KDA, s);
            kda_decode_step(normed_, l.kw,
                            kda_conv_ + (size_t)l.kda_slot * KDA_CONV_PER_LAYER,
                            kda_state_ + (size_t)l.kda_slot * KDA_STATE_PER_LAYER,
                            sub_, ws_kda_, s);
            dprof_end(DP_KDA, s);
        } else {
            dprof_begin(DP_MLA, s);
            IndexerState IS = idxState(l.mla_slot);
            mla_decode_step_dsa(normed_, l.mw, l.iw, IS,
                                mla_cache_ + (size_t)l.mla_slot * cfg_.max_ctx * MLA_KV_LORA,
                                pos, cfg_.max_ctx, idx_sel_, idx_n_, sub_, ws_mla_, ws_idx_, s);
            dprof_end(DP_MLA, s);
        }
        dprof_end(DP_ATTN, s);
        dprof_begin(DP_HC_POST_ATTN, s);
        hc_apply(alt, st, sub_, post_, comb_, s);
        { float* t = st; st = alt; alt = t; }
        dprof_end(DP_HC_POST_ATTN, s);

        // ---- MLP site ----
        dprof_begin(DP_HC_PRE_FFN, s);
        hc_compose(st, l.hc_ffn, coll_, post_, comb_, hcws_, s);
        dprof_end(DP_HC_PRE_FFN, s);
        dprof_begin(DP_NORM_FFN, s);
        rmsnorm(normed_, coll_, l.ln_post, GEMV_BF16, HIDDEN, s);
        dprof_end(DP_NORM_FFN, s);
        dprof_begin(DP_FFN, s);
        if (l.moe) { dprof_begin(DP_MOE,   s); moe_forward(normed_, l.ml, sub_, sel_, selw_, ws_moe_, s); dprof_end(DP_MOE,   s); }
        else       { dprof_begin(DP_DENSE, s); dense_mlp(normed_, l.dense, sub_, ws_mlp_, s);             dprof_end(DP_DENSE, s); }
        dprof_end(DP_FFN, s);
        dprof_begin(DP_HC_POST_FFN, s);
        hc_apply(alt, st, sub_, post_, comb_, s);
        { float* t = st; st = alt; alt = t; }
        dprof_end(DP_HC_POST_FFN, s);
    }

    if (!logits) return;                                    // stack gate stops here
    dprof_begin(DP_HEAD_MEAN, s);
    hc_head_mean(st, pooled_, s);                     // unweighted mean over the 4 streams
    rmsnorm(pooled_, pooled_, final_norm_, GEMV_BF16, HIDDEN, s);
    dprof_end(DP_HEAD_MEAN, s);
    dprof_begin(DP_LM_HEAD, s);
    gemv(logits, lm_head_, pooled_, VOCAB, HIDDEN, GEMV_BF16, s);
    dprof_end(DP_LM_HEAD, s);
}

// ---- prefill -----------------------------------------------------------------------------------
//
// Chunked through forward_batch, which is the whole reason that kernel exists. A sequential
// prefill pays the full 19.76 GB weight read for EVERY prompt token; a chunk of C amortises the
// 15.005 GB non-expert half across all C of them, so cost per token falls from 19.76 GB to
// 15.005/C + 4.756 GB — 8.5 GB at C=4, 5.2 GB at C=32 (ROOFLINE.md §4). That is 2.3x to 3.8x on
// the single largest cost a request pays before its first token.
//
// lm_head runs only for the LAST prompt token, and only on the last chunk: it is 6.4% of B_tok,
// saved on everything else for one `nullptr`.
int Engine::prefill(const std::vector<int>& ids, float* logits_out) {
    if (ids.empty()) throw std::runtime_error("prefill: empty prompt");
    const int start = (int)seq_.size();
    if (start + (int)ids.size() > cfg_.max_ctx)
        throw std::runtime_error("prefill: context " + std::to_string(start + ids.size()) +
                                 " exceeds max_ctx " + std::to_string(cfg_.max_ctx));
    const int N = (int)ids.size();
    const int C = chunk_ > 0 ? chunk_ : (cfg_.max_batch < 1 ? 1 : cfg_.max_batch);
    for (int off = 0; off < N; off += C) {
        const int m = std::min(C, N - off);
        const bool last = (off + m == N);
        forward_batch(ids.data() + off, m, start + off, last ? logits_dev_ : nullptr,
                      /*all_logits=*/false, 0);
        for (int i = 0; i < m; ++i) seq_.push_back(ids[off + i]);
    }
    CU(cudaMemcpy(logits_host_, logits_dev_, (size_t)VOCAB * 4, cudaMemcpyDeviceToHost));
    if (logits_out) memcpy(logits_out, logits_host_, (size_t)VOCAB * 4);
    return argmax(logits_host_, VOCAB);
}

// ---- generate ----------------------------------------------------------------------------------
GenStats Engine::generate(const std::vector<int>& ids, const GenParams& p,
                          const std::function<bool(int)>& on_token) {
    using clock = std::chrono::steady_clock;
    GenStats st;
    st.prompt_tokens = (int)ids.size();
    if (ids.empty()) throw std::runtime_error("generate: empty prompt");

    // PREFIX REUSE. The state is a RECURRENCE, so it cannot be rewound: a prefix is reusable only
    // when the new request begins with EVERYTHING the state has already consumed, prompt and
    // previous generation alike. That is the ordinary multi-turn shape, and it turns the second
    // turn of a conversation from a full re-prefill into just the new tokens. Any divergence —
    // including a client that drops reasoning_content from the history, which changes the rendered
    // prompt — falls back to a full reset. Strictly a prefix, never a partial match.
    size_t reuse = 0;
    if (!seq_.empty() && ids.size() > seq_.size() &&
        std::equal(seq_.begin(), seq_.end(), ids.begin()))
        reuse = seq_.size();
    else
        reset(0);
    st.cached_tokens = (int)reuse;

    const auto t0 = clock::now();
    std::vector<int> tail(ids.begin() + reuse, ids.end());
    prefill(tail);
    CU(cudaDeviceSynchronize());
    const auto t1 = clock::now();
    st.prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    uint64_t rng = p.has_seed ? p.seed
                              : (uint64_t)clock::now().time_since_epoch().count();
    auto is_eos = [&](int t) {
        for (int e : p.eos_ids) if (t == e) return true;
        return false;
    };

    // log-softmax of one token, plus the top alternatives. Two passes for numerical safety: a
    // straight exp() over 154880 logits overflows long before the max-subtracted form does.
    auto record_lp = [&](int tok) {
        if (!p.out_logprobs) return;
        const float* L = logits_host_;
        float mx = L[0];
        for (int i = 1; i < VOCAB; ++i) if (L[i] > mx) mx = L[i];
        double sum = 0;
        for (int i = 0; i < VOCAB; ++i) sum += exp((double)L[i] - mx);
        const double lse = mx + log(sum);
        TokenLogprob tl;
        tl.id = tok;
        tl.logprob = (float)((double)L[tok] - lse);
        if (p.n_logprobs > 0) {
            const int k = p.n_logprobs < VOCAB ? p.n_logprobs : VOCAB;
            std::vector<int> idx(VOCAB);
            for (int i = 0; i < VOCAB; ++i) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                              [&](int a, int b) { return L[a] > L[b]; });
            for (int i = 0; i < k; ++i) tl.top.emplace_back(idx[i], (float)((double)L[idx[i]] - lse));
        }
        p.out_logprobs->push_back(std::move(tl));
    };
    if (p.out_logprobs) p.out_logprobs->clear();

    int next = sample(logits_host_, VOCAB, p.sampling, rng, scratch_);
    for (int n = 0; n < p.max_tokens; ++n) {
        if (is_eos(next)) { st.hit_eos = true; break; }     // EOS never reaches the callback
        ++st.completion_tokens;
        record_lp(next);
        if (on_token && !on_token(next)) break;
        if ((int)seq_.size() >= cfg_.max_ctx) break;
        if (n + 1 >= p.max_tokens) break;                   // no need to run a forward we discard
        decode(next, (int)seq_.size(), logits_dev_, 0);
        seq_.push_back(next);
        CU(cudaMemcpy(logits_host_, logits_dev_, (size_t)VOCAB * 4, cudaMemcpyDeviceToHost));
        next = sample(logits_host_, VOCAB, p.sampling, rng, scratch_);
    }
    CU(cudaDeviceSynchronize());
    st.decode_ms = std::chrono::duration<double, std::milli>(clock::now() - t1).count();
    st.tok_per_s = st.decode_ms > 0 ? st.completion_tokens * 1000.0 / st.decode_ms : 0.0;
    return st;
}

}  // namespace glm5
