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
#include "gemv.h"
#include "weight_store.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

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
    // The dense-MLA limit is NOT index_topk. A trailing INCOMPLETE k-pool is never selectable
    // (pool_valid needs all 4 tokens) but its tokens are appended raw by append_visible_tail, so a
    // context of 2051 still has every token visible to every query. 2052 is the first length at
    // which the indexer actually drops something. Measured against the real module in
    // ref/gen_indexer.py, not derived from reading it.
    if (cfg_.max_ctx > DENSE_CTX_LIMIT)
        throw std::runtime_error("max_ctx > " + std::to_string(DENSE_CTX_LIMIT) +
                                 " requires the DSA indexer, which is not implemented yet");

    // Load only what the requested layer count needs. On a box that cannot currently hold the
    // whole 98 GiB checkpoint this is the difference between a smoke test and an OOM.
    std::string pats;
    if (cfg_.n_layer < N_LAYER) {
        for (int i = 0; i < cfg_.n_layer; ++i) pats += "layers." + std::to_string(i) + ".,";
        pats += "embed_tokens,lm_head,language_model.norm";
    }
    ws_ = new st::WeightStore(cfg_.model_dir, nullptr, pats.empty() ? nullptr : pats.c_str());
    resident_ = ws_->loadedGiB();
    if (cfg_.verbose)
        printf("engine: %zu tensors, %.2f GiB resident, %d layers, max_ctx %d\n",
               ws_->count(), resident_, cfg_.n_layer, cfg_.max_ctx);

    embed_      = ws_->get("model.language_model.embed_tokens.weight").dev;
    final_norm_ = ws_->get("model.language_model.norm.weight").dev;
    lm_head_    = ws_->get("lm_head.weight").dev;

    L_.resize(cfg_.n_layer);
    std::vector<Nvfp4Mat> hostE(3 * N_ROUTED_EXPERT), hostS(3);

    for (int i = 0; i < cfg_.n_layer; ++i) {
        LayerW& l = L_[i];
        const std::string P = "model.language_model.layers." + std::to_string(i) + ".";
        auto D = [&](const std::string& s) { return ws_->get(P + s).dev; };
        auto F = [&](const std::string& s) { return ws_->dev<float>(P + s); };

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
            l.kw.q_proj  = D("self_attn.q_proj.weight");
            l.kw.k_proj  = D("self_attn.k_proj.weight");
            l.kw.v_proj  = D("self_attn.v_proj.weight");
            l.kw.o_proj  = D("self_attn.o_proj.weight");
            l.kw.conv1d  = cw;
            l.kw.f_a     = D("self_attn.f_a_proj.weight");
            l.kw.f_b     = D("self_attn.f_b_proj.weight");
            l.kw.dt_bias = F("self_attn.dt_bias");
            l.kw.A_log   = F("self_attn.A_log");
            l.kw.b_proj  = D("self_attn.b_proj.weight");
            l.kw.g_a     = D("self_attn.g_a_proj.weight");
            l.kw.g_b     = D("self_attn.g_b_proj.weight");
            l.kw.o_norm  = D("self_attn.o_norm.weight");
        } else {
            l.mla_slot = n_full_++;
            l.mw.dtype     = GEMV_BF16;
            l.mw.q_a       = D("self_attn.q_a_proj.weight");
            l.mw.q_a_norm  = D("self_attn.q_a_layernorm.weight");
            l.mw.q_b       = D("self_attn.q_b_proj.weight");
            l.mw.kv_a      = D("self_attn.kv_a_proj_with_mqa.weight");
            l.mw.kv_a_norm = D("self_attn.kv_a_layernorm.weight");
            l.mw.kv_b      = D("self_attn.kv_b_proj.weight");
            l.mw.o_proj    = D("self_attn.o_proj.weight");
        }

        if (!l.moe) {
            l.dense = {D("mlp.gate_proj.weight"), D("mlp.up_proj.weight"), D("mlp.down_proj.weight"),
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
    seq_.clear();
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

// M tokens, one forward. See engine.h for why this exists.
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

    for (int m = 0; m < M; ++m)
        k_embed_broadcast<<<(HIDDEN + 255) / 256, 256, 0, s>>>(
            b_streams_ + (size_t)m * HC_MULT * HIDDEN, (const __nv_bfloat16*)embed_, tokens[m]);

    for (int i = 0; i < cfg_.n_layer; ++i) {
        LayerW& l = L_[i];

        // ---- attention site ----
        CU(cudaMemcpyAsync(b_resid_, b_streams_, (size_t)M * HC_MULT * HIDDEN * 4,
                           cudaMemcpyDeviceToDevice, s));
        // hc runs per token: it is 0.4% of B_tok and its own Sinkhorn is per-token state, so
        // looping costs M x 786 KB per site (~2% of a 4-wide forward). Batching k_hc_mix would
        // remove that; it is not the largest term and has not been done yet.
        for (int m = 0; m < M; ++m) {
            hc_compose(b_streams_ + (size_t)m * HC_MULT * HIDDEN, l.hc_attn,
                       b_coll_ + (size_t)m * HIDDEN, b_post_ + (size_t)m * HC_MULT,
                       b_comb_ + (size_t)m * HC_MULT * HC_MULT,
                       b_hcws_ + (size_t)m * hc_workspace_floats(), s);
            rmsnorm(b_normed_ + (size_t)m * HIDDEN, b_coll_ + (size_t)m * HIDDEN,
                    l.ln_in, GEMV_BF16, HIDDEN, s);
        }
        if (l.kda)
            // Slot stride is the whole per-slot state, so layer l's slot m sits at
            // base + m*stride + l*per_layer. Stride 0 is the in-place autoregressive case.
            kda_batch_step_slots(b_normed_, l.kw,
                           kda_conv_ + (size_t)l.kda_slot * KDA_CONV_PER_LAYER,
                           snapshot ? (size_t)n_kda_ * KDA_CONV_PER_LAYER : 0,
                           kda_state_ + (size_t)l.kda_slot * KDA_STATE_PER_LAYER,
                           snapshot ? (size_t)n_kda_ * KDA_STATE_PER_LAYER : 0,
                           b_sub_, b_ws_kda_, M, s);
        else
            mla_batch_step(b_normed_, l.mw,
                           mla_cache_ + (size_t)l.mla_slot * cfg_.max_ctx * MLA_KV_LORA,
                           pos0, M, cfg_.max_ctx, b_sub_, b_ws_mla_, s);
        for (int m = 0; m < M; ++m)
            hc_apply(b_streams_ + (size_t)m * HC_MULT * HIDDEN, b_resid_ + (size_t)m * HC_MULT * HIDDEN,
                     b_sub_ + (size_t)m * HIDDEN, b_post_ + (size_t)m * HC_MULT,
                     b_comb_ + (size_t)m * HC_MULT * HC_MULT, s);

        // ---- MLP site ----
        CU(cudaMemcpyAsync(b_resid_, b_streams_, (size_t)M * HC_MULT * HIDDEN * 4,
                           cudaMemcpyDeviceToDevice, s));
        for (int m = 0; m < M; ++m) {
            hc_compose(b_streams_ + (size_t)m * HC_MULT * HIDDEN, l.hc_ffn,
                       b_coll_ + (size_t)m * HIDDEN, b_post_ + (size_t)m * HC_MULT,
                       b_comb_ + (size_t)m * HC_MULT * HC_MULT,
                       b_hcws_ + (size_t)m * hc_workspace_floats(), s);
            rmsnorm(b_normed_ + (size_t)m * HIDDEN, b_coll_ + (size_t)m * HIDDEN,
                    l.ln_post, GEMV_BF16, HIDDEN, s);
        }
        if (l.moe) {
            // THE ROUTED EXPERTS ARE NOT BATCHED, and that is a measured choice, not an omission.
            // At the widths speculation uses, K tokens select almost disjoint expert sets — 29.4
            // distinct of a possible 32 at K=4 — so batching them would save 4.7% (ROOFLINE §4).
            // It is worth doing for wide PREFILL chunks, where the 144 experts saturate and the
            // saving reaches 1.9x at K=32. Until then this reuses the already-gated batch-1 path.
            for (int m = 0; m < M; ++m)
                moe_forward(b_normed_ + (size_t)m * HIDDEN, l.ml, b_sub_ + (size_t)m * HIDDEN,
                            sel_, selw_, ws_moe_, s);
        } else {
            dense_mlp_batch(b_normed_, l.dense, b_sub_, b_ws_mlp_, M, s);
        }
        for (int m = 0; m < M; ++m)
            hc_apply(b_streams_ + (size_t)m * HC_MULT * HIDDEN, b_resid_ + (size_t)m * HC_MULT * HIDDEN,
                     b_sub_ + (size_t)m * HIDDEN, b_post_ + (size_t)m * HC_MULT,
                     b_comb_ + (size_t)m * HC_MULT * HC_MULT, s);
    }

    if (!logits) return;
    const int first = all_logits ? 0 : M - 1;
    for (int m = first; m < M; ++m) {
        hc_head_mean(b_streams_ + (size_t)m * HC_MULT * HIDDEN, b_pooled_ + (size_t)m * HIDDEN, s);
        rmsnorm(b_pooled_ + (size_t)m * HIDDEN, b_pooled_ + (size_t)m * HIDDEN,
                final_norm_, GEMV_BF16, HIDDEN, s);
    }
    gemm(logits, lm_head_, b_pooled_ + (size_t)first * HIDDEN, M - first, VOCAB, HIDDEN, GEMV_BF16, s);
}

void Engine::decode(int token_id, int pos, float* logits, cudaStream_t s) {
    if (pos >= cfg_.max_ctx) { fprintf(stderr, "engine: pos %d >= max_ctx %d\n", pos, cfg_.max_ctx); abort(); }
    if (token_id < 0 || token_id >= VOCAB) { fprintf(stderr, "engine: token %d out of range\n", token_id); abort(); }

    k_embed_broadcast<<<(HIDDEN + 255) / 256, 256, 0, s>>>(streams_, (const __nv_bfloat16*)embed_, token_id);

    for (int i = 0; i < cfg_.n_layer; ++i) {
        LayerW& l = L_[i];

        // ---- attention site ----
        CU(cudaMemcpyAsync(resid_, streams_, (size_t)HC_MULT * HIDDEN * 4, cudaMemcpyDeviceToDevice, s));
        hc_compose(streams_, l.hc_attn, coll_, post_, comb_, hcws_, s);
        rmsnorm(normed_, coll_, l.ln_in, GEMV_BF16, HIDDEN, s);
        if (l.kda)
            kda_decode_step(normed_, l.kw,
                            kda_conv_ + (size_t)l.kda_slot * KDA_CONV_PER_LAYER,
                            kda_state_ + (size_t)l.kda_slot * KDA_STATE_PER_LAYER,
                            sub_, ws_kda_, s);
        else
            mla_decode_step(normed_, l.mw,
                            mla_cache_ + (size_t)l.mla_slot * cfg_.max_ctx * MLA_KV_LORA,
                            pos, cfg_.max_ctx, sub_, ws_mla_, s);
        hc_apply(streams_, resid_, sub_, post_, comb_, s);

        // ---- MLP site ----
        CU(cudaMemcpyAsync(resid_, streams_, (size_t)HC_MULT * HIDDEN * 4, cudaMemcpyDeviceToDevice, s));
        hc_compose(streams_, l.hc_ffn, coll_, post_, comb_, hcws_, s);
        rmsnorm(normed_, coll_, l.ln_post, GEMV_BF16, HIDDEN, s);
        if (l.moe) moe_forward(normed_, l.ml, sub_, sel_, selw_, ws_moe_, s);
        else       dense_mlp(normed_, l.dense, sub_, ws_mlp_, s);
        hc_apply(streams_, resid_, sub_, post_, comb_, s);
    }

    if (!logits) return;                                    // stack gate stops here
    hc_head_mean(streams_, pooled_, s);                     // unweighted mean over the 4 streams
    rmsnorm(pooled_, pooled_, final_norm_, GEMV_BF16, HIDDEN, s);
    gemv(logits, lm_head_, pooled_, VOCAB, HIDDEN, GEMV_BF16, s);
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
    const int C = cfg_.max_batch < 1 ? 1 : cfg_.max_batch;
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

    int next = sample(logits_host_, VOCAB, p.sampling, rng, scratch_);
    for (int n = 0; n < p.max_tokens; ++n) {
        if (is_eos(next)) { st.hit_eos = true; break; }     // EOS never reaches the callback
        ++st.completion_tokens;
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
