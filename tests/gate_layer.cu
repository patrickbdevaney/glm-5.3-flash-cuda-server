// gate_layer.cu — a COMPLETE decoder layer against transformers: two hyper-connection sites, the
// KDA sublayer, the dense MLP, and the four-stream residual mix.
//
//   cd ~/glm-5.3-reap && ./.venv/bin/python ~/glm-5.3-flash-cuda-server/ref/gen_layer.py
//   ./build/gate_layer [model_dir] [ref_dir]
//
// Every weight is read from the checkpoint. This is the gate that catches wiring, not arithmetic:
// stream ordering, the comb TRANSPOSE in the residual mix, which norm feeds which site.
#include "kda.h"
#include "layer.h"
#include "gemv.h"
#include "glm5_config.h"
#include "weight_store.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

using namespace glm5;
#define CU(x) do { cudaError_t e_=(x); if(e_){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(2);} } while(0)

static std::string RD = "ref/layer";
static int failures = 0, checks = 0;

static std::vector<float> load(const char* n, size_t expect = 0) {
    std::string p = RD + "/" + n; FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "FATAL: oracle missing: %s\n  run ref/gen_layer.py first\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END); long b = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v(b / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) exit(2);
    fclose(f);
    if (expect && v.size() != expect) { fprintf(stderr, "FATAL: %s size %zu != %zu\n", n, v.size(), expect); exit(2); }
    return v;
}
static float* to_dev(const std::vector<float>& h) {
    float* d; CU(cudaMalloc(&d, h.size() * 4));
    CU(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); return d;
}
static void check(const char* what, const float* dev, const std::vector<float>& ref,
                  double cos_min = 1 - 1e-6, double rel_max = 4e-3) {
    ++checks;
    std::vector<float> got(ref.size());
    CU(cudaMemcpy(got.data(), dev, ref.size() * 4, cudaMemcpyDeviceToHost));
    double dot = 0, na = 0, nb = 0, worst = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double a = got[i], b = ref[i];
        if (!std::isfinite(a)) { printf("  %-14s FAIL non-finite at %zu\n", what, i); ++failures; return; }
        dot += a * b; na += a * a; nb += b * b;
    }
    const double scale = std::sqrt(nb / ref.size());
    for (size_t i = 0; i < ref.size(); ++i)
        worst = std::max(worst, std::fabs((double)got[i] - ref[i]) / (std::fabs((double)ref[i]) + scale));
    const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    const bool ok = cos >= cos_min && worst <= rel_max;
    if (!ok) ++failures;
    printf("  %-14s %s  cos %.9f  max_rel %.3e  (n=%zu)\n", what, ok ? "PASS" : "FAIL", cos, worst, ref.size());
}

int main(int argc, char** argv) {
    std::string model = argc > 1 ? argv[1]
        : std::string(getenv("HOME")) + "/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2";
    if (argc > 2) RD = argv[2];
    const int L = 0;

    printf("gate_layer — complete decoder layer %d (KDA + dense MLP + 2x mHC) vs transformers\n\n", L);
    char onlyp[64]; snprintf(onlyp, sizeof onlyp, "layers.%d.", L);
    st::WeightStore WS(model, nullptr, onlyp);
    char P[128]; snprintf(P, sizeof P, "model.language_model.layers.%d.", L);
    printf("loaded %zu tensors, %.2f GiB resident\n\n", WS.count(), WS.loadedGiB());
    auto N = [&](const char* suf) { return std::string(P) + suf; };

    // KDA conv weights ship SPLIT as q_/k_/v_conv1d [8192,1,4]; the reference has one [24576,1,4].
    // Concatenate in q,k,v order and widen to fp32 — k_conv_silu reads it as fp32 by design, and
    // it is 393 KB, far too small to be worth a bf16 path.
    float* d_conv_w;
    {
        const size_t per = (size_t)KDA_QKV_DIM * KDA_CONV_K;
        CU(cudaMalloc(&d_conv_w, 3 * per * 4));
        const char* nm[3] = {"self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight"};
        for (int i = 0; i < 3; ++i) {
            const auto& t = WS.get(N(nm[i]));
            if ((size_t)t.numel() != per) { fprintf(stderr, "FATAL: %s numel %lld != %zu\n", nm[i], (long long)t.numel(), per); return 2; }
            f32_from_bf16_dev(d_conv_w + i * per, t.dev, per, 0);
        }
        CU(cudaDeviceSynchronize());
    }

    KdaWeights W{};
    W.dtype   = GEMV_BF16;
    W.q_proj  = WS.get(N("self_attn.q_proj.weight")).dev;
    W.k_proj  = WS.get(N("self_attn.k_proj.weight")).dev;
    W.v_proj  = WS.get(N("self_attn.v_proj.weight")).dev;
    W.o_proj  = WS.get(N("self_attn.o_proj.weight")).dev;
    W.conv1d  = d_conv_w;
    W.f_a     = WS.get(N("self_attn.f_a_proj.weight")).dev;
    W.f_b     = WS.get(N("self_attn.f_b_proj.weight")).dev;
    W.dt_bias = WS.dev<float>(N("self_attn.dt_bias"));
    W.A_log   = WS.dev<float>(N("self_attn.A_log"));
    W.b_proj  = WS.get(N("self_attn.b_proj.weight")).dev;
    W.g_a     = WS.get(N("self_attn.g_a_proj.weight")).dev;
    W.g_b     = WS.get(N("self_attn.g_b_proj.weight")).dev;
    W.o_norm  = WS.get(N("self_attn.o_norm.weight")).dev;

    HcWeights HA{WS.get(N("hc_attn_fn")).dev, WS.dev<float>(N("hc_attn_base")), WS.dev<float>(N("hc_attn_scale"))};
    HcWeights HF{WS.get(N("hc_ffn_fn")).dev,  WS.dev<float>(N("hc_ffn_base")),  WS.dev<float>(N("hc_ffn_scale"))};
    DenseMlp MP{WS.get(N("mlp.gate_proj.weight")).dev, WS.get(N("mlp.up_proj.weight")).dev,
                WS.get(N("mlp.down_proj.weight")).dev, DENSE_INTER, GEMV_BF16};
    const void* ln_in  = WS.get(N("input_layernorm.weight")).dev;
    const void* ln_post = WS.get(N("post_attention_layernorm.weight")).dev;

    // ---- oracle ----
    const size_t SD = (size_t)HC_MULT * HIDDEN;
    auto streams_in = load("streams_in.bin", SD);
    auto conv_in = load("conv_state_in.bin", (size_t)3 * KDA_QKV_DIM * KDA_CONV_STATE);
    auto S_in    = load("S_in.bin", (size_t)KDA_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM);

    float* d_streams = to_dev(streams_in);
    float* d_resid   = to_dev(streams_in);
    float* d_conv    = to_dev(conv_in);
    float* d_S       = to_dev(S_in);
    float *d_coll, *d_post, *d_comb, *d_n, *d_sub, *d_ws, *d_mlpws, *d_hcws;
    CU(cudaMalloc(&d_coll, HIDDEN * 4)); CU(cudaMalloc(&d_post, HC_MULT * 4));
    CU(cudaMalloc(&d_comb, HC_MULT * HC_MULT * 4)); CU(cudaMalloc(&d_n, HIDDEN * 4));
    CU(cudaMalloc(&d_sub, HIDDEN * 4));
    CU(cudaMalloc(&d_ws, kda_workspace_floats() * 4));
    CU(cudaMalloc(&d_mlpws, dense_mlp_workspace_floats() * 4));
    CU(cudaMalloc(&d_hcws, hc_workspace_floats() * 4));

    // ---- attention site ----
    hc_compose(d_streams, HA, d_coll, d_post, d_comb, d_hcws, 0);
    CU(cudaDeviceSynchronize());
    check("hc_a post", d_post, load("post_a.bin", HC_MULT));
    check("hc_a comb", d_comb, load("comb_a.bin", HC_MULT * HC_MULT));
    check("hc_a collapse", d_coll, load("coll_a.bin", HIDDEN));

    rmsnorm(d_n, d_coll, ln_in, GEMV_BF16, HIDDEN, 0);
    CU(cudaDeviceSynchronize());
    check("input_ln", d_n, load("n_a.bin", HIDDEN));

    kda_decode_step(d_n, W, d_conv, d_S, d_sub, d_ws, 0);
    CU(cudaDeviceSynchronize());
    check("kda sublayer", d_sub, load("sub_a.bin", HIDDEN));

    hc_apply(d_streams, d_resid, d_sub, d_post, d_comb, 0);
    CU(cudaDeviceSynchronize());
    check("residual mix", d_streams, load("mid.bin", SD));

    // ---- MLP site ----
    CU(cudaMemcpy(d_resid, d_streams, SD * 4, cudaMemcpyDeviceToDevice));
    hc_compose(d_streams, HF, d_coll, d_post, d_comb, d_hcws, 0);
    CU(cudaDeviceSynchronize());
    check("hc_f post", d_post, load("post_f.bin", HC_MULT));
    check("hc_f comb", d_comb, load("comb_f.bin", HC_MULT * HC_MULT));
    check("hc_f collapse", d_coll, load("coll_f.bin", HIDDEN));

    rmsnorm(d_n, d_coll, ln_post, GEMV_BF16, HIDDEN, 0);
    CU(cudaDeviceSynchronize());
    check("post_attn_ln", d_n, load("n_f.bin", HIDDEN));

    dense_mlp(d_n, MP, d_sub, d_mlpws, 0);
    CU(cudaDeviceSynchronize());
    check("dense mlp", d_sub, load("sub_f.bin", HIDDEN));

    hc_apply(d_streams, d_resid, d_sub, d_post, d_comb, 0);
    CU(cudaDeviceSynchronize());
    check("LAYER OUTPUT", d_streams, load("out.bin", SD));

    printf("\n%s  %d/%d checks passed\n", failures ? "GATE FAILED" : "GATE PASSED",
           checks - failures, checks);
    return failures ? 1 : 0;
}
