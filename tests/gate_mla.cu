// gate_mla.cu — MLA full-attention decode vs transformers, on real layer-3 weights.
//
//   cd ~/glm-5.3-reap && ./.venv/bin/python ~/glm-5.3-flash-cuda-server/ref/gen_mla.py
//   ./build/gate_mla [model_dir] [ref_dir]
#include "mla.h"
#include "gemv.h"
#include "layer.h"
#include "glm5_config.h"
#include "weight_store.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <cstring>
#include <vector>
#include <algorithm>

using namespace glm5;
#define CU(x) do { cudaError_t e_=(x); if(e_){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(2);} } while(0)

static std::string RD = "ref/mla";
static int failures = 0, checks = 0;

static std::vector<float> load(const char* n, size_t expect = 0) {
    std::string p = RD + "/" + n; FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "FATAL: oracle missing: %s\n  run ref/gen_mla.py first\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END); long b = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v(b / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) exit(2);
    fclose(f);
    if (expect && v.size() != expect) { fprintf(stderr, "FATAL: %s size %zu != %zu\n", n, v.size(), expect); exit(2); }
    return v;
}
static int meta_int(const char* key) {
    std::string p = RD + "/meta.txt"; FILE* f = fopen(p.c_str(), "r");
    if (!f) { fprintf(stderr, "FATAL: %s missing\n", p.c_str()); exit(2); }
    char line[256]; int v = -1;
    while (fgets(line, sizeof line, f)) {
        char* eq = strchr(line, '=');
        if (!eq) continue;
        *eq = 0;
        if (!strcmp(line, key)) { v = atoi(eq + 1); break; }
    }
    fclose(f);
    if (v < 0) { fprintf(stderr, "FATAL: %s not in meta.txt\n", key); exit(2); }
    return v;
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
    const double sc = std::sqrt(nb / ref.size());
    for (size_t i = 0; i < ref.size(); ++i)
        worst = std::max(worst, std::fabs((double)got[i] - ref[i]) / (std::fabs((double)ref[i]) + sc));
    const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    const bool ok = cos >= cos_min && worst <= rel_max;
    if (!ok) ++failures;
    printf("  %-14s %s  cos %.9f  max_rel %.3e  (n=%zu)\n", what, ok ? "PASS" : "FAIL", cos, worst, ref.size());
}

int main(int argc, char** argv) {
    std::string model = argc > 1 ? argv[1]
        : std::string(getenv("HOME")) + "/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2";
    if (argc > 2) RD = argv[2];
    const int L = meta_int("layer"), T = meta_int("T");
    printf("gate_mla — MLA decode (pure NoPE, absorbed) vs transformers, layer %d, T=%d\n\n", L, T);

    char onlyp[64]; snprintf(onlyp, sizeof onlyp, "layers.%d.self_attn.", L);
    st::WeightStore WS(model, nullptr, onlyp);
    char P[128]; snprintf(P, sizeof P, "model.language_model.layers.%d.self_attn.", L);
    auto N = [&](const char* s) { return std::string(P) + s; };
    printf("loaded %zu tensors, %.2f GiB resident\n\n", WS.count(), WS.loadedGiB());

    MlaWeights W{};
    W.dtype     = GEMV_BF16;
    W.q_a       = WS.get(N("q_a_proj.weight")).dev;
    W.q_a_norm  = WS.get(N("q_a_layernorm.weight")).dev;
    W.q_b       = WS.get(N("q_b_proj.weight")).dev;
    W.kv_a      = WS.get(N("kv_a_proj_with_mqa.weight")).dev;
    W.kv_a_norm = WS.get(N("kv_a_layernorm.weight")).dev;
    W.kv_b      = WS.get(N("kv_b_proj.weight")).dev;
    W.o_proj    = WS.get(N("o_proj.weight")).dev;

    auto x       = load("x.bin", HIDDEN);
    auto c_all   = load("c_kv_cache.bin", (size_t)T * MLA_KV_LORA);
    auto q_resid = load("q_resid.bin", MLA_Q_LORA);
    auto q_ref   = load("q.bin", (size_t)MLA_HEADS * MLA_QK_NOPE);
    auto y_ref   = load("y.bin", HIDDEN);

    float *d_x, *d_cache, *d_y, *d_ws;
    CU(cudaMalloc(&d_x, HIDDEN * 4));
    CU(cudaMemcpy(d_x, x.data(), HIDDEN * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&d_cache, (size_t)T * MLA_KV_LORA * 4));
    // seed the cache with the T-1 earlier tokens; the step writes row T-1 itself, so leave it dirty
    // — if the kernel failed to store the new latent, the gate would catch it.
    CU(cudaMemset(d_cache, 0x7f, (size_t)T * MLA_KV_LORA * 4));
    CU(cudaMemcpy(d_cache, c_all.data(), (size_t)(T - 1) * MLA_KV_LORA * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&d_y, HIDDEN * 4));
    CU(cudaMalloc(&d_ws, mla_workspace_floats(T) * 4));

    mla_decode_step(d_x, W, d_cache, T - 1, T, d_y, d_ws, 0);
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());

    check("q_resid", d_ws, q_resid);
    check("q", d_ws + MLA_Q_LORA, q_ref);
    // the latent the step just stored must equal the reference's last cache row
    std::vector<float> c_last(c_all.end() - MLA_KV_LORA, c_all.end());
    check("stored latent", d_cache + (size_t)(T - 1) * MLA_KV_LORA, c_last);
    check("attn out (y)", d_y, y_ref);

    printf("\n%s  %d/%d checks passed\n", failures ? "GATE FAILED" : "GATE PASSED",
           checks - failures, checks);
    return failures ? 1 : 0;
}
