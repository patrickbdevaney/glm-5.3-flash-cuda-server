// gate_moe.cu — hold the MoE block to the PyTorch reference, reading the REAL packed NVFP4
// experts out of the checkpoint shards.
//
//   cd ~/glm-5.3-reap && ./.venv/bin/python ~/glm-5.3-flash-cuda-server/ref/gen_moe.py
//   ./build/gate_moe [model_dir] [ref_dir]
//
// Nothing here is fed a convenient fp32 copy: the 4-bit unpack, the fp8 scales, the per-tensor
// global scale and the loader are all on the hook.
#include "moe.h"
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

static std::string RD = "ref/moe";
static int failures = 0, checks = 0;

static std::vector<float> load(const char* n, size_t expect = 0) {
    std::string p = RD + "/" + n; FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "FATAL: oracle missing: %s\n  run ref/gen_moe.py first\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END); long b = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v(b / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fprintf(stderr, "short read %s\n", p.c_str()); exit(2); }
    fclose(f);
    if (expect && v.size() != expect) { fprintf(stderr, "FATAL: %s size %zu != %zu\n", n, v.size(), expect); exit(2); }
    return v;
}
static std::vector<int32_t> load_i32(const char* n) {
    std::string p = RD + "/" + n; FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "FATAL: %s missing\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END); long b = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<int32_t> v(b / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) exit(2);
    fclose(f); return v;
}

static void check(const char* what, const float* dev, const std::vector<float>& ref,
                  double cos_min = 1 - 1e-6, double rel_max = 4e-3) {
    ++checks;
    std::vector<float> got(ref.size());
    CU(cudaMemcpy(got.data(), dev, ref.size() * 4, cudaMemcpyDeviceToHost));
    double dot = 0, na = 0, nb = 0, worst = 0, rms = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double a = got[i], b = ref[i];
        if (!std::isfinite(a)) { printf("  %-16s FAIL non-finite at %zu\n", what, i); ++failures; return; }
        dot += a * b; na += a * a; nb += b * b; rms += (a - b) * (a - b);
    }
    rms = std::sqrt(rms / ref.size());
    const double scale = std::sqrt(nb / ref.size());
    for (size_t i = 0; i < ref.size(); ++i)
        worst = std::max(worst, std::fabs((double)got[i] - ref[i]) / (std::fabs((double)ref[i]) + scale));
    const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    const bool ok = cos >= cos_min && worst <= rel_max;
    if (!ok) ++failures;
    printf("  %-16s %s  cos %.9f  max_rel %.3e  rms %.3e  (n=%zu)\n",
           what, ok ? "PASS" : "FAIL", cos, worst, rms, ref.size());
}

int main(int argc, char** argv) {
    std::string model = argc > 1 ? argv[1]
        : std::string(getenv("HOME")) + "/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2";
    if (argc > 2) RD = argv[2];
    const int L = 3;            // first sparse layer; must match ref/gen_moe.py --layer

    printf("gate_moe — MoE block vs transformers, real NVFP4 experts from the checkpoint\n");
    printf("model:  %s\n  layer %d\noracle: %s\n\n", model.c_str(), L, RD.c_str());

    char pfx[128]; snprintf(pfx, sizeof pfx, "model.language_model.layers.%d.mlp.", L);
    char onlyp[64]; snprintf(onlyp, sizeof onlyp, "layers.%d.mlp.", L);
    st::WeightStore WS(model, nullptr, onlyp);
    printf("loaded %zu tensors, %.2f GiB resident\n\n", WS.count(), WS.loadedGiB());

    auto mat = [&](const std::string& base) {
        Nvfp4Mat m;
        m.packed = WS.dev<uint8_t>(base + ".weight_packed");
        m.scale  = WS.dev<uint8_t>(base + ".weight_scale");
        m.gscale = WS.dev<float>(base + ".weight_global_scale");
        return m;
    };

    std::vector<Nvfp4Mat> host_e(3 * N_ROUTED_EXPERT), host_s(3);
    for (int e = 0; e < N_ROUTED_EXPERT; ++e) {
        std::string b = std::string(pfx) + "experts." + std::to_string(e) + ".";
        host_e[e * 3 + 0] = mat(b + "gate_proj");
        host_e[e * 3 + 1] = mat(b + "up_proj");
        host_e[e * 3 + 2] = mat(b + "down_proj");
    }
    host_s[0] = mat(std::string(pfx) + "shared_experts.gate_proj");
    host_s[1] = mat(std::string(pfx) + "shared_experts.up_proj");
    host_s[2] = mat(std::string(pfx) + "shared_experts.down_proj");

    Nvfp4Mat *d_e, *d_s;
    CU(cudaMalloc(&d_e, host_e.size() * sizeof(Nvfp4Mat)));
    CU(cudaMalloc(&d_s, 3 * sizeof(Nvfp4Mat)));
    CU(cudaMemcpy(d_e, host_e.data(), host_e.size() * sizeof(Nvfp4Mat), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_s, host_s.data(), 3 * sizeof(Nvfp4Mat), cudaMemcpyHostToDevice));

    // e_score_correction_bias is fp32 in the checkpoint; the router weight is bf16.
    MoeLayer ML{};
    ML.router_w    = WS.get(std::string(pfx) + "gate.weight").dev;
    ML.router_bias = WS.dev<float>(std::string(pfx) + "gate.e_score_correction_bias");
    ML.experts     = d_e;
    ML.shared      = d_s;
    ML.n_expert    = N_ROUTED_EXPERT;
    ML.topk        = N_EXPERT_PER_TOK;

    auto x        = load("x.bin", HIDDEN);
    auto logit_r  = load("logits.bin", N_ROUTED_EXPERT);
    auto wts_r    = load("topk_w.bin", N_EXPERT_PER_TOK);
    auto y_r      = load("y.bin", HIDDEN);
    auto idx_r    = load_i32("topk_idx.bin");

    float* d_x; CU(cudaMalloc(&d_x, HIDDEN * 4));
    CU(cudaMemcpy(d_x, x.data(), HIDDEN * 4, cudaMemcpyHostToDevice));
    float *d_y, *d_w, *d_ws; int32_t* d_sel;
    CU(cudaMalloc(&d_y, HIDDEN * 4));
    CU(cudaMalloc(&d_w, N_EXPERT_PER_TOK * 4));
    CU(cudaMalloc(&d_sel, N_EXPERT_PER_TOK * 4));
    CU(cudaMalloc(&d_ws, moe_workspace_floats() * 4));

    moe_forward(d_x, ML, d_y, d_sel, d_w, d_ws, 0);
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());

    // routing: compare as SETS. transformers' topk(sorted=False) fixes no order, and neither the
    // reference nor the kernel promises one; only the chosen set and their weights are meaningful.
    std::vector<int32_t> sel(N_EXPERT_PER_TOK);
    std::vector<float> gw(N_EXPERT_PER_TOK);
    CU(cudaMemcpy(sel.data(), d_sel, sel.size() * 4, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(gw.data(), d_w, gw.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<std::pair<int, float>> a, b;
    for (int i = 0; i < N_EXPERT_PER_TOK; ++i) { a.push_back({sel[i], gw[i]}); b.push_back({idx_r[i], wts_r[i]}); }
    std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
    ++checks;
    bool routing_ok = true;
    double wmax = 0;
    for (int i = 0; i < N_EXPERT_PER_TOK; ++i) {
        if (a[i].first != b[i].first) routing_ok = false;
        wmax = std::max(wmax, (double)std::fabs(a[i].second - b[i].second));
    }
    if (!routing_ok || wmax > 1e-5) ++failures;
    printf("  %-16s %s  experts", "routing", (routing_ok && wmax <= 1e-5) ? "PASS" : "FAIL");
    for (auto& p : a) printf(" %d", p.first);
    printf("   max|dw| %.2e\n", wmax);
    if (!routing_ok) {
        printf("      got     "); for (auto& p : a) printf(" %d", p.first);
        printf("\n      expected"); for (auto& p : b) printf(" %d", p.first); printf("\n");
    }

    float* d_logit; CU(cudaMalloc(&d_logit, N_ROUTED_EXPERT * 4));
    gemv(d_logit, ML.router_w, d_x, N_ROUTED_EXPERT, HIDDEN, GEMV_BF16, 0);
    CU(cudaDeviceSynchronize());
    check("router logits", d_logit, logit_r);
    check("moe output y", d_y, y_r);

    printf("\n%s  %d/%d checks passed\n", failures ? "GATE FAILED" : "GATE PASSED",
           checks - failures, checks);
    return failures ? 1 : 0;
}
