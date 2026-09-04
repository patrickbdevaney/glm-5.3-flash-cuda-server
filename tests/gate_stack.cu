// gate_stack.cu — the ENGINE against transformers over a stack of layers and several decode steps.
//
//   cd ~/glm-5.3-reap && ./.venv/bin/python ~/glm-5.3-flash-cuda-server/ref/gen_stack.py
//   ./build/gate_stack [model_dir] [ref_dir]
//
// The per-kernel gates prove the kernels. This proves what is BETWEEN them: that the four mHC
// residual streams carry from layer to layer, that each KDA layer keeps its own recurrent state
// and conv window rather than sharing one, that the embedding is broadcast to all four streams,
// and that state evolves correctly across successive decode steps.
#include "engine.h"
#include "glm5_config.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

using namespace glm5;
#define CU(x) do { cudaError_t e_=(x); if(e_){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(2);} } while(0)

static std::string RD = "ref/stack";
static int failures = 0, checks = 0;

static std::vector<float> load(const char* n, size_t expect = 0) {
    std::string p = RD + "/" + n; FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "FATAL: oracle missing: %s\n  run ref/gen_stack.py first\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END); long b = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v(b / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) exit(2);
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
static int meta_int(const char* key) {
    std::string p = RD + "/meta.txt"; FILE* f = fopen(p.c_str(), "r");
    if (!f) { fprintf(stderr, "FATAL: %s missing\n", p.c_str()); exit(2); }
    char line[256]; int v = -1;
    while (fgets(line, sizeof line, f)) {
        char* eq = strchr(line, '='); if (!eq) continue; *eq = 0;
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
    printf("  %-14s %s  cos %.9f  max_rel %.3e  |got|=%.5f\n",
           what, ok ? "PASS" : "FAIL", cos, worst, std::sqrt(na));
}

int main(int argc, char** argv) {
    EngineConfig cfg;
    cfg.model_dir = argc > 1 ? argv[1]
        : std::string(getenv("HOME")) + "/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2";
    if (argc > 2) RD = argv[2];
    cfg.n_layer = meta_int("layers");
    const int steps = meta_int("steps");

    printf("gate_stack — engine vs transformers, %d layers, %d decode steps\n\n", cfg.n_layer, steps);
    Engine E(cfg);
    E.reset();
    auto ids = load_i32("ids.bin");
    const size_t SD = (size_t)HC_MULT * HIDDEN;

    for (int t = 0; t < steps; ++t) {
        E.decode(ids[t], t, nullptr);
        CU(cudaDeviceSynchronize());
        CU(cudaGetLastError());
        char nm[32], lbl[32];
        snprintf(nm, sizeof nm, "streams_%d.bin", t);
        snprintf(lbl, sizeof lbl, "step %d tok %d", t, ids[t]);
        check(lbl, E.streamsDev(), load(nm, SD));
    }

    printf("\n%s  %d/%d checks passed\n", failures ? "GATE FAILED" : "GATE PASSED",
           checks - failures, checks);
    return failures ? 1 : 0;
}
