// gate_kda.cu — hold the KDA decode kernels to the PyTorch reference, on REAL layer-0 weights.
//
// Generate the oracle first:
//   cd ~/glm-5.3-reap && ./.venv/bin/python ~/glm-5.3-flash-cuda-server/ref/gen_kda.py
//
// Every stage is checked separately, so a failure names the stage rather than the layer. Exits
// non-zero on ANY failure and on a missing oracle — a gate that passes vacuously is worse than
// no gate (CLAUDE.md §2).
#include "kda.h"
#include "gemv.h"
#include "glm5_config.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

using namespace glm5;
#define CU(x) do { cudaError_t e_ = (x); if (e_) { \
    fprintf(stderr, "cuda %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

static std::string DIR = "ref/kda";
static int failures = 0, checks = 0;

static std::vector<float> load(const char* name, size_t expect = 0) {
    std::string p = DIR + "/" + name;
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "FATAL: oracle missing: %s\n  run ref/gen_kda.py first\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v(n / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fprintf(stderr, "FATAL: short read %s\n", p.c_str()); exit(2); }
    fclose(f);
    if (expect && v.size() != expect) {
        fprintf(stderr, "FATAL: %s has %zu floats, expected %zu\n", name, v.size(), expect); exit(2);
    }
    return v;
}

static float* to_dev(const std::vector<float>& h) {
    float* d; CU(cudaMalloc(&d, h.size() * 4)); CU(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

// cosine + max relative error against the reference, scaled by the reference's own magnitude so
// that near-zero coordinates do not manufacture huge relative errors.
static void check(const char* what, const float* dev, const std::vector<float>& ref,
                  double cos_min = 1 - 1e-6, double rel_max = 4e-3) {
    ++checks;
    std::vector<float> got(ref.size());
    CU(cudaMemcpy(got.data(), dev, ref.size() * 4, cudaMemcpyDeviceToHost));
    double dot = 0, na = 0, nb = 0, worst = 0, rms = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double a = got[i], b = ref[i];
        if (!std::isfinite(a)) { printf("  %-14s FAIL non-finite at %zu\n", what, i); ++failures; return; }
        dot += a * b; na += a * a; nb += b * b; rms += (a - b) * (a - b);
    }
    rms = std::sqrt(rms / ref.size());
    const double scale = std::sqrt(nb / ref.size());     // reference RMS
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = std::fabs((double)got[i] - ref[i]) / (std::fabs((double)ref[i]) + scale);
        if (d > worst) worst = d;
    }
    const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    const bool ok = cos >= cos_min && worst <= rel_max;
    if (!ok) ++failures;
    printf("  %-14s %s  cos %.9f  max_rel %.3e  rms %.3e  (n=%zu)\n",
           what, ok ? "PASS" : "FAIL", cos, worst, rms, ref.size());
}

int main(int argc, char** argv) {
    if (argc > 1) DIR = argv[1];
    printf("gate_kda — KDA decode step vs transformers, real layer-0 weights\n");
    printf("oracle: %s\n\n", DIR.c_str());

    const int Hh = KDA_HEADS, D = KDA_HEAD_DIM, Qd = KDA_QKV_DIM;

    // ---- oracle ----
    auto x        = load("x.bin", HIDDEN);
    auto conv_in  = load("conv_state_in.bin", (size_t)3 * Qd * KDA_CONV_STATE);
    auto conv_ref = load("conv_state_out.bin", (size_t)3 * Qd * KDA_CONV_STATE);
    auto S_in     = load("S_in.bin", (size_t)Hh * D * D);
    auto qkv_ref  = load("qkv_conv.bin", (size_t)3 * Qd);
    auto g_ref    = load("g.bin", (size_t)Qd);
    auto beta_ref = load("beta.bin", (size_t)Hh);
    auto qn_ref   = load("q_n.bin", (size_t)Qd);
    auto kn_ref   = load("k_n.bin", (size_t)Qd);
    auto core_ref = load("core_out.bin", (size_t)Qd);
    auto S_ref    = load("S_out.bin", (size_t)Hh * D * D);
    auto gate_ref = load("gate.bin", (size_t)Qd);
    auto norm_ref = load("normed.bin", (size_t)Qd);
    auto y_ref    = load("y.bin", HIDDEN);

    KdaWeights W{};
    W.dtype   = GEMV_F32;
    W.q_proj  = to_dev(load("w_q_proj.bin", (size_t)Qd * HIDDEN));
    W.k_proj  = to_dev(load("w_k_proj.bin", (size_t)Qd * HIDDEN));
    W.v_proj  = to_dev(load("w_v_proj.bin", (size_t)Qd * HIDDEN));
    W.o_proj  = to_dev(load("w_o_proj.bin", (size_t)HIDDEN * Qd));
    W.conv1d  = to_dev(load("w_conv1d.bin", (size_t)3 * Qd * KDA_CONV_K));
    W.f_a     = to_dev(load("w_f_a.bin", (size_t)KDA_GATE_RANK * HIDDEN));
    W.f_b     = to_dev(load("w_f_b.bin", (size_t)Qd * KDA_GATE_RANK));
    W.dt_bias = to_dev(load("w_dt_bias.bin", (size_t)Qd));
    W.A_log   = to_dev(load("w_A_log.bin", (size_t)Hh));
    W.b_proj  = to_dev(load("w_b_proj.bin", (size_t)Hh * HIDDEN));
    W.g_a     = to_dev(load("w_g_a.bin", (size_t)KDA_GATE_RANK * HIDDEN));
    W.g_b     = to_dev(load("w_g_b.bin", (size_t)Qd * KDA_GATE_RANK));
    W.o_norm  = to_dev(load("w_o_norm.bin", (size_t)D));

    float* d_x    = to_dev(x);
    float* d_conv = to_dev(conv_in);
    float* d_S    = to_dev(S_in);
    float *d_y, *d_ws;
    CU(cudaMalloc(&d_y, HIDDEN * 4));
    CU(cudaMalloc(&d_ws, kda_workspace_floats() * 4));
    CU(cudaMemset(d_ws, 0, kda_workspace_floats() * 4));

    // ---- stage by stage, on the same buffers the fused path uses ----
    float* qkv  = d_ws;
    float* farea = d_ws + 3 * (size_t)Qd;
    float* gate = d_ws + 4 * (size_t)Qd + 256;
    float* g    = d_ws + 5 * (size_t)Qd + 256;
    float* beta = d_ws + 6 * (size_t)Qd + 256;
    float* q_n  = d_ws + 6 * (size_t)Qd + 256 + Hh;
    float* k_n  = d_ws + 7 * (size_t)Qd + 256 + Hh;
    float* core = d_ws + 8 * (size_t)Qd + 256 + Hh;
    float* nrm  = d_ws + 9 * (size_t)Qd + 256 + Hh;

    kda_qkv_conv(d_x, W, d_conv, qkv, d_ws, 0);
    CU(cudaDeviceSynchronize());
    check("qkv+conv+silu", qkv, qkv_ref);
    check("conv_state", d_conv, conv_ref);

    kda_gates(d_x, W, g, beta, gate, farea, 0);
    CU(cudaDeviceSynchronize());
    check("forget gate g", g, g_ref);
    check("beta", beta, beta_ref);
    check("output gate", gate, gate_ref);

    kda_norm_qk(qkv, q_n, k_n, 0);
    CU(cudaDeviceSynchronize());
    check("l2norm q", q_n, qn_ref);
    check("l2norm k", k_n, kn_ref);

    kda_recurrence(q_n, k_n, qkv + 2 * (size_t)Qd, g, beta, d_S, core, 0);
    CU(cudaDeviceSynchronize());
    check("recurrence out", core, core_ref);
    check("state S_out", d_S, S_ref);

    kda_out_norm(core, gate, W.o_norm, W.dtype, nrm, 0);
    CU(cudaDeviceSynchronize());
    check("gated o_norm", nrm, norm_ref);

    gemv(d_y, W.o_proj, nrm, HIDDEN, Qd, W.dtype, 0);
    CU(cudaDeviceSynchronize());
    check("o_proj (y)", d_y, y_ref);

    // ---- and the fused entry point, from clean state, end to end ----
    CU(cudaMemcpy(d_conv, conv_in.data(), conv_in.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_S, S_in.data(), S_in.size() * 4, cudaMemcpyHostToDevice));
    kda_decode_step(d_x, W, d_conv, d_S, d_y, d_ws, 0);
    CU(cudaDeviceSynchronize());
    printf("\n  fused kda_decode_step:\n");
    check("  y", d_y, y_ref);
    check("  S_out", d_S, S_ref);

    // ---- production dtype: bf16 weights, fp32 accumulate ----
    //
    // The checkpoint IS bf16; the oracle merely widened it. So this is not a precision experiment,
    // it is a check that the bf16 GEMV path indexes and unpacks the same numbers. Anything worse
    // than the fp32 path here is a layout bug, not rounding.
    KdaWeights B = W;
    B.dtype = GEMV_BF16;
    auto mk_bf16 = [&](const void* f32, size_t n) -> void* {
        void* d; CU(cudaMalloc(&d, n * 2)); f32_to_bf16(d, (const float*)f32, n, 0); return d;
    };
    B.q_proj = mk_bf16(W.q_proj.p, (size_t)Qd * HIDDEN);
    B.k_proj = mk_bf16(W.k_proj.p, (size_t)Qd * HIDDEN);
    B.v_proj = mk_bf16(W.v_proj.p, (size_t)Qd * HIDDEN);
    B.o_proj = mk_bf16(W.o_proj.p, (size_t)HIDDEN * Qd);
    B.f_a    = mk_bf16(W.f_a.p, (size_t)KDA_GATE_RANK * HIDDEN);
    B.f_b    = mk_bf16(W.f_b.p, (size_t)Qd * KDA_GATE_RANK);
    B.g_a    = mk_bf16(W.g_a.p, (size_t)KDA_GATE_RANK * HIDDEN);
    B.g_b    = mk_bf16(W.g_b.p, (size_t)Qd * KDA_GATE_RANK);
    B.b_proj = mk_bf16(W.b_proj.p, (size_t)Hh * HIDDEN);
    B.o_norm = mk_bf16(W.o_norm, (size_t)D);
    // conv1d stays fp32: it is [24576, 4], 393 KB, and k_conv_silu reads it as fp32 by design.
    CU(cudaDeviceSynchronize());

    CU(cudaMemcpy(d_conv, conv_in.data(), conv_in.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_S, S_in.data(), S_in.size() * 4, cudaMemcpyHostToDevice));
    kda_decode_step(d_x, B, d_conv, d_S, d_y, d_ws, 0);
    CU(cudaDeviceSynchronize());
    printf("\n  bf16 weights (production dtype):\n");
    check("  y", d_y, y_ref);
    check("  S_out", d_S, S_ref);

    printf("\n%s  %d/%d checks passed\n", failures ? "GATE FAILED" : "GATE PASSED",
           checks - failures, checks);
    return failures ? 1 : 0;
}
