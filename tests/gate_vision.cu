// gate_vision.cu — the vision tower against transformers' own Glm5NextVisionModel.
//
// Staged on purpose. A single end-to-end number cannot say WHICH of patch_embed, the blocks, the
// downsample or the merger is wrong, and this tower has four ways to differ from the language
// model (clamped SwiGLU, LayerNorm-with-bias, per-head q/k RMSNorm, bidirectional attention) that
// all produce plausible-looking output when got wrong.
//
// cos/sin come from the oracle so that a failure here is the TOWER, not position-id generation;
// those ids are checked separately below.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "vision.h"
#include "weight_store.h"
#include "glm5_config.h"

using namespace glm5;
#define CU(x) do { cudaError_t e_=(x); if(e_){ fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(e_)); exit(2);} } while(0)

static std::vector<float> readbin(const std::string& p, size_t n) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "MISSING %s — run ref/gen_vision.py first\n", p.c_str()); exit(2); }
    std::vector<float> v(n);
    if (fread(v.data(), 4, n, f) != n) { fprintf(stderr, "SHORT %s\n", p.c_str()); exit(2); }
    fclose(f); return v;
}
static int g_pass = 0, g_fail = 0;
// RELATIVE L2, not max-per-element. A tensor of a million activations always contains a few near
// zero, and dividing an absolute error by one of those reports a huge number for a correct kernel
// -- the trap that rejected the whole (HG,NT) grid in OPTIMIZATION_LOG #16.
static void cmp(const char* name, const std::vector<float>& got, const std::vector<float>& want,
                double cos_min, double rel_max) {
    double dot = 0, na = 0, nb = 0, d2 = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        dot += (double)got[i] * want[i]; na += (double)got[i] * got[i]; nb += (double)want[i] * want[i];
        const double e = (double)got[i] - want[i]; d2 += e * e;
    }
    const double c = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    const double rel = sqrt(d2) / (sqrt(nb) + 1e-30);
    const bool ok = c >= cos_min && rel <= rel_max;
    printf("  %-16s %s  cos %.9f  relL2 %.3e  (n=%zu)\n", name, ok ? "PASS" : "FAIL", c, rel, want.size());
    ok ? ++g_pass : ++g_fail;
}

int main(int argc, char** argv) {
    const std::string ck = argc > 1 ? argv[1]
        : std::string(getenv("HOME")) + "/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2";
    // GLM5_VIS_REF=fp32 compares against the fp32 oracle. The engine keeps fp32 activations, so the
    // bf16 oracle is the less precise reference and its error compounds across 24 blocks.
    // DEFAULTS TO THE fp32 ORACLE, because that is the correct comparison: this engine keeps fp32
    // activations with bf16 weights, so a bf16 reference is the LESS precise of the two and its
    // rounding compounds over 24 blocks. Against bf16 the tower reads cos 0.9972 and looks broken;
    // against fp32 it is cos 1.000000000. GLM5_VIS_REF=bf16 selects the other one.
    const char* rd = getenv("GLM5_VIS_REF");
    const std::string R = std::string(getenv("HOME")) + "/glm-5.3-flash-cuda-server/ref/"
                        + ((rd && std::string(rd) == "bf16") ? "vision/" : "vision_fp32/");
    printf("=== gate_vision (24-block tower vs transformers) ===\n");

    st::WeightStore ws(ck, nullptr, "model.visual.");
    printf("loaded %zu vision tensors, %.2f GiB\n", ws.count(), ws.loadedGiB());
    auto T = [&](const std::string& n) { return ws.get("model.visual." + n).dev; };

    VisionWeights W{};
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

    const int S = 1024, NO = S / 4;
    auto h_x   = readbin(R + "input.bin", (size_t)S * VIS_IN_DIM);
    auto h_cos = readbin(R + "cos.bin", (size_t)S * VIS_HEAD_DIM);
    auto h_sin = readbin(R + "sin.bin", (size_t)S * VIS_HEAD_DIM);

    float *d_x, *d_cos, *d_sin, *d_out, *d_ws;
    CU(cudaMalloc(&d_x, h_x.size() * 4));       CU(cudaMemcpy(d_x, h_x.data(), h_x.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&d_cos, h_cos.size() * 4));   CU(cudaMemcpy(d_cos, h_cos.data(), h_cos.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&d_sin, h_sin.size() * 4));   CU(cudaMemcpy(d_sin, h_sin.data(), h_sin.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&d_out, (size_t)NO * VIS_OUT_HIDDEN * 4));
    CU(cudaMalloc(&d_ws, vision_workspace_floats(S) * 4));

    // Bisect: patch_embed, then block 0, then block 1, then the whole tower.
    std::vector<float> stg((size_t)S * VIS_HIDDEN);
    auto stage = [&](const char* env, const char* ref, const char* name, double cm, double rm) {
        setenv("GLM5_VIS_STOP", env, 1);
        vision_forward(d_x, d_cos, d_sin, W, S, d_out, d_ws, 0);
        CU(cudaDeviceSynchronize());
        CU(cudaMemcpy(stg.data(), d_ws, stg.size() * 4, cudaMemcpyDeviceToHost));
        cmp(name, stg, readbin(R + ref, stg.size()), cm, rm);
    };
    stage("0",  "patch_embed.bin", "patch_embed", 0.999995, 5e-3);
    stage("1",  "block0.bin",      "block 0",     0.999990, 5e-3);
    stage("2",  "block1.bin",      "block 1",     0.999985, 6e-3);
    stage("24", "block23.bin",     "all 24 blocks", 0.99900, 5e-2);
    unsetenv("GLM5_VIS_STOP");

    vision_forward(d_x, d_cos, d_sin, W, S, d_out, d_ws, 0);
    CU(cudaDeviceSynchronize());

    std::vector<float> got((size_t)NO * VIS_OUT_HIDDEN);
    CU(cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost));
    cmp("merged", got, readbin(R + "merged.bin", got.size()), 0.99900, 5e-2);


    // Position ids and the rope tables the engine will build for itself, against the oracle's.
    // The gate above fed it the oracle's cos/sin on purpose, so these are two separate failures.
    {
        const int GH = 32, GW = 32;
        std::vector<int> pid(2 * S);
        vision_position_ids(1, GH, GW, VIS_MERGE, pid.data());
        auto want_p = readbin(R + "pos_ids.bin", 2 * (size_t)S);
        std::vector<float> gp(2 * S);
        for (size_t i = 0; i < gp.size(); ++i) gp[i] = (float)pid[i];
        cmp("position ids", gp, want_p, 0.9999999, 1e-6);

        std::vector<float> gc((size_t)S * VIS_HEAD_DIM), gs((size_t)S * VIS_HEAD_DIM);
        vision_rope_tables(pid.data(), S, gc.data(), gs.data());
        cmp("rope cos", gc, readbin(R + "cos.bin", gc.size()), 0.9999999, 1e-5);
        cmp("rope sin", gs, readbin(R + "sin.bin", gs.size()), 0.9999999, 1e-5);
    }

    printf("\n--- %d passed, %d failed ---\n", g_pass, g_fail);
    if (g_fail) { printf("GATE FAILED\n"); return 1; }
    printf("ALL VISION GATES PASS\n");
    return 0;
}
