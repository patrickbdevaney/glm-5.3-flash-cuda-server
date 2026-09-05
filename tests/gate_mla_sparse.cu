// gate_mla_sparse.cu — the DSA sparse attention path against the dense one, BIT-EXACT.
//
// Above DENSE_CTX_LIMIT there is no dense answer to compare against — that is the whole point of
// the indexer. So the check is the equivalence that must hold BELOW it: while the indexer selects
// every visible key, sparse attention must reproduce dense attention exactly. Two things make that
// an exact test rather than an approximate one:
//   * the indexer really does select everything at or below 2051 (gate_indexer proves it), and
//   * k_select_emit emits ASCENDING, so the fp32 context sum accumulates in the same order.
// If either stopped being true this would fail, which is the point.
//
// It also runs the PRODUCTION bf16 arm on real checkpoint weights, so it covers the indexer's bf16
// dtype dispatch — the fp32 arm in gate_indexer does not.
//
// Above the limit the two paths are SUPPOSED to differ. That is asserted too: a sparse path that
// silently kept agreeing with dense at 3000 tokens would mean the indexer was not dropping
// anything, i.e. that it was not working.
#include "mla.h"
#include "indexer.h"
#include "gemv.h"
#include "weight_store.h"
#include "glm5_config.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

using namespace glm5;
#define CU(x) do { cudaError_t e_=(x); if(e_){ printf("cuda %s:%d %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(e_)); exit(1);} } while(0)

static int pass = 0, fail = 0;
static void ck(bool ok, const char* what) {
    if (ok) { ++pass; printf("  ok   %s\n", what); }
    else    { ++fail; printf("  FAIL %s\n", what); }
}

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1]
        : std::string(getenv("HOME")) + "/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2";
    const std::string ref = argc > 2 ? argv[2]
        : std::string(getenv("HOME")) + "/glm-5.3-flash-cuda-server/ref";
    const int L = 3, T = 3003;

    printf("=== gate_mla_sparse (layer %d, %d steps, dense limit %d) ===\n", L, T, DENSE_CTX_LIMIT);
    char onlyp[64]; snprintf(onlyp, sizeof onlyp, "layers.%d.self_attn.", L);
    st::WeightStore WS(model, nullptr, onlyp);
    char P[128]; snprintf(P, sizeof P, "model.language_model.layers.%d.self_attn.", L);
    auto N = [&](const char* s) { return std::string(P) + s; };
    printf("loaded %zu tensors, %.2f GiB\n", WS.count(), WS.loadedGiB());

    MlaWeights W{};
    W.dtype     = GEMV_BF16;
    W.q_a       = WS.get(N("q_a_proj.weight")).dev;
    W.q_a_norm  = WS.get(N("q_a_layernorm.weight")).dev;
    W.q_b       = WS.get(N("q_b_proj.weight")).dev;
    W.kv_a      = WS.get(N("kv_a_proj_with_mqa.weight")).dev;
    W.kv_a_norm = WS.get(N("kv_a_layernorm.weight")).dev;
    W.kv_b      = WS.get(N("kv_b_proj.weight")).dev;
    W.o_proj    = WS.get(N("o_proj.weight")).dev;

    IndexerWeights IW{};
    IW.dtype         = GEMV_BF16;
    IW.wq_b          = WS.get(N("indexer.wq_b.weight")).dev;
    IW.wk            = WS.get(N("indexer.wk.weight")).dev;
    IW.k_norm_w      = WS.get(N("indexer.k_norm.weight")).dev;
    IW.k_norm_b      = WS.get(N("indexer.k_norm.bias")).dev;
    IW.weights_proj  = WS.get(N("indexer.weights_proj.weight")).dev;
    IW.compress_ape  = WS.get(N("indexer.index_kpool_compress_ape")).dev;
    IW.compress_gate = WS.get(N("indexer.index_kpool_compress_gate")).dev;

    // Real hidden states, from the indexer oracle's dump.
    std::vector<float> x((size_t)T * HIDDEN);
    {
        const std::string p = ref + "/indexer_h_3003.bin";
        FILE* f = fopen(p.c_str(), "rb");
        if (!f) { printf("  FAIL cannot open %s\n", p.c_str()); return 1; }
        if (fread(x.data(), 4, x.size(), f) != x.size()) { printf("  FAIL short read\n"); return 1; }
        fclose(f);
    }
    float* xd; CU(cudaMalloc(&xd, x.size() * 4));
    CU(cudaMemcpy(xd, x.data(), x.size() * 4, cudaMemcpyHostToDevice));

    const int max_ctx = T;
    float *cache_a, *cache_b, *ws_a, *ws_b, *iws, *y_a, *y_b, *istate;
    CU(cudaMalloc(&cache_a, (size_t)max_ctx * MLA_KV_LORA * 4));
    CU(cudaMalloc(&cache_b, (size_t)max_ctx * MLA_KV_LORA * 4));
    CU(cudaMalloc(&ws_a, mla_workspace_floats(max_ctx) * 4));
    CU(cudaMalloc(&ws_b, mla_workspace_floats(max_ctx) * 4));
    CU(cudaMalloc(&iws, indexer_workspace_floats(max_ctx) * 4));
    CU(cudaMalloc(&istate, indexer_state_floats(max_ctx) * 4));
    CU(cudaMemset(istate, 0, indexer_state_floats(max_ctx) * 4));
    CU(cudaMalloc(&y_a, HIDDEN * 4));
    CU(cudaMalloc(&y_b, HIDDEN * 4));
    int32_t *sel, *nsel;
    CU(cudaMalloc(&sel, IDX_OUT_WIDTH * 4));
    CU(cudaMalloc(&nsel, 4));

    IndexerState IS{};
    IS.pool_keys = istate;
    IS.roll_k    = istate + (size_t)idx_max_pools(max_ctx) * IDX_HEAD_DIM;
    IS.roll_gate = IS.roll_k + IDX_KPOOL * IDX_HEAD_DIM;

    std::vector<float> ya(HIDDEN), yb(HIDDEN);
    int mismatch_below = 0, first_bad = -1, identical_above = 0, checked_below = 0, checked_above = 0;
    int bad_count = 0;

    for (int t = 0; t < T; ++t) {
        mla_decode_step(xd + (size_t)t * HIDDEN, W, cache_a, t, max_ctx, y_a, ws_a, 0);
        // force_sparse: below the limit mla_decode_step_dsa would otherwise take the DENSE
        // branch, and this gate would be comparing dense against dense and passing vacuously.
        mla_decode_step_dsa(xd + (size_t)t * HIDDEN, W, IW, IS, cache_b, t, max_ctx,
                            sel, nsel, y_b, ws_b, iws, 0, /*force_sparse=*/true);
        CU(cudaDeviceSynchronize());
        CU(cudaMemcpy(ya.data(), y_a, HIDDEN * 4, cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(yb.data(), y_b, HIDDEN * 4, cudaMemcpyDeviceToHost));

        int n = 0;
        CU(cudaMemcpy(&n, nsel, 4, cudaMemcpyDeviceToHost));
        const int expect = (t + 1 <= DENSE_CTX_LIMIT) ? t + 1
                         : IDX_SELECT_MAX * IDX_KPOOL + (t + 1 - ((t + 1) / IDX_KPOOL) * IDX_KPOOL);
        if (n != expect && bad_count < 4) {
            printf("    t=%d: selected %d keys, expected %d\n", t, n, expect);
            ++bad_count;
        }

        bool same = true;
        for (int i = 0; i < HIDDEN; ++i) if (ya[i] != yb[i]) { same = false; break; }
        if (t + 1 <= DENSE_CTX_LIMIT) {
            ++checked_below;
            if (!same) { ++mismatch_below; if (first_bad < 0) first_bad = t; }
        } else {
            ++checked_above;
            if (same) ++identical_above;
        }
    }

    if (mismatch_below)
        printf("    %d of %d steps below the limit differ, first at t=%d\n",
               mismatch_below, checked_below, first_bad);
    char tag[128];
    snprintf(tag, sizeof tag, "sparse == dense BIT-EXACT for all %d steps at or below %d",
             checked_below, DENSE_CTX_LIMIT);
    ck(mismatch_below == 0, tag);

    snprintf(tag, sizeof tag, "key count is exactly right at every one of %d steps", T);
    ck(bad_count == 0, tag);

    // Above the limit the indexer must actually be dropping keys, so the two paths must diverge.
    // A sparse path that still agreed would mean the indexer was a no-op.
    printf("    above the limit: %d of %d steps still identical to dense\n",
           identical_above, checked_above);
    snprintf(tag, sizeof tag, "sparse DIVERGES from dense above %d (indexer is doing work)",
             DENSE_CTX_LIMIT);
    ck(checked_above > 0 && identical_above == 0, tag);

    printf("--- %d passed, %d failed ---\n", pass, fail);
    printf(fail ? "GATE FAILED\n" : "ALL SPARSE-MLA GATES PASS\n");
    return fail ? 1 : 0;
}
