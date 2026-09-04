// gate_batch.cu — the multi-token forward against the sequential one, BIT-EXACT.
//
// WHY EXACT AND NOT CLOSE. Speculative decoding is only lossless if verifying K drafted tokens in
// one forward produces exactly the logits the autoregressive path would have produced. If the two
// differ even at 1e-6, then "the draft token matches the target's argmax" is being decided against
// a slightly different model, and the output distribution silently stops being the AR one. That is
// unfalsifiable in a benchmark and shows up only as a model that is a little worse than it should
// be. So this compares floats for equality.
//
// It is achievable because gemm is written to reduce in the same order as gemv (one block per
// output row, same shuffle tree), and every non-gemm kernel is called per token with offset
// pointers — the same kernel, on the same data, in the same order.
//
// Needs no PyTorch oracle: the reference IS the already-gated sequential path. That makes it the
// one whole-engine gate that runs while the box is busy.
#include "engine.h"
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

// Exact equality, with the worst offender reported so a failure localises immediately.
static bool same(const std::vector<float>& a, const std::vector<float>& b, const char* tag) {
    if (a.size() != b.size()) { printf("    %s: size %zu vs %zu\n", tag, a.size(), b.size()); return false; }
    size_t ndiff = 0, worst = 0;
    double wmag = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i]) continue;
        ++ndiff;
        const double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > wmag) { wmag = d; worst = i; }
    }
    if (!ndiff) return true;
    printf("    %s: %zu of %zu differ; worst at %zu: %.9g vs %.9g (|d| %.3g)\n",
           tag, ndiff, a.size(), worst, a[worst], b[worst], wmag);
    return false;
}

int main(int argc, char** argv) {
    EngineConfig cfg;
    cfg.model_dir = argc > 1 ? argv[1]
        : std::string(getenv("HOME")) + "/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2";
    cfg.n_layer   = argc > 2 ? atoi(argv[2]) : 4;      // 4 reaches layer 3: the first MLA + MoE layer
    cfg.max_ctx   = 256;
    cfg.max_batch = 8;
    cfg.verbose   = true;

    printf("=== gate_batch (%d layers) ===\n", cfg.n_layer);
    Engine eng(cfg);
    printf("layer types in range: ");
    for (int i = 0; i < cfg.n_layer; ++i)
        printf("%d=%s/%s ", i, is_kda(i) ? "kda" : "mla", is_moe(i) ? "moe" : "dense");
    printf("\n");

    // Real token ids, not random ints: routing and the recurrence both depend on the embedding
    // actually being on the manifold the model was trained on, and a random id lands off it.
    const std::vector<int> toks = { 3639, 374, 279, 6864, 315, 9822, 30, 358, 1781, 433, 596, 12366 };
    const int T = (int)toks.size();
    const size_t V = VOCAB;

    float* dlog = nullptr;
    CU(cudaMalloc(&dlog, V * cfg.max_batch * 4));

    // ---- reference: one token at a time ----
    std::vector<float> ref(V * T);
    eng.reset(0);
    for (int t = 0; t < T; ++t) {
        eng.decode(toks[t], t, dlog, 0);
        CU(cudaDeviceSynchronize());
        CU(cudaMemcpy(ref.data() + (size_t)t * V, dlog, V * 4, cudaMemcpyDeviceToHost));
    }
    std::vector<float> ref_state((size_t)KDA_STATE_PER_LAYER);
    printf("reference: %d sequential decodes done\n", T);

    // ---- batched, at every width the verify path could use ----
    for (int K : { 1, 2, 3, 4, 5, 8 }) {
        std::vector<float> got(V * T);
        eng.reset(0);
        for (int t = 0; t < T; t += K) {
            const int m = std::min(K, T - t);
            eng.forward_batch(toks.data() + t, m, t, dlog, /*all_logits=*/true, 0);
            CU(cudaDeviceSynchronize());
            CU(cudaMemcpy(got.data() + (size_t)t * V, dlog, (size_t)m * V * 4, cudaMemcpyDeviceToHost));
        }
        char tag[64]; snprintf(tag, sizeof tag, "K=%d", K);
        ck(same(ref, got, tag), tag);
    }

    // A ragged split — the shape a speculative verify actually produces once some drafts are
    // rejected and the next iteration starts mid-stream with a different width.
    {
        std::vector<float> got(V * T);
        const int widths[] = { 3, 1, 4, 2, 2 };
        int t = 0;
        eng.reset(0);
        for (int w : widths) {
            const int m = std::min(w, T - t);
            if (m <= 0) break;
            eng.forward_batch(toks.data() + t, m, t, dlog, true, 0);
            CU(cudaDeviceSynchronize());
            CU(cudaMemcpy(got.data() + (size_t)t * V, dlog, (size_t)m * V * 4, cudaMemcpyDeviceToHost));
            t += m;
        }
        ck(same(ref, got, "ragged 3,1,4,2,2"), "ragged widths");
    }

    // all_logits=false must produce the SAME last-token logits, having skipped lm_head for the
    // rest. This is the prefill path; if it disagreed, prefill and decode would be different models.
    {
        eng.reset(0);
        eng.forward_batch(toks.data(), 8, 0, dlog, /*all_logits=*/false, 0);
        CU(cudaDeviceSynchronize());
        std::vector<float> last(V);
        CU(cudaMemcpy(last.data(), dlog, V * 4, cudaMemcpyDeviceToHost));
        std::vector<float> want(ref.begin() + (size_t)7 * V, ref.begin() + (size_t)8 * V);
        ck(same(want, last, "all_logits=false"), "last-token-only logits match");
    }

    // Interleaving the two paths must also agree: prefill a chunk, then decode single tokens on
    // top of the state it left. This is exactly what generate() does.
    {
        std::vector<float> got(V * T);
        eng.reset(0);
        eng.forward_batch(toks.data(), 5, 0, dlog, true, 0);
        CU(cudaDeviceSynchronize());
        CU(cudaMemcpy(got.data(), dlog, 5 * V * 4, cudaMemcpyDeviceToHost));
        for (int t = 5; t < T; ++t) {
            eng.decode(toks[t], t, dlog, 0);
            CU(cudaDeviceSynchronize());
            CU(cudaMemcpy(got.data() + (size_t)t * V, dlog, V * 4, cudaMemcpyDeviceToHost));
        }
        ck(same(ref, got, "batch-then-sequential"), "a batched prefill leaves a usable state");
    }

    cudaFree(dlog);
    printf("--- %d passed, %d failed ---\n", pass, fail);
    printf(fail ? "GATE FAILED\n" : "ALL BATCH GATES PASS\n");
    return fail ? 1 : 0;
}
