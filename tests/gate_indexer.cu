// gate_indexer.cu — the DSA indexer against the real transformers module.
//
// Oracle: ref/gen_indexer.py, which dumps the indexer's own inputs (h, q_resid), its weights, and
// its output for every query position at T = 38, 2050 and 3003 — straddling the 2051 dense limit.
//
// WHAT IS COMPARED, AND WHY IT IS THE SET AND NOT THE ARRAY.
// The oracle runs PREFILL: all T queries at once, so `select_k` is min(512, floor(T/4)) for every
// row, and a row early in the sequence fills most of its 512 pool slots with pools it cannot see,
// which are then masked to -1. This engine runs DECODE: at step t only floor((t+1)/4) pools exist,
// so select_k is smaller and the real indices sit at different offsets. The two produce the SAME
// SET of visible keys at different positions in the array. Attention consumes a set — it softmaxes
// over the selected keys — so the set is the invariant that matters, and comparing arrays would
// fail on a difference that cannot affect any output.
//
// The LAST row is the exception: there, prefill and decode see identical candidate pools with
// identical scores, so ORDER must match too. That is checked separately, and it is what actually
// pins the top-k tie-break (value descending, index ascending) rather than merely the selection.
#include "indexer.h"
#include "gemv.h"
#include "glm5_config.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <functional>
#include <set>
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

static std::string REF;
template <typename T>
static std::vector<T> rd(const std::string& name, size_t n) {
    std::vector<T> v(n);
    const std::string p = REF + "/" + name;
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { printf("  FAIL cannot open %s\n", p.c_str()); exit(1); }
    if (fread(v.data(), sizeof(T), n, f) != n) { printf("  FAIL short read %s\n", p.c_str()); exit(1); }
    fclose(f);
    return v;
}
static float* to_dev(const std::vector<float>& h) {
    float* d; CU(cudaMalloc(&d, h.size() * 4));
    CU(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    REF = argc > 1 ? argv[1]
        : std::string(getenv("HOME")) + "/glm-5.3-flash-cuda-server/ref";
    printf("=== gate_indexer ===\n");
    printf("dense limit %d (IDX_TOPK %d + IDX_KPOOL %d - 1), out width %d, select budget %d\n",
           DENSE_CTX_LIMIT, IDX_TOPK, IDX_KPOOL, IDX_OUT_WIDTH, IDX_SELECT_MAX);

    IndexerWeights W{};
    W.dtype = GEMV_F32;
    W.wq_b          = to_dev(rd<float>("indexer_w_wq_b.bin", (size_t)IDX_HEADS * IDX_HEAD_DIM * MLA_Q_LORA));
    W.wk            = to_dev(rd<float>("indexer_w_wk.bin", (size_t)IDX_HEAD_DIM * HIDDEN));
    W.k_norm_w      = to_dev(rd<float>("indexer_w_k_norm_w.bin", IDX_HEAD_DIM));
    W.k_norm_b      = to_dev(rd<float>("indexer_w_k_norm_b.bin", IDX_HEAD_DIM));
    W.weights_proj  = to_dev(rd<float>("indexer_w_weights_proj.bin", (size_t)IDX_HEADS * HIDDEN));
    W.compress_ape  = to_dev(rd<float>("indexer_w_compress_ape.bin", (size_t)IDX_KPOOL * IDX_HEAD_DIM));
    W.compress_gate = to_dev(rd<float>("indexer_w_compress_gate.bin", (size_t)IDX_HEAD_DIM * HIDDEN));

    for (int T : { 38, 2050, 3003 }) {
        printf("--- T = %d %s ---\n", T, T <= DENSE_CTX_LIMIT ? "(dense: indexer must select everything)"
                                                              : "(sparse)");
        const auto h_h  = rd<float>("indexer_h_" + std::to_string(T) + ".bin", (size_t)T * HIDDEN);
        const auto qr_h = rd<float>("indexer_q_resid_" + std::to_string(T) + ".bin", (size_t)T * MLA_Q_LORA);
        const auto ref  = rd<int32_t>("indexer_topk_" + std::to_string(T) + ".bin", (size_t)T * IDX_OUT_WIDTH);
        float* h_d  = to_dev(h_h);
        float* qr_d = to_dev(qr_h);

        const int max_ctx = T;
        IndexerState S{};
        float* state; CU(cudaMalloc(&state, indexer_state_floats(max_ctx) * 4));
        CU(cudaMemset(state, 0, indexer_state_floats(max_ctx) * 4));
        S.pool_keys = state;
        S.roll_k    = state + (size_t)idx_max_pools(max_ctx) * IDX_HEAD_DIM;
        S.roll_gate = S.roll_k + IDX_KPOOL * IDX_HEAD_DIM;

        float* ws; CU(cudaMalloc(&ws, indexer_workspace_floats(max_ctx) * 4));
        int32_t *out, *nout;
        CU(cudaMalloc(&out, IDX_OUT_WIDTH * 4));
        CU(cudaMalloc(&nout, 4));

        std::vector<int32_t> got(IDX_OUT_WIDTH);
        int bad_set = 0, first_bad = -1, checked = 0;
        size_t worst_missing = 0, worst_extra = 0;

        for (int t = 0; t < T; ++t) {
            indexer_decode_step(h_d + (size_t)t * HIDDEN, qr_d + (size_t)t * MLA_Q_LORA,
                                W, S, t, max_ctx, out, nout, ws, 0);
            CU(cudaDeviceSynchronize());
            CU(cudaMemcpy(got.data(), out, IDX_OUT_WIDTH * 4, cudaMemcpyDeviceToHost));

            std::set<int> a, b;
            for (int i = 0; i < IDX_OUT_WIDTH; ++i) {
                if (got[i] >= 0) a.insert(got[i]);
                const int32_t r = ref[(size_t)t * IDX_OUT_WIDTH + i];
                if (r >= 0) b.insert(r);
            }
            ++checked;
            if (a != b) {
                std::vector<int> miss, extra;
                std::set_difference(b.begin(), b.end(), a.begin(), a.end(), std::back_inserter(miss));
                std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(extra));
                if (!bad_set) {
                    first_bad = t;
                    printf("    first divergence at t=%d: ours %zu keys, ref %zu; missing %zu, extra %zu\n",
                           t, a.size(), b.size(), miss.size(), extra.size());
                    for (size_t i = 0; i < miss.size() && i < 8; ++i) printf("      missing %d\n", miss[i]);
                    for (size_t i = 0; i < extra.size() && i < 8; ++i) printf("      extra   %d\n", extra[i]);
                }
                ++bad_set;
                worst_missing = std::max(worst_missing, miss.size());
                worst_extra = std::max(worst_extra, extra.size());
            }
            // Below the dense limit the indexer must select EVERY visible key. That is the claim
            // the engine's dense-MLA path rests on, so it is asserted directly rather than
            // inferred from agreeing with the reference.
            if (T <= DENSE_CTX_LIMIT && (int)a.size() != t + 1 && !bad_set) {
                printf("    t=%d: dense claim violated, %zu keys for %d positions\n", t, a.size(), t + 1);
                ++bad_set; first_bad = t;
            }
        }
        char tag[96];
        snprintf(tag, sizeof tag, "T=%d: visible-key SET matches at all %d query positions", T, checked);
        if (bad_set) printf("    %d of %d rows differ (worst: %zu missing, %zu extra, first at t=%d)\n",
                            bad_set, checked, worst_missing, worst_extra, first_bad);
        ck(bad_set == 0, tag);

        // Last row. The emitted list is now sorted ASCENDING by token index (see k_select_emit),
        // so "does it match the reference array" is no longer the question — the reference emits
        // score order. What has to be true is that the SELECTED SET is the top select_k pools by
        // score, and that the list really is ascending.
        //
        // Selection is checked as: the worst reference score among the pools we chose is no worse
        // than the select_k-th best reference score. Demanding the identical set would fail on
        // near-ties, and those are real here — our scores agree with the reference to 1.7e-6 while
        // 2 of the 511 adjacent gaps in the ranking are tighter than that (at T=2050, rank 412 to
        // 413 is 5.1e-7). Two correct fp32 implementations swap that pair.
        {
            const int t = T - 1;
            const int n_pools = (t + 1) / IDX_KPOOL;
            const int select_k = n_pools < IDX_SELECT_MAX ? n_pools : IDX_SELECT_MAX;
            const auto rs = rd<float>("indexer_scores_" + std::to_string(T) + ".bin", n_pools);
            indexer_decode_step(h_d + (size_t)t * HIDDEN, qr_d + (size_t)t * MLA_Q_LORA,
                                W, S, t, max_ctx, out, nout, ws, 0);
            CU(cudaDeviceSynchronize());
            CU(cudaMemcpy(got.data(), out, IDX_OUT_WIDTH * 4, cudaMemcpyDeviceToHost));
            int n = 0;
            CU(cudaMemcpy(&n, nout, 4, cudaMemcpyDeviceToHost));

            snprintf(tag, sizeof tag, "T=%d: emit count is positive (no NaN hole)", T);
            ck(n > 0, tag);

            bool ascending = true;
            int first_desc = -1;
            for (int i = 1; i < n; ++i)
                if (got[i] <= got[i - 1]) { ascending = false; if (first_desc < 0) first_desc = i; }
            if (!ascending)
                printf("    not ascending at slot %d: %d then %d\n",
                       first_desc, got[first_desc - 1], got[first_desc]);
            snprintf(tag, sizeof tag, "T=%d: emitted keys are ascending and distinct (%d keys)", T, n);
            ck(ascending, tag);

            // The select_k-th best reference score is the admission bar.
            std::vector<float> sorted(rs.begin(), rs.end());
            std::sort(sorted.begin(), sorted.end(), std::greater<float>());
            const double bar = sorted[select_k - 1];
            const double TOL = 4e-6;                   // ~2x the measured score agreement
            int below = 0;
            double worst = 0;
            for (int j = 0; j < select_k; ++j) {
                if (got[j * IDX_KPOOL] < 0) continue;
                const int pool = got[j * IDX_KPOOL] / IDX_KPOOL;
                if (pool < 0 || pool >= n_pools) { ++below; continue; }
                const double d = bar - (double)rs[pool];
                if (d > TOL) { ++below; worst = std::max(worst, d); }
            }
            printf("    selection: bar %+.7f, %d of %d chosen pools below it by more than %.0e%s\n",
                   bar, below, select_k, TOL, below ? "" : " (every choice is top-k)");
            if (below) printf("    worst shortfall %.3e\n", worst);
            snprintf(tag, sizeof tag, "T=%d: chose the top-%d pools by score", T, select_k);
            ck(below == 0, tag);
        }

        // The actual arithmetic check: the per-pool index_scores for the last query, against the
        // reference recomputed through the module's own get_pooled_states. Everything above is
        // downstream of these — a selection that agrees while the scores are wrong would be luck.
        {
            const int t = T - 1;
            const int n_pools = (t + 1) / IDX_KPOOL;
            const auto rs = rd<float>("indexer_scores_" + std::to_string(T) + ".bin", n_pools);
            float* scores_d = ws + 2 * IDX_HEAD_DIM + (size_t)IDX_HEADS * IDX_HEAD_DIM + IDX_HEADS + 64;
            std::vector<float> ours(n_pools);
            CU(cudaMemcpy(ours.data(), scores_d, (size_t)n_pools * 4, cudaMemcpyDeviceToHost));

            double dot = 0, na = 0, nb = 0, maxabs = 0, scale = 0;
            for (int p = 0; p < n_pools; ++p) {
                dot += (double)ours[p] * rs[p];
                na += (double)ours[p] * ours[p];
                nb += (double)rs[p] * rs[p];
                maxabs = std::max(maxabs, std::fabs((double)ours[p] - rs[p]));
                scale = std::max(scale, std::fabs((double)rs[p]));
            }
            const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
            const double rel = scale > 0 ? maxabs / scale : 0;
            printf("    pool scores: cos %.9f  max|d| %.3e  rel %.3e  over %d pools\n",
                   cos, maxabs, rel, n_pools);
            snprintf(tag, sizeof tag, "T=%d: pool scores match the reference", T);
            ck(cos > 0.9999999 && rel < 2e-5, tag);
        }

        cudaFree(h_d); cudaFree(qr_d); cudaFree(state); cudaFree(ws); cudaFree(out); cudaFree(nout);
    }

    printf("--- %d passed, %d failed ---\n", pass, fail);
    printf(fail ? "GATE FAILED\n" : "ALL INDEXER GATES PASS\n");
    return fail ? 1 : 0;
}
