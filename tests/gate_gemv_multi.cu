// gate_gemv_multi.cu — is a fused multi-gemv BIT-IDENTICAL to the gemvs it replaces?
//
// kernels/gemv.cu's gemv_multi() folds several independent [N_i, K] gemvs that share K into one
// launch, because the KDA decode path is launch-bound rather than bandwidth-bound (dprof had
// kda:gates at 16.3 GB/s, 7% of achievable, across 238 launches per token). The fusion is supposed
// to be free of numerical consequence: a block still owns the same rows and sums them in the same
// order, so only the launch geometry changes.
//
// "Supposed to be" is not a gate. A y-indexed grid is exactly the kind of change that silently
// perturbs a reduction (a different BS chosen per job, a shared __shared__ buffer reused across
// blockIdx.y, an off-by-one in the row base), and the KDA path feeds speculative verification,
// which compares a batched forward against what the sequential path WOULD have produced. Close is
// not a comparison there. So this asserts memcmp-equality, not cosine.
//
// It runs on the real overlay tensors for the three groups production actually fuses, so a shape
// class that only exists in the model is still covered. A gate that passes against an absent file
// is worse than no gate (CLAUDE.md §2): missing tensors exit non-zero.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "safetensors.h"
#include "gemv.h"

using namespace glm5;
#define CU(x) do { cudaError_t e_=(x); if(e_){ fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(e_)); exit(2);} } while(0)

static void* up(const st::Tensor& t) {
    void* d; CU(cudaMalloc(&d, t.nbytes)); CU(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice));
    return d;
}

struct Group {
    const char* name;
    int         n;
    const char* stem[GEMV_MULTI_MAX];
    bool        shared_x;   // false = each job gets its own x, as f_b/g_b do
};

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1]
        : std::string(getenv("HOME")) + "/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2";
    const std::string over = argc > 2 ? argv[2]
        : std::string(getenv("HOME")) + "/glm-5.3-reap/output/glm-5.3-flash-dense-nvfp4-overlay";

    st::ShardedSafeTensors *B = nullptr, *O = nullptr;
    try { B = new st::ShardedSafeTensors(model); O = new st::ShardedSafeTensors(over); }
    catch (const std::exception& e) { fprintf(stderr, "gate_gemv_multi: %s\n", e.what()); return 2; }

    const char* L0 = "model.language_model.layers.0.self_attn.";
    const Group groups[] = {
        {"kda qkv  (K=4096, shared x)", 3,
         {"q_proj", "k_proj", "v_proj"}, true},
        {"kda gates lo (K=4096, shared x, ragged N)", 3,
         {"f_a_proj", "g_a_proj", "b_proj"}, true},
        {"kda gates hi (K=128, per-job x)", 2,
         {"f_b_proj", "g_b_proj"}, false},
    };

    int pass = 0, total = 0;
    for (const Group& G : groups) {
        total++;
        std::vector<WRef>  W(G.n);
        std::vector<int>   N(G.n);
        std::vector<void*> owned;
        int K = -1;
        bool missing = false;
        for (int i = 0; i < G.n; ++i) {
            const std::string stem = std::string(L0) + G.stem[i];
            const std::string wb = stem + ".weight";
            if (!B->has(wb) || !O->has(stem + ".weight_packed")) {
                printf("  %-42s MISSING %s\n", G.name, stem.c_str()); missing = true; break;
            }
            const st::Tensor& tb = B->get(wb);
            const st::Tensor& tp = O->get(stem + ".weight_packed");
            const st::Tensor& ts = O->get(stem + ".weight_scale");
            const st::Tensor& tg = O->get(stem + ".weight_global_scale");
            N[i] = (int)tb.shape[0];
            const int k = (int)tb.shape[1];
            if (K < 0) K = k;
            if (k != K) { printf("  %-42s K MISMATCH %d vs %d\n", G.name, k, K); missing = true; break; }
            void *p = up(tp), *s = up(ts), *g = up(tg);
            owned.push_back(p); owned.push_back(s); owned.push_back(g);
            W[i] = WRef((const uint8_t*)p, (const uint8_t*)s, (const float*)g);
        }
        if (missing) { for (void* p : owned) CU(cudaFree(p)); continue; }

        // One x per job so the per-job-x path is exercised; shared_x groups point them all at x0.
        const int NX = G.shared_x ? 1 : G.n;
        std::vector<float> hx((size_t)NX * K);
        unsigned st_ = 987654321u;
        for (auto& v : hx) { st_ = st_ * 1664525u + 1013904223u;
                             v = (float)((int)(st_ >> 8) % 2001 - 1000) / 1000.f; }
        float* dx; CU(cudaMalloc(&dx, hx.size() * 4));
        CU(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));

        int tot_rows = 0; for (int i = 0; i < G.n; ++i) tot_rows += N[i];
        float *y_ref, *y_fus;
        CU(cudaMalloc(&y_ref, (size_t)tot_rows * 4));
        CU(cudaMalloc(&y_fus, (size_t)tot_rows * 4));
        CU(cudaMemset(y_ref, 0xA5, (size_t)tot_rows * 4));     // poison: a skipped job must show
        CU(cudaMemset(y_fus, 0x5A, (size_t)tot_rows * 4));

        GemvJob jobs[GEMV_MULTI_MAX];
        int off = 0;
        for (int i = 0; i < G.n; ++i) {
            const float* xi = dx + (size_t)(G.shared_x ? 0 : i) * K;
            gemv(y_ref + off, W[i], xi, N[i], K, GEMV_BF16, 0);       // the path being replaced
            jobs[i] = GemvJob{ y_fus + off, W[i], xi, N[i] };
            off += N[i];
        }
        gemv_multi(jobs, G.n, K, GEMV_BF16, 0);
        CU(cudaDeviceSynchronize());

        std::vector<float> hr(tot_rows), hf(tot_rows);
        CU(cudaMemcpy(hr.data(), y_ref, (size_t)tot_rows * 4, cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(hf.data(), y_fus, (size_t)tot_rows * 4, cudaMemcpyDeviceToHost));

        int bad = 0, nonzero = 0;
        for (int i = 0; i < tot_rows; ++i) {
            if (memcmp(&hr[i], &hf[i], 4) != 0) bad++;
            if (hr[i] != 0.f) nonzero++;
        }
        // A pair of all-zero buffers would memcmp-equal and prove nothing.
        const bool ok = (bad == 0) && (nonzero > tot_rows / 2);
        printf("  %-42s n=%d K=%5d rows=%6d  %s%s\n", G.name, G.n, K, tot_rows,
               ok ? "PASS bit-identical" : "FAIL",
               bad ? "  mismatched lanes" : (nonzero <= tot_rows / 2 ? "  output is degenerate" : ""));
        if (bad) printf("      %d/%d lanes differ\n", bad, tot_rows);
        pass += ok;

        CU(cudaFree(dx)); CU(cudaFree(y_ref)); CU(cudaFree(y_fus));
        for (void* p : owned) CU(cudaFree(p));
    }

    printf("gate_gemv_multi: %d/%d\n", pass, total);
    return pass == total ? 0 : 1;
}
