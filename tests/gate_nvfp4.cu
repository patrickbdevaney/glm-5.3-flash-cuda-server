// gate_nvfp4.cu — does the ROOFLINE §3 overlay actually decode to the weights it came from?
//
// tools/requant_dense_nvfp4.py and kernels/gemv.cu independently implement the same format. A
// disagreement between them is silent: the engine still runs, still produces text, and is simply
// wrong. So this reads BOTH the bf16 tensor out of the base checkpoint and the NVFP4 triple out
// of the overlay, runs the same gemv through both paths on the same x, and compares.
//
// The number to expect is set by the format, not by the implementation: NVFP4 is e2m1 with one
// fp8 scale per 16, which round-trips Gaussian weights at rel ~0.095 / cos ~0.9955. A LAYOUT bug
// (nibble order, scale stride, global scale inverted) does not land near that band — it lands at
// cos ~0. The band is therefore the discriminator, and it is deliberately wide.
//
// It also asserts the invariant the batched path depends on: gemv IS gemm at M=1, BIT-for-bit,
// and row m of a width-M gemm is bit-identical to the m-th gemv. Speculative verification
// compares a batched forward against what the sequential path would have produced, and "close"
// is not a comparison.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "safetensors.h"
#include "gemv.h"

using namespace glm5;
#define CU(x) do { cudaError_t e_=(x); if(e_){ fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(e_)); exit(2);} } while(0)

static void* up(const st::Tensor& t) {                      // host mmap -> device
    void* d; CU(cudaMalloc(&d, t.nbytes)); CU(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1]
        : std::string(getenv("HOME")) + "/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2";
    const std::string over = argc > 2 ? argv[2]
        : std::string(getenv("HOME")) + "/glm-5.3-reap/output/glm-5.3-flash-dense-nvfp4-overlay";

    // A gate that passes against an absent file is worse than no gate (CLAUDE.md §2).
    st::ShardedSafeTensors *B = nullptr, *O = nullptr;
    try { B = new st::ShardedSafeTensors(model); O = new st::ShardedSafeTensors(over); }
    catch (const std::exception& e) { fprintf(stderr, "gate_nvfp4: %s\n", e.what()); return 2; }

    // One tensor per family, chosen to cover every shape class the overlay contains: K=4096 with
    // BS 256, K=1536, K=512 with BS 64, K=128 with BS 32 (the short gate projections), and the
    // two widest matrices in the model.
    struct Case { const char* fam; const char* stem; };
    const Case cases[] = {
        {"kda_qkv",   "model.language_model.layers.0.self_attn.q_proj"},
        {"kda_qkv",   "model.language_model.layers.0.self_attn.v_proj"},
        {"o_proj",    "model.language_model.layers.0.self_attn.o_proj"},
        {"o_proj",    "model.language_model.layers.3.self_attn.o_proj"},
        {"kda_gates", "model.language_model.layers.0.self_attn.f_b_proj"},   // K=128
        {"kda_gates", "model.language_model.layers.0.self_attn.b_proj"},     // N=64
        {"mla",       "model.language_model.layers.3.self_attn.q_b_proj"},   // K=1536
        {"mla",       "model.language_model.layers.3.self_attn.kv_a_proj_with_mqa"},
        {"indexer",   "model.language_model.layers.3.self_attn.indexer.wq_b"},
        {"dense_mlp", "model.language_model.layers.0.mlp.gate_proj"},
        {"dense_mlp", "model.language_model.layers.0.mlp.down_proj"},        // K=12288
        {"lm_head",   "lm_head"},
    };

    int pass = 0, total = 0;
    printf("%-10s %-52s %6s %6s %9s %11s %s\n", "family", "tensor", "N", "K", "rel", "cos", "");
    for (const Case& c : cases) {
        const std::string w = std::string(c.stem) + ".weight";
        if (!B->has(w) || !O->has(std::string(c.stem) + ".weight_packed")) {
            printf("%-10s %-52s   MISSING (base=%d overlay=%d)\n", c.fam, c.stem,
                   (int)B->has(w), (int)O->has(std::string(c.stem) + ".weight_packed"));
            total++; continue;
        }
        const st::Tensor& tb = B->get(w);
        const st::Tensor& tp = O->get(std::string(c.stem) + ".weight_packed");
        const st::Tensor& ts = O->get(std::string(c.stem) + ".weight_scale");
        const st::Tensor& tg = O->get(std::string(c.stem) + ".weight_global_scale");
        const int N = (int)tb.shape[0], K = (int)tb.shape[1];
        if (tp.shape[0] != N || tp.shape[1] != K / 2 || ts.shape[1] != K / 16) {
            printf("%-10s %-52s   SHAPE MISMATCH\n", c.fam, c.stem); total++; continue;
        }

        void* dbf = up(tb);
        WRef Wq((const uint8_t*)up(tp), (const uint8_t*)up(ts), (const float*)up(tg));
        WRef Wb(dbf);

        const int M = 4;
        std::vector<float> hx((size_t)M * K);
        unsigned st_ = 12345u;
        for (auto& v : hx) { st_ = st_ * 1664525u + 1013904223u; v = (float)((int)(st_ >> 8) % 2001 - 1000) / 1000.f; }
        float *dx, *y_q, *y_b, *y_m;
        CU(cudaMalloc(&dx, hx.size() * 4)); CU(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        CU(cudaMalloc(&y_q, (size_t)N * 4)); CU(cudaMalloc(&y_b, (size_t)N * 4));
        CU(cudaMalloc(&y_m, (size_t)M * N * 4));

        gemv(y_q, Wq, dx, N, K, GEMV_BF16, 0);
        gemv(y_b, Wb, dx, N, K, GEMV_BF16, 0);
        gemm(y_m, Wq, dx, M, N, K, GEMV_BF16, 0);
        CU(cudaDeviceSynchronize());

        std::vector<float> hq(N), hb(N), hm((size_t)M * N);
        CU(cudaMemcpy(hq.data(), y_q, N * 4, cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(hb.data(), y_b, N * 4, cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(hm.data(), y_m, (size_t)M * N * 4, cudaMemcpyDeviceToHost));

        double num = 0, da = 0, db = 0, dot = 0;
        for (int i = 0; i < N; ++i) { double d = hq[i] - hb[i]; num += d * d;
                                      da += (double)hq[i] * hq[i]; db += (double)hb[i] * hb[i];
                                      dot += (double)hq[i] * hb[i]; }
        const double rel = std::sqrt(num / (db + 1e-30));
        const double cos = dot / (std::sqrt(da * db) + 1e-30);

        // gemv must BE gemm at M=1 — same kernel, same reduction tree, bit for bit.
        int bad_m0 = 0;
        for (int i = 0; i < N; ++i) if (memcmp(&hq[i], &hm[i], 4) != 0) bad_m0++;
        // and every other row of the width-4 gemm must equal its own gemv, bit for bit.
        int bad_mm = 0;
        for (int m = 1; m < M; ++m) {
            gemv(y_q, Wq, dx + (size_t)m * K, N, K, GEMV_BF16, 0);
            CU(cudaDeviceSynchronize());
            std::vector<float> r(N);
            CU(cudaMemcpy(r.data(), y_q, N * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < N; ++i) if (memcmp(&r[i], &hm[(size_t)m * N + i], 4) != 0) bad_mm++;
        }

        const bool ok = rel < 0.15 && cos > 0.988 && bad_m0 == 0 && bad_mm == 0;
        printf("%-10s %-52s %6d %6d %9.5f %11.7f  %s%s%s\n", c.fam, c.stem, N, K, rel, cos,
               ok ? "PASS" : "FAIL",
               bad_m0 ? "  gemv!=gemm(M=1)" : "", bad_mm ? "  gemm rows != gemv" : "");
        pass += ok; total++;

        CU(cudaFree(dbf)); CU(cudaFree((void*)Wq.packed)); CU(cudaFree((void*)Wq.scale));
        CU(cudaFree((void*)Wq.gscale)); CU(cudaFree(dx));
        CU(cudaFree(y_q)); CU(cudaFree(y_b)); CU(cudaFree(y_m));
    }
    printf("gate_nvfp4: %d/%d\n", pass, total);
    return pass == total ? 0 : 1;
}
