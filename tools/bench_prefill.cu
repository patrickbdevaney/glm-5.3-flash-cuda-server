// bench_prefill.cu — what does a prompt token actually cost, and why?
//
// Prefill was the server's single largest cost per request and had never been profiled. The one
// long-context attempt on the full model — a ~3,400-token retrieval prompt — ran past 34 minutes
// at 97% GPU without returning, i.e. slower PER TOKEN than decode. That is backwards for a
// batched path and it is the anomaly this tool exists to attribute.
//
// It sweeps the chunk width round-robin rather than in blocks (CLAUDE.md §6.2) and prices every
// phase against a streaming read measured in THIS process, immediately before the loop.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "engine.h"
#include "dprof.h"

using namespace glm5;
#define CU(x) do { cudaError_t e_=(x); if(e_){ fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(e_)); exit(2);} } while(0)

__global__ void k_stream(const float4* __restrict__ p, float* out, size_t n4) {
    float4 a = make_float4(0, 0, 0, 0);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        float4 v = p[i]; a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
    }
    if (a.x == 12345.678f) out[0] = a.x + a.y + a.z + a.w;
}

static double measure_stream_bw() {
    const size_t bytes = 512ull << 20;
    float4* buf; float* sink;
    CU(cudaMalloc(&buf, bytes)); CU(cudaMalloc(&sink, 4)); CU(cudaMemset(buf, 1, bytes));
    const size_t n4 = bytes / sizeof(float4);
    k_stream<<<1024, 256>>>(buf, sink, n4);
    CU(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < 8; ++i) k_stream<<<1024, 256>>>(buf, sink, n4);
    cudaEventRecord(b);
    CU(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaFree(buf); cudaFree(sink);
    return 8.0 * bytes / (ms / 1000.0) / 1e9;
}

int main(int argc, char** argv) {
    EngineConfig ec;
    ec.model_dir = std::string(getenv("HOME")) + "/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2";
    ec.n_layer = N_LAYER;
    int npr = 512, reps = 2;
    std::vector<int> widths;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]{ return std::string(argv[++i]); };
        if      (a == "--ckpt")    ec.model_dir = next();
        else if (a == "--n-layer") ec.n_layer   = atoi(next().c_str());
        else if (a == "--prompt")  npr          = atoi(next().c_str());
        else if (a == "--reps")    reps         = atoi(next().c_str());
        else if (a == "--widths") {
            std::string w = next(), cur;
            for (char c : w) { if (c == ',') { widths.push_back(atoi(cur.c_str())); cur.clear(); }
                               else cur.push_back(c); }
            if (!cur.empty()) widths.push_back(atoi(cur.c_str()));
        }
    }
    if (widths.empty()) widths = {1, 4, 16};
    ec.max_batch = *std::max_element(widths.begin(), widths.end());
    // Every rep re-prefills from scratch, so the cache has to hold reps*widths.size() prompts.
    ec.max_ctx = npr * (int)widths.size() * reps + 64;

    Engine eng(ec);
    std::vector<int> prompt(npr);
    unsigned st = 991u;
    for (int i = 0; i < npr; ++i) { st = st * 1664525u + 1013904223u; prompt[i] = (int)(st % 100000u) + 100; }

    // One untimed prefill: the first one pays page faults and module load, and folding those in
    // would blame whichever width happened to run first.
    eng.prefill(std::vector<int>(prompt.begin(), prompt.begin() + std::min(npr, 64)));

    const double bw = measure_stream_bw();
    printf("streaming read RIGHT NOW: %.1f GB/s\n", bw);
    printf("prompt %d tokens, %d layers, widths:", npr, ec.n_layer);
    for (int w : widths) printf(" %d", w);
    printf(", %d reps, ROUND-ROBIN\n\n", reps);

    std::vector<std::vector<double>> ms(widths.size());
    for (int r = 0; r < reps; ++r)
        for (size_t wi = 0; wi < widths.size(); ++wi) {
            eng.setChunk(widths[wi]);
            CU(cudaDeviceSynchronize());
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            cudaEventRecord(a);
            eng.prefill(prompt);
            cudaEventRecord(b); CU(cudaEventSynchronize(b));
            float t = 0; cudaEventElapsedTime(&t, a, b);
            ms[wi].push_back(t);
            printf("  rep %d  width %-3d  %9.1f ms  %7.3f ms/tok  %7.1f tok/s\n",
                   r, widths[wi], t, t / npr, 1000.0 * npr / t);
            fflush(stdout);
        }

    printf("\n%-8s %12s %12s %12s\n", "width", "min ms/tok", "med ms/tok", "tok/s (min)");
    for (size_t wi = 0; wi < widths.size(); ++wi) {
        std::vector<double> v = ms[wi];
        std::sort(v.begin(), v.end());
        const double mn = v.front() / npr, md = v[v.size() / 2] / npr;
        printf("%-8d %12.3f %12.3f %12.1f\n", widths[wi], mn, md, 1000.0 / mn);
    }

    // The dprof table covers EVERY prefill above, so it attributes the mix rather than one width.
    // Per-token columns are meaningless across mixed widths; the ms and % columns are the point.
    dprof_report("prefill (all widths, all reps)", npr * (int)widths.size() * reps, bw);
    return 0;
}
