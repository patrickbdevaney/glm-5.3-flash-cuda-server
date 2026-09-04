// bench_batch.cu — the batch-cost curve, measured.
//
// ROOFLINE.md §4 predicts it analytically and flags one assumption it cannot prove: that a gemm at
// M>1 holds the same DRAM bandwidth as the gemv at M=1. If it does not, every speculation number
// in that table is optimistic. This measures it.
//
// The box is shared. OPTIMIZATION_LOG #1 records a kernel being called "3x off" when it was
// actually at the machine's ceiling, because it was measured against an idle-box figure while
// something else held the GPU. So this runs a pure streaming-read probe IN THIS PROCESS,
// immediately before the timing loop, and reports everything against that.
#include "engine.h"
#include "glm5_config.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <vector>

using namespace glm5;
#define CU(x) do { cudaError_t e_=(x); if(e_){ printf("cuda %s:%d %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(e_)); exit(1);} } while(0)

__global__ void k_stream(const float4* __restrict__ src, float* __restrict__ sink, size_t n4) {
    float acc = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        const float4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 1234.5f) sink[0] = acc;                  // never true; defeats dead-code removal
}

// GB/s a pure sequential read achieves RIGHT NOW, on this box, under whatever else is running.
static double measure_stream_bw() {
    const size_t bytes = 2ull << 30;                    // 2 GiB, far past any cache
    float4* buf; float* sink;
    CU(cudaMalloc(&buf, bytes));
    CU(cudaMalloc(&sink, 4));
    CU(cudaMemset(buf, 1, bytes));
    const size_t n4 = bytes / sizeof(float4);
    k_stream<<<1024, 256>>>(buf, sink, n4);             // warm
    CU(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < 4; ++i) k_stream<<<1024, 256>>>(buf, sink, n4);
    cudaEventRecord(b);
    CU(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaFree(buf); cudaFree(sink);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return 4.0 * bytes / (ms / 1000.0) / 1e9;
}

int main(int argc, char** argv) {
    EngineConfig cfg;
    cfg.model_dir = argc > 1 ? argv[1]
        : std::string(getenv("HOME")) + "/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2";
    cfg.n_layer   = argc > 2 ? atoi(argv[2]) : 4;
    cfg.max_ctx   = 512;
    cfg.max_batch = 16;
    cfg.verbose   = true;

    printf("=== bench_batch (%d layers) ===\n", cfg.n_layer);
    Engine eng(cfg);

    const double bw = measure_stream_bw();
    printf("streaming read RIGHT NOW: %.1f GB/s  (idle-box reference is 240; anything well below\n"
           "  that means the box is contended and the absolute numbers here are depressed --\n"
           "  the RATIOS between widths are what this bench is for)\n\n", bw);

    float* dlog; CU(cudaMalloc(&dlog, (size_t)cfg.max_batch * VOCAB * 4));
    std::vector<int> toks(64);
    for (size_t i = 0; i < toks.size(); ++i) toks[i] = 1000 + (int)i * 37;

    // METHODOLOGY, and it is not optional on a shared box. A first attempt timed each width in one
    // contiguous block of 8 reps and produced K=2 measuring FASTER than K=1 in absolute ms/forward
    // — arithmetically impossible, and a clear sign that contention spikes were dominating. So:
    // every rep is timed individually, the widths are visited ROUND-ROBIN so a spike lands on all
    // of them alike, and the reported figure is the MINIMUM over reps. The minimum is the right
    // estimator here: the true cost is a floor set by bandwidth, and every disturbance can only
    // push a sample above it. The median is printed alongside so the spread is visible — if min
    // and median are far apart, the box was busy and the run should be repeated.
    const int WID[] = { 1, 2, 3, 4, 5, 6, 8, 12, 16 };
    const int NW = (int)(sizeof(WID) / sizeof(WID[0]));
    const int reps = 32;
    std::vector<std::vector<double>> t(NW);

    for (int w = 0; w < NW; ++w) {                      // warm every width first
        if (WID[w] > cfg.max_batch) continue;
        eng.reset(0);
        eng.forward_batch(toks.data(), WID[w], 0, dlog, true, 0);
    }
    CU(cudaDeviceSynchronize());

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int r = 0; r < reps; ++r) {
        for (int w = 0; w < NW; ++w) {
            const int K = WID[w];
            if (K > cfg.max_batch) continue;
            eng.reset(0);
            CU(cudaDeviceSynchronize());
            cudaEventRecord(a);
            eng.forward_batch(toks.data(), K, 0, dlog, true, 0);
            cudaEventRecord(b);
            CU(cudaEventSynchronize(b));
            float ms = 0; cudaEventElapsedTime(&ms, a, b);
            t[w].push_back(ms);
        }
    }
    cudaEventDestroy(a); cudaEventDestroy(b);

    printf("%4s %10s %10s %11s %10s %9s\n", "K", "min ms", "med ms", "ms/token", "vs K=1", "speedup");
    double base = 0;
    for (int w = 0; w < NW; ++w) {
        if (t[w].empty()) continue;
        std::sort(t[w].begin(), t[w].end());
        const double mn = t[w].front(), md = t[w][t[w].size() / 2];
        const int K = WID[w];
        if (K == 1) base = mn;
        printf("%4d %10.2f %10.2f %11.3f %10.3f %8.2fx\n", K, mn, md, mn / K, mn / base,
               base / (mn / K));
    }

    printf("\n'vs K=1' is what ROOFLINE.md §4 predicts as the cost column; 'speedup' is the\n"
           "ceiling a verify of that width could reach if every drafted token were accepted.\n"
           "If min and med differ by more than ~15%%, the box was contended and this run is not a\n"
           "measurement — repeat it when the GPU is idle.\n");
    cudaFree(dlog);
    return 0;
}
