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

    printf("%4s %10s %11s %10s %9s\n", "K", "ms/fwd", "ms/token", "vs K=1", "speedup");
    double base = 0;
    for (int K : { 1, 2, 3, 4, 5, 6, 8, 12, 16 }) {
        if (K > cfg.max_batch) break;
        const int reps = 8;
        eng.reset(0);
        eng.forward_batch(toks.data(), K, 0, dlog, true, 0);      // warm
        CU(cudaDeviceSynchronize());

        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        eng.reset(0);
        cudaEventRecord(a);
        for (int r = 0; r < reps; ++r) eng.forward_batch(toks.data(), K, r * K, dlog, true, 0);
        cudaEventRecord(b);
        CU(cudaEventSynchronize(b));
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        cudaEventDestroy(a); cudaEventDestroy(b);

        const double per_fwd = ms / reps;
        const double per_tok = per_fwd / K;
        if (K == 1) base = per_fwd;
        printf("%4d %10.2f %11.3f %10.3f %8.2fx\n", K, per_fwd, per_tok, per_fwd / base,
               base / per_tok);
    }
    printf("\n'vs K=1' is what ROOFLINE.md §4 predicts as the cost column; 'speedup' is the\n"
           "ceiling a verify of that width could reach if every drafted token were accepted.\n");
    cudaFree(dlog);
    return 0;
}
