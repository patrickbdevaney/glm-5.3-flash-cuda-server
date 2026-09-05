// bench_decode.cu — where the decode step's time actually goes.
//
// The full 45-layer engine runs at 3.4 tok/s against a roofline of 11.7. That gap is a kernel
// efficiency problem, and an aggregate tok/s cannot say which kernel owns it. This runs plain
// AR decode with dprof enabled and prints per-phase achieved bandwidth.
//
// Measured against a streaming probe run IN THIS PROCESS immediately before the loop, for the
// reason OPTIMIZATION_LOG #1 records: the box is shared, and a kernel already at the machine's
// ceiling looks "3x off" when priced against an idle-box constant.
#include "engine.h"
#include "glm5_config.h"
#include "dprof.h"
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

static double measure_stream_bw() {
    const size_t bytes = 1ull << 30;
    float4* buf; float* sink;
    CU(cudaMalloc(&buf, bytes));
    CU(cudaMalloc(&sink, 4));
    CU(cudaMemset(buf, 1, bytes));
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
    ec.n_layer = N_LAYER; ec.max_ctx = 512; ec.max_batch = 1;
    int steps = 24, warm = 6;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]{ return std::string(argv[++i]); };
        if      (a == "--ckpt")    ec.model_dir = next();
        else if (a == "--n-layer") ec.n_layer   = atoi(next().c_str());
        else if (a == "--steps")   steps        = atoi(next().c_str());
        else if (a == "--seqmax")  ec.max_ctx   = atoi(next().c_str());
    }
    setenv("GLM5_DPROF", "1", 1);                       // this tool exists to profile; always on
    dprof_init();

    Engine eng(ec);
    std::vector<int> prompt = {5, 17, 42, 100, 256};
    eng.prefill(prompt);

    // Warm up OUTSIDE the measured window: the first decode pays one-time page faults and
    // kernel-module load, and folding those into phase totals would blame whichever phase ran first.
    for (int i = 0; i < warm; ++i) eng.decode(7, (int)prompt.size() + i, eng.logitsDev(), 0);
    CU(cudaDeviceSynchronize());
    dprof_reset();

    const double bw = measure_stream_bw();
    printf("streaming read RIGHT NOW: %.1f GB/s  (model resident; phases are priced against THIS)\n", bw);

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < steps; ++i)
        eng.decode(7, (int)prompt.size() + warm + i, eng.logitsDev(), 0);
    cudaEventRecord(b);
    CU(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b);

    printf("\nwall: %.2f ms for %d steps = %.2f ms/tok = %.2f tok/s\n",
           ms, steps, ms / steps, 1000.0 * steps / ms);
    printf("roofline at %.1f GB/s with B_tok 19.761 G: %.2f tok/s  ->  we are at %.0f%%\n",
           bw, bw / 19.761, 100.0 * (1000.0 * steps / ms) / (bw / 19.761));
    dprof_report("AR decode", steps, bw);
    return 0;
}
