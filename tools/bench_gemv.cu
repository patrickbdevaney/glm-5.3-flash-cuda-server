// bench_gemv.cu — the NVFP4 dense gemv, in isolation.
//
// The engine-level A/B says the overlay removes 50.6% of B_tok and buys nothing. That could be
// the kernel, the block size, the x traffic, or the shapes; a 100 GiB reload per experiment is
// the wrong instrument for finding out. This times the real shapes on synthetic buffers so the
// loop is seconds, and reports achieved WEIGHT bandwidth against a streaming read measured in
// this process (CLAUDE.md §6.1).
//
// The comparison whose sign is known (§6.4): bf16 must be within a few percent of the streaming
// probe. If it is not, the run is contended and nothing else on the page means anything.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "gemv.h"

using namespace glm5;
#define CU(x) do { cudaError_t e_=(x); if(e_){ fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(e_)); exit(2);} } while(0)

__global__ void k_stream(const float4* __restrict__ p, float* out, size_t n4) {
    float4 a = make_float4(0, 0, 0, 0);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) { float4 v = p[i]; a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w; }
    if (a.x == 12345.678f) out[0] = a.x + a.y + a.z + a.w;
}
static double stream_bw() {
    const size_t bytes = 512ull << 20;
    float4* buf; float* sink;
    CU(cudaMalloc(&buf, bytes)); CU(cudaMalloc(&sink, 4)); CU(cudaMemset(buf, 1, bytes));
    const size_t n4 = bytes / 16;
    k_stream<<<1024, 256>>>(buf, sink, n4); CU(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < 8; ++i) k_stream<<<1024, 256>>>(buf, sink, n4);
    cudaEventRecord(b); CU(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaFree(buf); cudaFree(sink);
    return 8.0 * bytes / (ms / 1000.0) / 1e9;
}

struct Shape { const char* name; int N, K; };

int main(int argc, char** argv) {
    int M = 1, reps = 5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--m") M = atoi(argv[++i]);
        else if (a == "--reps") reps = atoi(argv[++i]);
    }
    const Shape shapes[] = {
        {"kda q/k/v_proj",   8192,  4096},
        {"kda o_proj",       4096,  8192},
        {"mla o_proj",       4096, 16384},
        {"mla q_b_proj",    16384,  1536},
        {"dense gate/up",   12288,  4096},
        {"lm_head",        154880,  4096},
    };

    const double bw = stream_bw();
    printf("streaming read RIGHT NOW: %.1f GB/s   M=%d, %d reps, round-robin\n\n", bw, M, reps);
    printf("%-18s %7s %7s | %9s %8s %6s | %9s %8s %6s | %s\n",
           "shape", "N", "K", "bf16 ms", "GB/s", "%BW", "fp4 ms", "GB/s", "%BW", "speedup");

    for (const Shape& sh : shapes) {
        const size_t NK = (size_t)sh.N * sh.K;
        void *dbf, *dpk, *dsc, *dgs, *dx, *dy;
        CU(cudaMalloc(&dbf, NK * 2)); CU(cudaMemset(dbf, 0x3c, NK * 2));   // ~1.0 in bf16
        CU(cudaMalloc(&dpk, NK / 2)); CU(cudaMemset(dpk, 0x24, NK / 2));
        CU(cudaMalloc(&dsc, NK / 16)); CU(cudaMemset(dsc, 0x3c, NK / 16));
        CU(cudaMalloc(&dgs, 4)); { float g = 1.f; CU(cudaMemcpy(dgs, &g, 4, cudaMemcpyHostToDevice)); }
        CU(cudaMalloc(&dx, (size_t)M * sh.K * 4)); CU(cudaMemset(dx, 0, (size_t)M * sh.K * 4));
        CU(cudaMalloc(&dy, (size_t)M * sh.N * 4));
        WRef Wb(dbf), Wq((const uint8_t*)dpk, (const uint8_t*)dsc, (const float*)dgs);

        gemm((float*)dy, Wb, (const float*)dx, M, sh.N, sh.K, GEMV_BF16, 0);
        gemm((float*)dy, Wq, (const float*)dx, M, sh.N, sh.K, GEMV_BF16, 0);
        CU(cudaDeviceSynchronize());

        std::vector<double> tb, tq;
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        for (int r = 0; r < reps; ++r) {                    // round-robin, not blocks
            float ms;
            cudaEventRecord(a);
            for (int i = 0; i < 4; ++i) gemm((float*)dy, Wb, (const float*)dx, M, sh.N, sh.K, GEMV_BF16, 0);
            cudaEventRecord(b); CU(cudaEventSynchronize(b));
            cudaEventElapsedTime(&ms, a, b); tb.push_back(ms / 4);
            cudaEventRecord(a);
            for (int i = 0; i < 4; ++i) gemm((float*)dy, Wq, (const float*)dx, M, sh.N, sh.K, GEMV_BF16, 0);
            cudaEventRecord(b); CU(cudaEventSynchronize(b));
            cudaEventElapsedTime(&ms, a, b); tq.push_back(ms / 4);
        }
        std::sort(tb.begin(), tb.end()); std::sort(tq.begin(), tq.end());
        const double mb = tb.front(), mq = tq.front();
        const double gb = NK * 2.0 / (mb / 1000.0) / 1e9;
        const double gq = NK * 0.5625 / (mq / 1000.0) / 1e9;
        printf("%-18s %7d %7d | %9.3f %8.1f %5.0f%% | %9.3f %8.1f %5.0f%% | %.2fx\n",
               sh.name, sh.N, sh.K, mb, gb, 100 * gb / bw, mq, gq, 100 * gq / bw, mb / mq);
        cudaFree(dbf); cudaFree(dpk); cudaFree(dsc); cudaFree(dgs); cudaFree(dx); cudaFree(dy);
    }
    return 0;
}
