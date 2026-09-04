// bench_kda.cu — what fraction of achievable bandwidth does the KDA decode step reach?
//
// KDA is 47.4% of B_tok (ROOFLINE.md §1), so its bandwidth efficiency very nearly IS the engine's.
// The byte model below is exact — it counts what the kernels are obliged to move, nothing else —
// so "% of achievable" here is an honest number, not an ncu-derived one (CLAUDE.md §3).
#include "kda.h"
#include "gemv.h"
#include "glm5_config.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

// Self-calibrating: the same binary measures a pure streaming read immediately before the KDA
// timing loop, so the denominator is what the box can deliver RIGHT NOW, not an inherited number.
// This matters on a shared box: with an unattended stage running, achievable read collapsed from
// 240 GB/s to 82.8 GB/s, and a kernel at 85% of achievable would have been misreported as 29%.
__global__ void k_stream_read(const float4* __restrict__ p, size_t n4, float* __restrict__ out) {
    float acc = 0.f;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = p[i]; acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 1234.5678f) out[0] = acc;
}

using namespace glm5;
#define CU(x) do { cudaError_t e_=(x); if(e_){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(2);} } while(0)

static std::string DIR = "ref/kda";
static std::vector<float> load(const char* n) {
    std::string p = DIR + "/" + n; FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "FATAL: %s missing; run ref/gen_kda.py\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END); long b = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v(b / 4); if (fread(v.data(), 4, v.size(), f) != v.size()) exit(2);
    fclose(f); return v;
}
static void* dev_bf16(const std::vector<float>& h) {
    float* t; CU(cudaMalloc(&t, h.size() * 4)); CU(cudaMemcpy(t, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    void* d; CU(cudaMalloc(&d, h.size() * 2)); f32_to_bf16(d, t, h.size(), 0);
    CU(cudaDeviceSynchronize()); CU(cudaFree(t)); return d;
}
static void* dev_f32(const std::vector<float>& h) {
    void* d; CU(cudaMalloc(&d, h.size() * 4)); CU(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); return d;
}

int main(int argc, char** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 200;
    const int Hh = KDA_HEADS, D = KDA_HEAD_DIM, Qd = KDA_QKV_DIM;

    KdaWeights W{}; W.dtype = GEMV_BF16;
    W.q_proj = dev_bf16(load("w_q_proj.bin")); W.k_proj = dev_bf16(load("w_k_proj.bin"));
    W.v_proj = dev_bf16(load("w_v_proj.bin")); W.o_proj = dev_bf16(load("w_o_proj.bin"));
    W.f_a = dev_bf16(load("w_f_a.bin")); W.f_b = dev_bf16(load("w_f_b.bin"));
    W.g_a = dev_bf16(load("w_g_a.bin")); W.g_b = dev_bf16(load("w_g_b.bin"));
    W.b_proj = dev_bf16(load("w_b_proj.bin")); W.o_norm = dev_bf16(load("w_o_norm.bin"));
    W.conv1d = dev_f32(load("w_conv1d.bin"));
    W.dt_bias = (const float*)dev_f32(load("w_dt_bias.bin"));
    W.A_log = (const float*)dev_f32(load("w_A_log.bin"));

    float* d_x = (float*)dev_f32(load("x.bin"));
    float* d_conv = (float*)dev_f32(load("conv_state_in.bin"));
    float* d_S = (float*)dev_f32(load("S_in.bin"));
    float *d_y, *d_ws;
    CU(cudaMalloc(&d_y, HIDDEN * 4));
    CU(cudaMalloc(&d_ws, kda_workspace_floats() * 4));
    CU(cudaMemset(d_ws, 0, kda_workspace_floats() * 4));

    // exact byte model for one layer, one token
    const double W_bf16 = 2.0;
    double bytes = 0;
    bytes += 4.0 * Qd * HIDDEN * W_bf16;                    // q,k,v,o_proj
    bytes += 2.0 * KDA_GATE_RANK * HIDDEN * W_bf16;         // f_a, g_a
    bytes += 2.0 * (double)Qd * KDA_GATE_RANK * W_bf16;     // f_b, g_b
    bytes += (double)Hh * HIDDEN * W_bf16;                  // b_proj
    bytes += 3.0 * Qd * KDA_CONV_K * 4;                     // conv1d fp32
    bytes += (double)Qd * 4 + Hh * 4 + D * W_bf16;          // dt_bias, A_log, o_norm
    bytes += 2.0 * Hh * D * D * 4;                          // recurrent state, read + write
    bytes += 2.0 * 3 * Qd * KDA_CONV_STATE * 4;             // conv window, read + write

    // ---- calibrate: achievable streaming read, measured now, on this machine, in this state ----
    double achievable;
    {
        const size_t nbytes = 1024ull << 20;            // 1 GiB, far past any cache
        float4* p; CU(cudaMalloc(&p, nbytes)); CU(cudaMemset(p, 0, nbytes));
        float* o; CU(cudaMalloc(&o, 4));
        int sms; CU(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
        const size_t n4 = nbytes / sizeof(float4);
        cudaEvent_t ca, cb; CU(cudaEventCreate(&ca)); CU(cudaEventCreate(&cb));
        for (int i = 0; i < 3; ++i) k_stream_read<<<sms * 16, 256>>>(p, n4, o);
        CU(cudaDeviceSynchronize());
        std::vector<double> bw;
        for (int r = 0; r < 5; ++r) {
            CU(cudaEventRecord(ca));
            for (int i = 0; i < 5; ++i) k_stream_read<<<sms * 16, 256>>>(p, n4, o);
            CU(cudaEventRecord(cb)); CU(cudaEventSynchronize(cb));
            float e; CU(cudaEventElapsedTime(&e, ca, cb));
            bw.push_back(nbytes * 5.0 / (e * 1e-3) / 1e9);
        }
        std::sort(bw.begin(), bw.end());
        achievable = bw[bw.size() / 2];
        CU(cudaFree(p)); CU(cudaFree(o));
    }

    for (int i = 0; i < 20; ++i) kda_decode_step(d_x, W, d_conv, d_S, d_y, d_ws, 0);
    CU(cudaDeviceSynchronize());

    std::vector<double> ms;
    cudaEvent_t a, b; CU(cudaEventCreate(&a)); CU(cudaEventCreate(&b));
    for (int r = 0; r < 5; ++r) {
        CU(cudaEventRecord(a));
        for (int i = 0; i < iters; ++i) kda_decode_step(d_x, W, d_conv, d_S, d_y, d_ws, 0);
        CU(cudaEventRecord(b)); CU(cudaEventSynchronize(b));
        float e; CU(cudaEventElapsedTime(&e, a, b)); ms.push_back(e / iters);
    }
    std::sort(ms.begin(), ms.end());
    const double t = ms[ms.size() / 2];
    const double gbs = bytes / (t * 1e-3) / 1e9;

    printf("KDA decode step, one layer, batch 1 (bf16 weights)\n");
    printf("  byte model        %.2f MB/layer\n", bytes / 1e6);
    printf("  median            %.3f ms   (best %.3f, worst %.3f, %d iters x 5)\n",
           t, ms.front(), ms.back(), iters);
    printf("  achieved          %.1f GB/s\n", gbs);
    printf("  achievable NOW    %.1f GB/s  (streaming probe, same process, same instant)\n", achievable);
    printf("  efficiency        %.1f%% of concurrent achievable   |   %.1f%% of 240 uncontended\n",
           100 * gbs / achievable, 100 * gbs / 240);
    if (achievable < 200)
        printf("  NOTE: the box is contended (%.0f GB/s << 240 uncontended). The efficiency figure\n"
               "        against concurrent achievable is the meaningful one; the absolute ms is not.\n",
               achievable);
    printf("\n  extrapolated to %d KDA layers: %.1f ms/token\n", N_KDA_LAYER, t * N_KDA_LAYER);
    printf("  whole-model AR at this EFFICIENCY on an idle box: %.2f tok/s\n",
           240e9 * (gbs / achievable) / 19.761e9);
    return 0;
}
