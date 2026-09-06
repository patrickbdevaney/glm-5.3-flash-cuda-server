// bench_context.cu — find the (HG, NT) optimum for k_context without rebooting the engine.
//
// WHY THIS EXISTS. k_context is the largest sdpa row and the only MLA cost that grows with
// context. Two knobs trade against each other and the byte model does not predict the winner:
//   * HG (heads per block) -- lowering it multiplies blocks AND cache re-reads by the same factor.
//   * NT (t-tiles)         -- raising it multiplies blocks at NO extra cache traffic, because the
//                             tiles own disjoint slices, but adds a partial buffer and a reduce.
// HG=16/NT=8 has 2x the blocks and 1/4 the cache traffic of HG=4/NT=1 and measured SLOWER in the
// engine, so the grid gets swept rather than reasoned about.
//
// Measured on synthetic buffers of the real shapes, against a streaming read taken in THIS process
// (CLAUDE.md §6.1), round-robin over the grid (§6.2), min reported with median beside it (§6.3).
// The known-sign anchor (§6.4): for a fixed NT, cost must rise as n_tok rises.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>
#include "mla_context.cuh"

using namespace glm5;
#define CU(x) do { cudaError_t e_=(x); if(e_){ fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(e_)); exit(2);} } while(0)

__global__ void k_stream(const float4* __restrict__ p, float* out, size_t n4) {
    float4 a = make_float4(0,0,0,0);
    for (size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x; i < n4; i += (size_t)gridDim.x*blockDim.x) {
        float4 v = p[i]; a.x+=v.x; a.y+=v.y; a.z+=v.z; a.w+=v.w;
    }
    if (a.x == 12345.678f) out[0] = a.x+a.y+a.z+a.w;
}
static double stream_bw() {
    const size_t bytes = 512ull<<20; float4* b; float* sink;
    CU(cudaMalloc(&b,bytes)); CU(cudaMalloc(&sink,4)); CU(cudaMemset(b,1,bytes));
    const size_t n4 = bytes/sizeof(float4);
    k_stream<<<1024,256>>>(b,sink,n4); CU(cudaDeviceSynchronize());
    cudaEvent_t x,y; cudaEventCreate(&x); cudaEventCreate(&y);
    cudaEventRecord(x); for (int i=0;i<8;++i) k_stream<<<1024,256>>>(b,sink,n4); cudaEventRecord(y);
    CU(cudaEventSynchronize(y)); float ms=0; cudaEventElapsedTime(&ms,x,y);
    cudaFree(b); cudaFree(sink); return 8.0*bytes/(ms/1000.0)/1e9;
}

static constexpr int Hh_ = ctxk::Hh, Lk_ = ctxk::Lk;
struct Cfg { int hg, nt; };
static float *g_part, *g_ctx, *g_s, *g_cache, *g_ref;
static int g_max_ctx;

// One (HG, NT) point. Templated because the kernels are; dispatched through a table below.
template <int HG, int NT>
static void run_point(int n_tok, cudaStream_t s) {
    k_context_part<HG,NT><<<dim3(Hh_/HG, NT), Lk_, 0, s>>>(g_part, g_s, g_cache, n_tok, g_max_ctx);
    k_context_reduce<NT><<<Hh_*Lk_/256, 256, 0, s>>>(g_ctx, g_part);
}
typedef void (*RunFn)(int, cudaStream_t);
struct Point { const char* name; int hg, nt; RunFn fn; };
#define P(h,n) { #h "/" #n, h, n, &run_point<h,n> }
static Point kPoints[] = {
    P(1,4), P(1,8), P(1,16),
    P(2,4), P(2,8), P(2,16),
    P(4,1), P(4,2), P(4,4), P(4,8), P(4,16), P(4,32),
    P(8,4), P(8,8),
    P(16,4), P(16,8), P(16,16),
};
#undef P

int main(int argc, char** argv) {
    std::vector<int> ns = {256, 1024, 2048};
    int reps = 5;
    for (int i=1;i<argc;++i){ std::string a=argv[i];
        if (a=="--reps") reps=atoi(argv[++i]);
        else if (a=="--n"){ ns.clear(); std::string w=argv[++i],c;
            for(char ch:w){ if(ch==','){ns.push_back(atoi(c.c_str()));c.clear();} else c.push_back(ch);} 
            if(!c.empty()) ns.push_back(atoi(c.c_str())); } }
    g_max_ctx = *std::max_element(ns.begin(), ns.end());

    CU(cudaMalloc(&g_cache, (size_t)g_max_ctx*Lk_*sizeof(float)));
    CU(cudaMalloc(&g_s,     (size_t)Hh_*g_max_ctx*sizeof(float)));
    CU(cudaMalloc(&g_ctx,   (size_t)Hh_*Lk_*sizeof(float)));
    CU(cudaMalloc(&g_ref,   (size_t)Hh_*Lk_*sizeof(float)));
    CU(cudaMalloc(&g_part,  (size_t)32*Hh_*Lk_*sizeof(float)));
    {   // deterministic non-trivial content, so a kernel that reads the wrong thing shows up
        std::vector<float> h((size_t)g_max_ctx*Lk_); unsigned st=7u;
        for (auto& v : h){ st=st*1664525u+1013904223u; v=(float)(st>>8)/8388608.0f-1.0f; }
        CU(cudaMemcpy(g_cache,h.data(),h.size()*4,cudaMemcpyHostToDevice));
        // `s` MUST BE A SOFTMAX OUTPUT, not random signs. With random +/- weights the 2048-term
        // sum cancels to near zero and a per-element relative error is measured against noise --
        // the first run of this bench rejected every NT>1 point at max_rel ~1e-2 for exactly that
        // reason, which is an artefact of the test data and not of the kernel. Non-negative rows
        // summing to 1 are both what the kernel actually sees and well-conditioned.
        std::vector<float> hs((size_t)Hh_*g_max_ctx);
        for (int h=0; h<Hh_; ++h){
            double tot=0;
            for (int t=0;t<g_max_ctx;++t){ st=st*1664525u+1013904223u;
                const float e=(float)(st>>8)/8388608.0f; hs[(size_t)h*g_max_ctx+t]=e; tot+=e; }
            for (int t=0;t<g_max_ctx;++t) hs[(size_t)h*g_max_ctx+t] /= (float)tot;
        }
        CU(cudaMemcpy(g_s,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));
    }

    const double bw = stream_bw();
    printf("streaming read RIGHT NOW: %.1f GB/s\n", bw);
    printf("k_context (HG/NT), %d n_tok values, %d reps, ROUND-ROBIN\n\n", (int)ns.size(), reps);

    const int NP = sizeof(kPoints)/sizeof(kPoints[0]);
    // CORRECTNESS FIRST: every (HG,NT) must reproduce the same ctx. A point that fails to launch
    // costs nothing and looks like a win -- that is exactly how the HG=32 row got believed once.
    std::vector<std::vector<float>> out(NP);
    for (int p=0;p<NP;++p){
        CU(cudaMemset(g_ctx,0,(size_t)Hh_*Lk_*4));
        kPoints[p].fn(ns.back(), 0);
        cudaError_t e = cudaGetLastError();
        if (e) { printf("  %-8s LAUNCH FAILED: %s\n", kPoints[p].name, cudaGetErrorString(e)); }
        CU(cudaDeviceSynchronize());
        out[p].resize((size_t)Hh_*Lk_);
        CU(cudaMemcpy(out[p].data(), g_ctx, out[p].size()*4, cudaMemcpyDeviceToHost));
    }
    // RELATIVE L2, not max per-element relative. Splitting t reassociates the sum, so NT>1 differs
    // from NT=1 in the last bits -- that is expected and unavoidable. But 32768 outputs mean a few
    // land near zero, and dividing a 1e-7 absolute error by a 1e-6 output reports 1e-1 and rejects
    // a correct kernel. The first two runs of this bench did exactly that, grouped perfectly by NT
    // and stable across reps, which is the signature of reassociation rather than a bug.
    int bad = 0;
    double rn = 0; for (double v : std::vector<double>(out[0].begin(), out[0].end())) rn += v*v;
    rn = sqrt(rn);
    for (int p=1;p<NP;++p){
        double d=0; for (size_t i=0;i<out[0].size();++i){ const double e=out[p][i]-out[0][i]; d+=e*e; }
        const double rel = sqrt(d)/rn;
        if (!(rel < 1e-5)) { printf("  MISMATCH %-8s vs %s: relL2 %.3e\n", kPoints[p].name, kPoints[0].name, rel); ++bad; }
    }
    printf("correctness: %d/%d points agree with %s\n\n", NP-1-bad, NP-1, kPoints[0].name);
    if (bad) { printf("REFUSING TO REPORT TIMINGS FOR A GRID THAT DISAGREES\n"); return 1; }

    std::vector<std::vector<std::vector<double>>> t(ns.size(),
        std::vector<std::vector<double>>(NP));
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int ITER = 20;
    for (int r=0;r<reps;++r)
      for (size_t ni=0;ni<ns.size();++ni)
        for (int p=0;p<NP;++p){
            kPoints[p].fn(ns[ni],0); CU(cudaDeviceSynchronize());
            cudaEventRecord(a);
            for (int k=0;k<ITER;++k) kPoints[p].fn(ns[ni],0);
            cudaEventRecord(b); CU(cudaEventSynchronize(b));
            float ms=0; cudaEventElapsedTime(&ms,a,b);
            t[ni][p].push_back(ms/ITER);
        }

    for (size_t ni=0;ni<ns.size();++ni){
        const int n = ns[ni];
        printf("n_tok = %d\n", n);
        printf("  %-8s %6s %7s %10s %10s %10s\n","HG/NT","blocks","reads","min us","med us","GB/s");
        double best=1e30; const char* bn="";
        for (int p=0;p<NP;++p){
            auto v=t[ni][p]; std::sort(v.begin(),v.end());
            const double mn=v.front(), md=v[v.size()/2];
            const int reads = Hh_/kPoints[p].hg;
            // bytes: cache read once per head-group, plus the partial buffer written and re-read
            const double bytes = (double)reads*n*Lk_*4.0
                               + 2.0*(double)kPoints[p].nt*Hh_*Lk_*4.0;
            printf("  %-8s %6d %7d %10.1f %10.1f %10.1f%s\n", kPoints[p].name,
                   (Hh_/kPoints[p].hg)*kPoints[p].nt, reads, mn*1000, md*1000,
                   bytes/(mn/1000.0)/1e9, mn<best?"   <-":"");
            if (mn<best){best=mn;bn=kPoints[p].name;}
        }
        printf("  best: %s at %.1f us\n\n", bn, best*1000);
    }
    return 0;
}
