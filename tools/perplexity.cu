// perplexity.cu — settle the ROOFLINE §3 NVFP4 dense-overlay accuracy question with a number that
// bears on generation quality, rather than a per-tensor cosine.
//
// The overlay halves B_tok and buys 1.42x on decode, and every gate it has says it is "close":
// gate_nvfp4 12/12 at cos 0.9950-0.9961 per tensor, gate_stack cos 0.9972 over three layers.
// None of that answers the operator's actual question, which is whether the model got worse.
// So: same tokens, same engine, overlay on vs off, and report NLL.
//
// WHY THE PRESERVED CORPUS. It is 20M tokens already tokenized with THIS model's tokenizer, and
// both conditions index it identically, so the comparison is exact rather than approximate --
// the same property that made the cross-quant control possible (artifacts/iq3m_mtp_capture).
// Every sequence is under DENSE_CTX_LIMIT, so dense MLA is exact and the DSA indexer is never
// engaged; a ppl difference cannot be an attention approximation in disguise.
//
// Per token it dumps the NLL of the true next token and the argmax, so the two runs can be
// compared for top-1 agreement -- a model can hold its perplexity and still change what it says.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "engine.h"
#include "glm5_config.h"

using namespace glm5;
#define CU(x) do { cudaError_t e_=(x); if(e_){ fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(e_)); exit(2);} } while(0)

// One block per row. Two passes in shared memory: max, then sum of exp. Doing this on the GPU
// avoids shipping 620 KB of logits per token to the host.
template <int BS>
__global__ void k_nll(const float* __restrict__ logits, const int32_t* __restrict__ tgt,
                      float* __restrict__ nll, int32_t* __restrict__ am, int V) {
    const int r = blockIdx.x;
    const float* row = logits + (size_t)r * V;
    __shared__ float sm[BS / 32];
    __shared__ int   si[BS / 32];
    float m = -1e30f; int mi = 0;
    for (int i = threadIdx.x; i < V; i += BS) { const float v = row[i]; if (v > m) { m = v; mi = i; } }
    for (int o = 16; o; o >>= 1) {
        const float om = __shfl_down_sync(0xffffffff, m, o);
        const int   oi = __shfl_down_sync(0xffffffff, mi, o);
        if (om > m) { m = om; mi = oi; }
    }
    if ((threadIdx.x & 31) == 0) { sm[threadIdx.x >> 5] = m; si[threadIdx.x >> 5] = mi; }
    __syncthreads();
    __shared__ float mx; __shared__ int mxi;
    if (threadIdx.x == 0) { float v = -1e30f; int vi = 0;
        for (int k = 0; k < BS / 32; ++k) if (sm[k] > v) { v = sm[k]; vi = si[k]; }
        mx = v; mxi = vi; }
    __syncthreads();
    float s = 0.f;
    for (int i = threadIdx.x; i < V; i += BS) s += __expf(row[i] - mx);
    for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0; for (int k = 0; k < BS / 32; ++k) t += sm[k];
        const int y = tgt[r];
        nll[r] = -(row[y] - mx - logf(t));      // -log softmax(row)[y]
        am[r]  = mxi;
    }
}

int main(int argc, char** argv) {
    EngineConfig ec;
    ec.model_dir = std::string(getenv("HOME")) + "/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2";
    std::string corpus = std::string(getenv("HOME")) + "/glm-5.3-reap/artifacts/iq3m_mtp_capture/corpus";
    std::string out;
    int nseq = 32, chunk = 32, minlen = 256, maxlen = 2000;
    // --concat N: glue corpus sequences together into streams of at least N tokens, so the run
    // crosses DENSE_CTX_LIMIT and exercises the DSA sparse path. Above that limit there is no
    // dense answer to compare against (gate_mla_sparse can only assert bit-equality BELOW it), so
    // the check is positional perplexity: with more context a correct model must not get WORSE.
    // A broken sparse path spikes exactly at the boundary, which is a known-sign test.
    int concat = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--ckpt")   ec.model_dir = argv[++i];
        else if (a == "--corpus") corpus = argv[++i];
        else if (a == "--n-seq")  nseq = atoi(argv[++i]);
        else if (a == "--chunk")  chunk = atoi(argv[++i]);
        else if (a == "--min-len") minlen = atoi(argv[++i]);
        else if (a == "--max-len") maxlen = atoi(argv[++i]);
        else if (a == "--out")    out = argv[++i];
        else if (a == "--concat") concat = atoi(argv[++i]);
    }
    ec.max_ctx = (concat > 0 ? concat : maxlen) + 16;   // concatenated streams are the long case
    ec.max_batch = chunk;
    ec.n_layer = N_LAYER;

    // Deterministic selection: chunk files in sorted order, sequences in capture order, first
    // `nseq` that pass the length filter. Both conditions therefore see byte-identical input.
    std::vector<std::vector<int>> seqs;
    for (int c = 0; c < 80 && (int)seqs.size() < nseq; ++c) {
        char lb[512], ib[512];
        snprintf(lb, sizeof lb, "%s/chunk_%04d.bin.lens", corpus.c_str(), c);
        snprintf(ib, sizeof ib, "%s/chunk_%04d.bin.ids",  corpus.c_str(), c);
        FILE* fl = fopen(lb, "rb"); FILE* fi = fopen(ib, "rb");
        if (!fl || !fi) { if (fl) fclose(fl); if (fi) fclose(fi); continue; }
        fseek(fl, 0, SEEK_END); const long nl = ftell(fl) / 4; fseek(fl, 0, SEEK_SET);
        std::vector<int32_t> L(nl); if (fread(L.data(), 4, nl, fl) != (size_t)nl) { }
        long off = 0;
        for (long s = 0; s < nl && (int)seqs.size() < nseq; ++s) {
            const int n = L[s];
            if (n >= minlen && n <= maxlen) {
                std::vector<int32_t> ids(n);
                fseek(fi, off * 4, SEEK_SET);
                if (fread(ids.data(), 4, n, fi) == (size_t)n)
                    seqs.emplace_back(ids.begin(), ids.end());
            }
            off += n;
        }
        fclose(fl); fclose(fi);
    }
    if (concat > 0) {
        std::vector<std::vector<int>> merged;
        std::vector<int> cur;
        for (auto& s2 : seqs) {
            cur.insert(cur.end(), s2.begin(), s2.end());
            if ((int)cur.size() >= concat) { cur.resize(concat); merged.push_back(cur); cur.clear(); }
        }
        seqs.swap(merged);
        printf("concatenated into %zu streams of %d tokens\n", seqs.size(), concat);
    }
    if ((int)seqs.size() < nseq) { fprintf(stderr, "only %zu sequences available\n", seqs.size()); }
    if (seqs.empty()) { fprintf(stderr, "no corpus at %s\n", corpus.c_str()); return 2; }

    Engine eng(ec);
    const char* off = getenv("GLM5_DENSE_NVFP4");
    const bool nvfp4_off = off && std::string(off) == "0";
    printf("condition: dense NVFP4 overlay %s\n", nvfp4_off ? "DISABLED (bf16 reference)" : "ACTIVE");
    printf("%zu sequences, chunk %d, len [%d, %d]\n", seqs.size(), chunk, minlen, maxlen);

    float* dlog; CU(cudaMalloc(&dlog, (size_t)chunk * VOCAB * 4));
    float* dnll; CU(cudaMalloc(&dnll, (size_t)chunk * 4));
    int32_t* dam; CU(cudaMalloc(&dam, (size_t)chunk * 4));
    int32_t* dtg; CU(cudaMalloc(&dtg, (size_t)chunk * 4));

    double tot_nll = 0; long tot_tok = 0;
    std::vector<float> h_nll; std::vector<int32_t> h_am;
    // NLL bucketed by absolute position, in 512-token bands. DENSE_CTX_LIMIT falls inside band 4.
    const int NB_ = 32; std::vector<double> bnll(NB_, 0); std::vector<long> bcnt(NB_, 0);
    for (size_t s = 0; s < seqs.size(); ++s) {
        const std::vector<int>& ids = seqs[s];
        const int n = (int)ids.size();
        eng.reset();
        for (int t = 0; t < n; t += chunk) {
            const int m = std::min(chunk, n - t);
            eng.forward_batch(ids.data() + t, m, t, dlog, /*all_logits=*/true, 0);
            // row i predicts ids[t+i+1]; the last token of the sequence has no target.
            const int rows = std::min(m, n - 1 - t);
            if (rows <= 0) continue;
            std::vector<int32_t> tg(rows);
            for (int i = 0; i < rows; ++i) tg[i] = ids[t + i + 1];
            CU(cudaMemcpy(dtg, tg.data(), rows * 4, cudaMemcpyHostToDevice));
            k_nll<256><<<rows, 256>>>(dlog, dtg, dnll, dam, VOCAB);
            CU(cudaDeviceSynchronize());
            std::vector<float> hn(rows); std::vector<int32_t> ha(rows);
            CU(cudaMemcpy(hn.data(), dnll, rows * 4, cudaMemcpyDeviceToHost));
            CU(cudaMemcpy(ha.data(), dam, rows * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < rows; ++i) {
                tot_nll += hn[i]; ++tot_tok;
                const int band = (t + i) / 512;
                if (band < NB_) { bnll[band] += hn[i]; bcnt[band] += 1; }
            }
            h_nll.insert(h_nll.end(), hn.begin(), hn.end());
            h_am.insert(h_am.end(), ha.begin(), ha.end());
        }
        if ((s + 1) % 8 == 0 || s + 1 == seqs.size())
            printf("  %3zu/%zu seq   tokens %ld   ppl %.5f\n", s + 1, seqs.size(), tot_tok,
                   exp(tot_nll / tot_tok));
        fflush(stdout);
    }
    printf("\nTOKENS %ld   MEAN_NLL %.8f   PPL %.6f\n", tot_tok, tot_nll / tot_tok,
           exp(tot_nll / tot_tok));
    {
        printf("\nppl by position band (DENSE_CTX_LIMIT = %d; a correct sparse path must not\n"
               "spike where the engine switches over):\n", DENSE_CTX_LIMIT);
        for (int b = 0; b < NB_; ++b) if (bcnt[b]) {
            const int lo = b * 512, hi = lo + 511;
            printf("   pos %5d-%5d  n=%7ld  ppl %9.4f%s\n", lo, hi, bcnt[b],
                   exp(bnll[b] / bcnt[b]),
                   (lo <= DENSE_CTX_LIMIT && DENSE_CTX_LIMIT <= hi) ? "   <- switchover" : "");
        }
    }
    if (!out.empty()) {
        FILE* f = fopen(out.c_str(), "wb");
        if (f) { long n = (long)h_nll.size(); fwrite(&n, 8, 1, f);
                 fwrite(h_nll.data(), 4, n, f); fwrite(h_am.data(), 4, n, f); fclose(f);
                 printf("wrote %s (%ld tokens)\n", out.c_str(), n); }
    }
    return 0;
}
