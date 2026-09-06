// dprof.cu — see include/dprof.h.
#include "dprof.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

bool g_dprof_on = false;
static bool g_inited = false;
static std::vector<cudaEvent_t> g_pool;
static std::vector<int>  g_id;
static std::vector<int>  g_open_at;          // index into g_pool of the open BEGIN, per id
static int g_used = 0;
static int g_open[DP_N];

static const char* kName[DP_N] = {
    "hc:pre_attn", "norm:attn", "ATTENTION", "hc:post_attn",
    "hc:pre_ffn",  "norm:ffn",  "FFN",       "hc:post_ffn",
    "embed", "head:mean", "lm_head",
    "  attn:kda", "  attn:mla",
    "    kda:qkv+conv", "    kda:gates", "    kda:norm_qk", "    kda:recurrence",
    "    kda:out_norm", "    kda:o_proj",
    "    mla:q_proj", "    mla:kv", "    mla:indexer", "    mla:absorb_q", "    mla:sdpa",
    "    mla:o_proj",
    "      sdpa:scores", "      sdpa:softmax", "      sdpa:context",
    "  ffn:moe", "  ffn:dense",
    "    moe:router", "    moe:w13+act", "    moe:w2+combine",
};

// Bytes of WEIGHT read per decoded token, from ROOFLINE.md §1 (safetensors headers, not estimates).
// Zero means "this phase moves no meaningful weight bytes" -- for those a GB/s figure would be
// noise, so the report prints a dash instead of a number that invites a wrong conclusion.
//
// A row printing over 100% of bandwidth is not a fast kernel -- it is a wrong byte count on that
// row, because no phase can beat the memory system. That is exactly how the 'moe router' bucket
// was caught double-counting the dense MLPs' gate_proj (OPTIMIZATION_LOG #9).
//
// The split of the 9.366 G KDA bucket across its six sub-phases follows the tensor shapes:
// q/k/v_proj are three bf16 [8192,4096] (67.1 MB each) and land in qkv+conv, o_proj is the fourth,
// and the low-rank gate projections are the 0.5 G remainder. The recurrence is state, not weights.
static double kBytes[DP_N] = {
    /* hc:pre_attn   */ 0.0355e9, /* norm:attn */ 0.0, /* ATTENTION */ 11.950e9, /* hc:post_attn */ 0.0,
    /* hc:pre_ffn    */ 0.0355e9, /* norm:ffn  */ 0.0, /* FFN       */  6.307e9, /* hc:post_ffn  */ 0.0,
    /* embed         */ 0.0, /* head:mean */ 0.0, /* lm_head */ 1.269e9,
    /* attn:kda      */ 9.366e9, /* attn:mla */ 2.584e9,
    /* kda:qkv+conv  */ 6.850e9, /* kda:gates */ 0.500e9, /* kda:norm_qk */ 0.0,
    /* kda:recurrence*/ 0.305e9, /* kda:out_norm */ 0.0, /* kda:o_proj */ 2.283e9,
    // The MLA sub-phases are deliberately unpriced. The 2.584 G bucket does not decompose to these
    // six marks without guessing, and the indexer's own 0.164 G is a LONG-CONTEXT figure -- at the
    // 512-token context this bench runs, it moves a fraction of that and printed a nonsensical
    // 180% of bandwidth. A dash is the honest cell; a wrong number here invites a wrong lever.
    /* mla:q_proj    */ 0.0, /* mla:kv */ 0.0, /* mla:indexer */ 0.0, /* mla:absorb_q */ 0.0,
    /* mla:sdpa      */ 0.0, /* mla:o_proj */ 0.0,
    /* sdpa:scores   */ 0.0, /* sdpa:softmax */ 0.0, /* sdpa:context */ 0.0,
    /* ffn:moe       */ 5.401e9, /* ffn:dense */ 0.906e9,
    /* moe:router    */ 0.050e9, /* moe:w13+act */ 3.568e9, /* moe:w2+combine */ 1.783e9,
};

// When the ROOFLINE §3 overlay is bound, the dense weights on the AR path are 0.5625 B/weight
// instead of 2.0 and every row above that counts them is wrong by 3.56x. Leaving them wrong is
// not a cosmetic problem: the report would print kda:qkv+conv at 158% of bandwidth, and the rule
// this file is built around is that a row over 100% is a WRONG BYTE COUNT, never a fast kernel.
//
// Only the converted families scale. kda:recurrence is state; ffn:moe was already NVFP4; the
// hyper-connection `fn` tensors and MLA's kv_b are still bf16 because neither is read through
// gemv, so mla keeps a bf16 share (kv_b is 33.6 of its 249.8 MB per layer).
void dprof_set_nvfp4_dense(bool on) {
    if (!on) return;
    const double q = 0.5625 / 2.0;
    kBytes[DP_K_QKVCONV] = 6.850e9 * q;
    kBytes[DP_K_GATES]   = 0.500e9 * q;
    kBytes[DP_K_OPROJ]   = 2.283e9 * q;
    kBytes[DP_KDA]       = kBytes[DP_K_QKVCONV] + kBytes[DP_K_GATES] + kBytes[DP_K_OPROJ]
                         + kBytes[DP_K_RECUR];
    kBytes[DP_MLA]       = 2.584e9 * ((216.2 * q + 33.55) / 249.8);
    kBytes[DP_LM_HEAD]   = 1.269e9 * q;
    kBytes[DP_DENSE]     = 0.906e9 * q;
    kBytes[DP_ATTN]      = kBytes[DP_KDA] + kBytes[DP_MLA];
    kBytes[DP_FFN]       = kBytes[DP_MOE] + kBytes[DP_DENSE];
}

void dprof_init(int max_marks){
    if (g_inited) return;
    g_dprof_on = getenv("GLM5_DPROF") != nullptr;
    if (!g_dprof_on) { g_inited = true; return; }
    g_pool.resize(max_marks);
    g_id.resize(max_marks);
    for (int i = 0; i < max_marks; ++i) cudaEventCreate(&g_pool[i]);
    for (int i = 0; i < DP_N; ++i) g_open[i] = -1;
    g_inited = true;
    printf("[dprof] enabled, %d marks\n", max_marks);
}

void dprof_begin(int id, cudaStream_t s){
    if (!g_dprof_on || g_used >= (int)g_pool.size()) return;
    cudaEventRecord(g_pool[g_used], s);
    g_id[g_used] = id;
    g_open[id] = g_used;
    ++g_used;
}

void dprof_end(int id, cudaStream_t s){
    if (!g_dprof_on || g_open[id] < 0 || g_used >= (int)g_pool.size()) return;
    cudaEventRecord(g_pool[g_used], s);
    g_id[g_used] = -1 - id;                 // negative marks an END
    ++g_used;
}

void dprof_reset(){ g_used = 0; for (int i = 0; i < DP_N; ++i) g_open[i] = -1; }

void dprof_report(const char* tag, int n_steps, double bw_gbs){
    if (!g_dprof_on || !g_used) return;
    cudaDeviceSynchronize();

    double sum[DP_N] = {0}; int cnt[DP_N] = {0}; int open_idx[DP_N];
    for (int i = 0; i < DP_N; ++i) open_idx[i] = -1;
    for (int i = 0; i < g_used; ++i) {
        const int raw = g_id[i];
        if (raw >= 0) { open_idx[raw] = i; }
        else {
            const int id = -1 - raw;
            if (open_idx[id] < 0) continue;
            float ms = 0.f;
            cudaEventElapsedTime(&ms, g_pool[open_idx[id]], g_pool[i]);
            sum[id] += ms; cnt[id] += 1; open_idx[id] = -1;
        }
    }
    double tot = 0;
    for (int i = 0; i <= DP_LM_HEAD; ++i) tot += sum[i];

    printf("\n[dprof] %s — decode step by sub-op, summed over %d step(s) and all layers\n", tag, n_steps);
    if (bw_gbs > 0)
        printf("[dprof] %-20s %10s %7s %8s %10s %8s %7s\n",
               "phase", "ms", "%", "calls", "GB/tok", "GB/s", "%BW");
    else
        printf("[dprof] %-20s %10s %7s %8s\n", "phase", "ms", "%", "calls");

    for (int i = 0; i < DP_N; ++i) {
        if (!cnt[i]) continue;
        if (bw_gbs > 0 && kBytes[i] > 0) {
            const double gbs = kBytes[i] * n_steps / (sum[i] / 1000.0) / 1e9;
            printf("[dprof] %-20s %10.2f %6.1f%% %8d %10.3f %8.1f %6.0f%%\n",
                   kName[i], sum[i], 100.0 * sum[i] / tot, cnt[i],
                   kBytes[i] / 1e9, gbs, 100.0 * gbs / bw_gbs);
        } else if (bw_gbs > 0) {
            printf("[dprof] %-20s %10.2f %6.1f%% %8d %10s %8s %7s\n",
                   kName[i], sum[i], 100.0 * sum[i] / tot, cnt[i], "-", "-", "-");
        } else {
            printf("[dprof] %-20s %10.2f %6.1f%% %8d\n", kName[i], sum[i], 100.0 * sum[i] / tot, cnt[i]);
        }
    }
    printf("[dprof] %-20s %10.2f  (%.2f ms/step, %.2f tok/s)\n",
           "TOTAL", tot, tot / n_steps, 1000.0 * n_steps / tot);

    // A child that outlasts its parent means a mark was recorded outside the reported window.
    // Print the contradiction rather than a table that merely looks plausible.
    const double c_attn = sum[DP_KDA] + sum[DP_MLA];
    const double c_ffn  = sum[DP_MOE] + sum[DP_DENSE];
    if (c_attn > sum[DP_ATTN] * 1.02)
        printf("[dprof] *** INVALID: attn children %.2f ms > ATTENTION %.2f ms\n", c_attn, sum[DP_ATTN]);
    if (c_ffn > sum[DP_FFN] * 1.02)
        printf("[dprof] *** INVALID: ffn children %.2f ms > FFN %.2f ms\n", c_ffn, sum[DP_FFN]);
    dprof_reset();
}
