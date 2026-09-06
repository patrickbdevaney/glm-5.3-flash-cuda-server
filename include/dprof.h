// dprof.h — named-phase GPU timing, for attributing a decode step to its sub-operations.
//
// WHY THIS EXISTS HERE. The full 45-layer engine decodes at 3.4 tok/s against a roofline of
// 11.7 (231.5 GB/s measured with the model resident, / B_tok = 19.761 G). That 29% is a KERNEL
// EFFICIENCY gap, not an algorithmic one, and a single aggregate number cannot say which kernel
// owns it. This splits the step into phases and prints each one's ACHIEVED BANDWIDTH, which is
// the only figure that distinguishes "this phase is big" from "this phase is slow".
//
// The enum is RETARGETED for GLM-5.3 and deliberately does not match the one in
// deepseek-v4-flash-0731-cuda that it was ported from: that model has no KDA layers, no
// hyper-connections and a different attention. Rows are not comparable across the two repos.
//
// Design notes, kept from the original because each one was learned the hard way:
//  - Off by default and compiled to nothing but a branch; enabled with GLM5_DPROF=1.
//  - Events record into a preallocated pool and are NOT synchronised until dprof_report(), so the
//    instrumented path stays asynchronous. A sync per phase would itself create the stalls we are
//    hunting -- the trap `dspark_forward_head` fell into.
//  - The pool is fixed-size; overflow stops recording rather than reallocating mid-measurement.
//  - A row over 100% of bandwidth is flagged with `!`. It used to mean exactly one thing -- a
//    wrong byte count, never a fast kernel. It now means one of TWO things, and they are told
//    apart by asking whether the tensor fits in L2:
//      * a wrong byte count (still the usual cause, and how the MoE router's double-count was
//        caught in OPTIMIZATION_LOG #9), or
//      * traffic genuinely served from L2 rather than DRAM, which %BW's denominator does not
//        model. The MLA latent cache is the known case: OPTIMIZATION_LOG #15 measured it at
//        487-610 GB/s against a 237 GB/s streaming read, which is why `sdpa:context` prints over
//        100% and is nonetheless correct.
//    Weights are far too large to be resident, so any weight-dominated row over 100% is still a
//    bug in the count.
//  - Children are checked against their parent and the report says INVALID rather than printing a
//    plausible-looking table, because a mark recorded outside its parent's window is silent.
#pragma once
#include <cuda_runtime.h>

enum DProfId {
    // ---- level 1: the eight phases of a decoder layer. These sum to TOTAL. ----
    DP_HC_PRE_ATTN = 0, DP_NORM_ATTN, DP_ATTN, DP_HC_POST_ATTN,
    DP_HC_PRE_FFN,      DP_NORM_FFN,  DP_FFN,  DP_HC_POST_FFN,
    // ---- level 1, outside the layer loop. Also in TOTAL. ----
    DP_EMBED, DP_HEAD_MEAN, DP_LM_HEAD,

    // ---- level 2: DP_ATTN split by layer flavour (34 KDA vs 11 MLA/DSA). ----
    // The roofline's headline is that KDA is 47.4% of B_tok and MLA only 13.1%, so an
    // undifferentiated ATTENTION row would hide a 4x difference in where the bytes are.
    DP_KDA, DP_MLA,
    // inside KDA
    DP_K_QKVCONV, DP_K_GATES, DP_K_NORMQK, DP_K_RECUR, DP_K_OUTNORM, DP_K_OPROJ,
    // inside MLA
    DP_M_QPROJ, DP_M_KV, DP_M_INDEXER, DP_M_ABSORB, DP_M_SDPA, DP_M_OPROJ,
    // level 3, inside DP_M_SDPA. It is the only MLA row that grows with context -- everything
    // else in the engine costs the same at token 100 and token 3000 -- so which of its three
    // kernels owns it decides whether the long-context lever is cache traffic or occupancy.
    DP_S_SCORES, DP_S_SOFTMAX, DP_S_CONTEXT,

    // ---- level 2: DP_FFN split by flavour (42 MoE layers vs 3 dense). ----
    DP_MOE, DP_DENSE,
    DP_E_ROUTER, DP_E_ACT, DP_E_DOWN,
    DP_N
};

extern bool g_dprof_on;

void dprof_init(int max_marks = 65536);

// Rescale the byte model for the ROOFLINE §3 NVFP4 dense overlay. The engine calls this at load
// time; without it every AR-path row is priced against weights the engine is no longer reading.
void dprof_set_nvfp4_dense(bool on);
// ---- measured byte accounting ----
//
// kBytes below is a PER-DECODED-TOKEN WEIGHT MODEL, and in prefill that is simply the wrong
// question: a weight read at chunk width M serves M tokens, so every gemm-backed row was priced
// ~M/ceil(M/MCHUNK) too high. It showed: `ffn:moe` printed 105% of bandwidth and `lm_head` 11399%,
// and by this file's own rule a row over 100% is a wrong byte count, never a fast kernel.
//
// MoE is worse than wrong-by-a-factor. Since the expert-gathering kernel (OPTIMIZATION_LOG #12)
// each distinct expert is read once per chunk rather than once per (token, slot), so the true
// count is DATA-DEPENDENT -- it depends on how many distinct experts M tokens happened to route
// to -- and no constant can express it.
//
// So the launch sites report what they actually read. dprof_bytes() credits every mark that is
// currently open, which makes a parent row the sum of its children for free. A row with measured
// bytes is priced from them; a row without falls back to the kBytes constant.
void dprof_bytes(double bytes);            // credit all currently-open marks
void dprof_bytes_to(int id, double bytes); // credit one row (for counts only known at report time)

// Registered by a translation unit that can only resolve its byte count at report time -- MoE,
// whose work-item count lives in a device counter and would cost a stream sync to read early.
// Called with credit=true from dprof_report, and credit=false from dprof_reset so a
// warm-up's work is discarded rather than billed to the timed run.
void dprof_set_flush(void (*fn)(bool credit));

void dprof_begin(int id, cudaStream_t s = 0);
void dprof_end(int id, cudaStream_t s = 0);
void dprof_reset();

// Sync, sum elapsed per id, print, reset.
//   `tag`      labels the row block.
//   `n_steps`  decode steps covered, so per-token byte counts become a rate.
//   `bw_gbs`   streaming bandwidth measured in THIS process, immediately before the run. Phases
//              are reported as a percentage of it. Pass 0 to omit the efficiency columns -- an
//              idle-box constant here is how OPTIMIZATION_LOG #1 called a kernel "3x off" when it
//              was already at the machine's ceiling.
void dprof_report(const char* tag, int n_steps = 1, double bw_gbs = 0.0);
