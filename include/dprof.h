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
