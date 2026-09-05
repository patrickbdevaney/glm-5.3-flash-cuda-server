# ROOFLINE.md — what decode costs, and where the levers actually are

Everything here is read out of the checkpoint by `tools/roofline.py` (safetensors headers, no
estimates) and priced against bandwidth **measured on this box**, not vendor spec.

Regenerate with:

    python3 tools/roofline.py ~/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2

---

## 1. The headline, and it is not the one you expect

> **The KDA linear-attention projections are 47.4% of `B_tok` — nearly twice the entire MoE.**

| bucket | resident | % resident | per token | **% `B_tok`** |
|---|---|---|---|---|
| moe routed experts (8 of 144) | 87.65 G | 83.2% | 4.756 G | 24.1% |
| **kda linear attention** | 9.37 G | 8.9% | **9.366 G** | **47.4%** |
| mla (dsa full-attn, 11 layers) | 2.82 G | 2.7% | 2.584 G | 13.1% |
| lm_head | 1.27 G | 1.2% | 1.269 G | 6.4% |
| moe shared expert | 0.61 G | 0.6% | 0.595 G | 3.0% |
| dense mlp (layers 0–2) | 0.91 G | 0.9% | 0.906 G | 4.6% |
| moe router | 0.05 G | 0.0% | 0.050 G | 0.3% |
| dsa indexer | 0.18 G | 0.2% | 0.164 G | 0.8% |
| hyper-connections | 0.07 G | 0.1% | 0.071 G | 0.4% |
| embed / vision tower / mtp glue | 2.47 G | 2.4% | ~0 | 0% |
| **TOTAL** | **105.39 G** | | **19.761 G** | |

**Why the inversion happens.** REAP+NVFP4 was applied to the *experts*, which are 83% of the
bytes on disk but only 24% of the bytes per token, because 8 of 144 are read. The 34 KDA layers
carry four bf16 `[8192, 4096]` matrices each (`q/k/v/o_proj`, 67.1 MB apiece = 268 MB/layer) and
**every one is read on every token**. Quantisation went where the disk was, not where the
bandwidth is.

## 2. The AR wall

Achievable bandwidth is **measured** — `tools/bw_probe.cu`, inherited from
`deepseek-v4-flash-0731-cuda` where it was run on this same Thor: **240 GB/s streaming,
212 GB/s under memory contention**, against 273 GB/s spec.

```
@ 212 GB/s (contended)  : 10.73 tok/s   (93.2 ms/tok)
@ 240 GB/s (achievable) : 12.14 tok/s   (82.3 ms/tok)   <- the operative wall
@ 273 GB/s (spec peak)  : 13.81 tok/s   (72.4 ms/tok)
```

That is the wall **before** any kernel inefficiency. Prior art on this box (`0731`) reached 37%
of achievable bandwidth on a first cut and treats 70–80% as the target for well-written batch-1
decode. So the honest pre-optimisation expectation for a correct-but-naive engine is **4.5–5
tok/s**, and the realistic post-grind target is **8.5–9.7 tok/s**.

## 3. Lever #1: quantise the bf16 dense weights

There are **13.32 GiB of bf16 weights on the AR path** that nothing has touched: KDA
projections, MLA projections, the indexer, `lm_head`, and the three dense MLPs. At NVFP4
(4 bits + one fp8 scale per 16 → 0.5625 B/weight vs 2.0):

```
B_tok     19.761 G -> 9.657 G   (-51.1%)
AR wall     12.14  -> 24.85 tok/s   @ 240 GB/s
resident    98.15  -> 88.57 GiB     (Thor envelope ~117 GiB)
```

**SUPERSEDED BY OPTIMIZATION_LOG #9 — measure before believing this.** The claim below assumed
every phase converts bytes to time at the same rate. Profiled, they do not: the phases this
lever targets (KDA, MLA, `lm_head`) already run at 72-86% of achievable bandwidth, while the
MoE runs at 13% and owns 66.5% of the step. Applied first, this lever is worth **1.17x**;
applied after the MoE kernels are fixed, 1.50x. Fix the MoE first.

~~**This single change is worth more than every kernel optimisation combined**~~, and it also buys
~9.6 GiB of envelope headroom, which is what makes long context and a resident MTP block
affordable. `gemma-cuda-server` cycle 22/23 measured the same lever on `lm_head` alone and got
+6.9% then +26%.

It is not free: KDA states are explicitly noted upstream as "susceptible to rounding errors",
so `q/k/v_proj` feed a recurrence that accumulates. The plan is to quantise in the order
`lm_head` → `o_proj` → `q/k/v_proj`, gating perplexity at each step, and to keep any tensor
that fails its gate in bf16. **Nothing here is assumed to convert; each step is measured.**

## 4. State and cache — both small, both context-flat where it matters

**KDA recurrent state is context-INDEPENDENT:**

```
34 layers x 64 heads x 128 x 128 fp32 = 136.00 MiB
conv windows (kernel 4, k-1=3)        =   9.56 MiB
                                total = 145.56 MiB
```

That total is an exact independent confirmation of the architecture model: llama.cpp's
`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` checkpoint for this model measured **145.56 MiB fixed by
model, independent of context length** (see `glm53-mtp-serving-path`). Two unrelated derivations
agreeing to the byte means the layer geometry here is right.

Read+write per token is 0.305 G = **1.5% of `B_tok`** — cheap. Storing state in bf16 would save
0.7% of `B_tok` and is **not** worth the numerical risk.

**MLA latent KV is tiny** because only 11 of 45 layers are full-attention and MLA stores one
512-wide latent per token (MQA):

| context | latent KV |
|---|---|
| 8 192 | 88 MiB |
| 32 768 | 352 MiB |
| 131 072 | 1 408 MiB |

Long context is essentially free on this model. The envelope constraint is weights, not cache.

## 5. Ranked optimisation targets

| # | target | worth | why |
|---|---|---|---|
| **1** | **NVFP4 the bf16 dense weights** | **-51% `B_tok`** | §3. Bigger than everything else combined. |
| 2 | KDA decode kernel at high BW efficiency | up to 47.4% of the step | one block per head, 64 KiB state, read-once |
| 3 | MoE GEMV with hardware FP4 unpack | 24.1% | `sm_110a` has FP4x2 unpack; see `dspark-decode-gap-research` |
| 4 | `lm_head` (154 880 x 4096) | 6.4% | included in #1; also the easiest to gate |
| 5 | MLA projection GEMVs | 13.1% | #1 in the 0731 model at 41%; only 13% here |
| 6 | HC compose | 0.4% | 0731 found it 9.4% of *time* for 1.2% of bytes — latency, not bytes |

## 6. What this says about speculative decoding

Spec decode was a **net slowdown** on the GGUF build of this model — not because the draft head
was weak (72.26% depth-1 acceptance un-fine-tuned, `glm53-nvfp4-mtp-gate2`) but because
llama.cpp's batch cost is flat below 32 tokens and a depth-K draft makes a K+1 batch
(`llamacpp-small-batch-offload-cliff`).

**The reason that does not have to repeat here is that the batch-cost curve is ours to build.**
At bs=1 decode this model reads 19.76 GB of weights to produce one token. Verifying K+1 tokens
in one batch reads *the same weights once*, plus K extra rows of activations — the marginal
bytes for the second through (K+1)-th token are the MoE experts they newly touch, and nothing
else. That is the whole thesis of the server, and §1's finding sharpens it: 76% of `B_tok` is
dense weights that are **completely shared** across a verify batch.

So the verify batch must be built so that batch cost is flat in K. That is a design constraint
on the kernels from day one, not an optimisation to retrofit — and it is the thing to measure
first, before any draft-head fine-tune.

---

## §4 — the batch-cost curve, and what it says about speculation

Speculation lost on the GGUF build for a reason that had nothing to do with the draft head: llama.cpp's
batch cost is flat below 32 tokens (`llamacpp-small-batch-offload-cliff`), so a depth-K draft always
landed on the wrong side of the cliff and a 72%-accurate head still made things slower. Here the
kernels are ours, so the curve is ours to set — and it is worth writing down what it *has* to look
like before building anything against it.

A K-wide forward reads the non-expert weights **once**, and reads whichever routed experts the K
tokens between them select. So the cost splits cleanly:

| | per forward |
|---|---|
| everything except routed experts | **15.005 G**, independent of K |
| each distinct routed expert touched | 0.5945 G |

With independent routing (8 of 144 per token), `E[distinct] = 144·(1 − (1 − 8/144)^K)`:

| K | E[distinct experts] | cost | vs one AR step | ceiling if all K accept |
|---|---|---|---|---|
| 1 | 8.0 | 19.76 G | 1.000 | 1.00x |
| 2 | 15.6 | 24.25 G | 1.227 | 1.63x |
| 3 | 22.7 | 28.50 G | 1.442 | 2.08x |
| 4 | 29.4 | 32.50 G | 1.645 | 2.43x |
| 6 | 41.8 | 39.86 G | 2.017 | 2.97x |
| 8 | 52.8 | 46.42 G | 2.349 | 3.41x |

Folding in the acceptance already measured on this checkpoint — 72.26% at depth 1, un-fine-tuned
(`glm53-nvfp4-mtp-gate2`) — gives the predicted end-to-end speedup:

| draft depth | verify width | tokens/iteration | cost | **speedup** |
|---|---|---|---|---|
| 1 | 2 | 1.72 | 1.227 | 1.40x |
| 2 | 3 | 2.24 | 1.442 | 1.55x |
| **3** | **4** | **2.62** | **1.645** | **1.59x** |
| 4 | 5 | 2.89 | 1.836 | 1.57x |

**Depth 3 is the optimum and it is a flat one** — depths 2 to 4 are all within 3% of each other, so
the exact choice does not much matter and tuning it is not worth a session.

Three things this table is honest about:

1. **It excludes the draft head's own cost.** The MTP block at layer 45 has to run once per drafted
   token. That has since been costed in SPEC_DECODE.md and it is not small: one MLA+MoE layer
   (0.371 G) plus `eh_proj` (0.067 G) plus **`lm_head` at 1.269 G**, which dominates. Folding it in
   pulls the 1.59x at depth 3 down to **1.38x**, and moves the optimum to depth 2. Quantising the
   draft `lm_head` to NVFP4 — system-level lossless, because a draft error costs a rejection and
   not an output error — brings it back to **1.49x**.
2. **Independent routing is the PESSIMISTIC assumption.** Adjacent tokens in a real sequence have
   correlated hidden states, so they share more experts than chance, `E[distinct]` is lower, and the
   true curve is cheaper than this one. Measuring the actual overlap needs the full 45-layer load.
3. **It assumes the batched GEMM holds the same bandwidth as the gemv.** At K ≤ 8 the operation is
   still firmly memory-bound, so it should; that is a measurement, not a proof.

The conclusion that matters for the build order: **a multi-token forward is the prerequisite, not
the speculation logic.** Verifying K drafted tokens one at a time costs exactly K AR steps and wins
nothing no matter how good the head is. The same kernel also fixes prefill, which is today the
server's single largest cost — sequential prefill pays the full 19.76 G for every prompt token,
where a K-wide chunk amortises 15.005 G of it across K.
