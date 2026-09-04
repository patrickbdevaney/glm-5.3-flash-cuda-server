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
| dense mlp (layers 0–2) | 0.60 G | 0.6% | 0.604 G | 3.1% |
| moe router | 0.35 G | 0.3% | 0.352 G | 1.8% |
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

**This single change is worth more than every kernel optimisation combined**, and it also buys
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
