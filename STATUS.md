# STATUS

**Goal.** A pure-CUDA inference server for GLM-5.3-Flash-REAP50-NVFP4 on Jetson AGX Thor,
engineered for AR decode and MTP speculative decode, with no Python on the request path.

**Why it exists.** The draft head is already good — 72.26% depth-1 acceptance un-fine-tuned
(`glm53-nvfp4-mtp-gate2`). Speculation still *lost* on the GGUF build because llama.cpp's batch
cost is flat below 32 tokens, so a depth-K draft always lands on the slow side of the cliff. The
batch-cost curve is the thing that decides whether any of this pays, and here it is ours to build.

## Where we are

| stage | state |
|---|---|
| architecture mapped from the checkpoint | **done** — `ARCH_DELTA.md` |
| roofline measured | **done** — `ROOFLINE.md`, `tools/roofline.py` |
| port verdict per subsystem | **done** — one NEW (KDA), three RETARGET, rest PORT |
| **KDA** (47.4% of `B_tok`) | **gated cosine-1.0**, 15/15, both dtypes — `tests/gate_kda.cu` |
| **MoE** (28.9%) | **gated cosine-1.0**, real packed NVFP4 from the shards — `tests/gate_moe.cu` |
| **mHC + norms + dense MLP** | **gated cosine-1.0** — inside `tests/gate_layer.cu` |
| **complete decoder layer 0** | **gated cosine-1.0**, 12/12 end-to-end — `tests/gate_layer.cu` |
| **MLA full attention** (13.1%) | **gated cosine-1.0**, 4/4 — `tests/gate_mla.cu` |
| DSA indexer (0.8%) | not started — **only needed above 2048 context** (verified) |
| **engine** (45 layers + lm_head) | **written and gated** — `tests/gate_stack.cu`, 4 decode steps cos 1.000000000 across 3 layers |
| **multi-token forward** | **gated BIT-EXACT** vs the sequential path, 13/13 — `tests/gate_batch.cu`, at 4 layers so MLA and MoE run inside the engine loop |
| **speculative rollback** | **gated** — accept 0/1/3/5 of a 5-wide snapshot then continue, all equal to a run that never speculated |
| **tokenizer** | **gated id-exact vs HF**, 170/170 — `tests/gate_tokenizer.cpp` |
| **chat encoder** | **gated byte-exact vs HF's own Jinja**, 42/42 — `tests/gate_encoding.cpp` |
| **sampler / stream / API** | **gated**, 44 checks, no GPU needed |
| **HTTP server** | **running** — OpenAI chat + completions, SSE, tools, prefix reuse; 19/19 live smoke |
| MTP + speculative decode | designed and costed, not built — `SPEC_DECODE.md` |

**92.9% of per-token bandwidth is now implemented and gated against `transformers` on real
checkpoint weights.** Everything remaining on the AR path is `lm_head` (6.4%), which is a gemv
that already exists and needs wiring, and the DSA indexer, which does not affect results below
2048 tokens of context.

## The two findings that set the agenda

1. **KDA linear attention is 47.4% of per-token bandwidth** — nearly twice the entire MoE.
   REAP+NVFP4 quantised the experts (83% of disk, 24% of per-token bytes) and left 13.32 GiB of
   bf16 dense weights untouched (76% of per-token bytes).
2. **Quantising those to NVFP4 halves `B_tok`** (19.76 → 9.66 G) and doubles the AR wall
   (12.1 → 24.9 tok/s at the measured 240 GB/s). Worth more than every kernel optimisation
   combined — and it must be gated tensor-family by tensor-family, because KDA feeds a
   recurrence that accumulates rounding error.

## Next

1. **Run the full 45-layer engine.** Needs ~98 GiB free; an unattended trace extraction holds ~77
   GiB and finishes around chunk 80 of 80. Everything below marked *blocked* waits on this.
2. **Widen `gate_stack` to layer 3** against the PyTorch oracle — *blocked*, the oracle needs ~20
   GiB for layer 3's experts. Note `gate_batch` now runs MLA and MoE inside the engine loop, so
   what remains uncovered is their absolute wiring, not their consistency.
3. **Speculative decode**, per `SPEC_DECODE.md`. Step 1 there — rolling back the KDA recurrence —
   is **done and gated**. Steps 2-4 (load layer 45, the draft step, the draft-verify loop) need the
   full model and are blocked.
4. **Resolve pre-norm vs post-norm for the MTP block's `h_prev`** before any head fine-tuning. Two
   lines, and getting it wrong would be baked into the fine-tune (`SPEC_DECODE.md`).
5. **DSA indexer**, to go past 2048 context. k-pooling (`kpool` 4 + compress gate + APE) is new;
   `index_topk` is 2048, not 512.
6. **Batch the routed experts** — worth 4.7% at verify widths, 1.9x for wide prefill chunks. Do it
   for prefill, not for speculation.

**Not yet measured:** `tools/bench_batch` has never had an uncontended run. Two attempts landed
while the box was at 97% GPU, and the first reported K=2 as faster than K=1 in absolute
ms/forward — arithmetically impossible. The methodology is fixed (round-robin widths, minimum over
32 reps, an in-process bandwidth probe) but the number is still owed, and `ROOFLINE.md` §4 remains
an analytic prediction rather than a measurement. See `CLAUDE.md` §6.

## Two things the build has already changed

- **`B_tok` is the lever, not the kernels.** KDA runs at the machine's achievable bandwidth
  already (`OPTIMIZATION_LOG` #1). The 51% `B_tok` cut from quantising the bf16 dense weights
  (`ROOFLINE` §3) is worth more than any kernel work left on the table.
- **Gate on real weights or do not bother.** Both "misaligned address" faults (`OPTIMIZATION_LOG`
  #2) were invisible to synthetic fixtures and fatal on the checkpoint.
