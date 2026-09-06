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
| **DSA indexer** (0.8%) | **gated**, 15/15 vs the real module — `tests/gate_indexer.cu` |
| **sparse MLA** | **gated bit-exact vs dense** for all 2051 in-limit steps — `tests/gate_mla_sparse.cu` |
| **engine** (45 layers + lm_head) | **written and gated** — `tests/gate_stack.cu`, 4 decode steps cos 1.000000000 across 3 layers |
| **multi-token forward** | **gated BIT-EXACT** vs the sequential path, 13/13 — `tests/gate_batch.cu`, at 4 layers so MLA and MoE run inside the engine loop |
| **speculative rollback** | **gated** — accept 0/1/3/5 of a 5-wide snapshot then continue, all equal to a run that never speculated |
| **tokenizer** | **gated id-exact vs HF**, 170/170 — `tests/gate_tokenizer.cpp` |
| **chat encoder** | **gated byte-exact vs HF's own Jinja**, 42/42 — `tests/gate_encoding.cpp` |
| **sampler / stream / API** | **gated**, 44 checks, no GPU needed |
| **HTTP server** | **running** — OpenAI chat + completions, SSE, tools, prefix reuse; 19/19 live smoke |
| MTP + speculative decode | designed and costed, not built — `SPEC_DECODE.md` |
| **full 45-layer engine** | **RUNNING** — 57,604 tensors, 98.49 GiB resident, loads in ~80 s; serves coherent text |

**The context limit is gone.** A 4,073-token prompt now serves end to end; the engine ran dense
below 2051 and switched to the indexer above it. **92.9% of per-token bandwidth is implemented and gated against `transformers` on real
checkpoint weights.** Everything remaining on the AR path is `lm_head` (6.4%), which is a gemv
that already exists and needs wiring, and the DSA indexer, which does not affect results below
2048 tokens of context.

## The full model runs

`scripts/run_full.sh --seqmax 8192` brings up all 45 layers: **57,604 tensors, 98.49 GiB
resident**, loaded in ~80 s, answering on the OpenAI endpoint with coherent, on-topic text.

**Decode is 3.4 tok/s against a roofline of 11.7** (231.5 GB/s measured with the model resident,
divided by `B_tok` = 19.761 G). So 29% of roofline — the same shortfall the DSpark CUDA server hit
(7.89 tok/s, ~25%), and the same diagnosis: the gap is *kernel efficiency, not algorithm*. The
recorded top lever is a hardware-unpack FP4 MoE GEMV. Note this is measured BEFORE the -51% `B_tok`
win from NVFP4-ing the bf16 dense weights (§3), which moves the roofline, not the efficiency.

`include/dprof.h` is ported but **not wired into any kernel here** — attributing the 3.4-vs-11.7 gap
to a sub-op needs those marks placed first. That is the next measurement, and it is cheap.

### Two operational rules this run established

**Never probe for the cudaMalloc ceiling.** On Thor `cudaMalloc` draws from the same DRAM as
everything else, so over-allocation is a *global OOM kill*, not an error return. An
allocate-until-failure probe took down the Claude Code session, `gnome-software` and
`update-manager` before it died itself. Size the load by summing safetensors headers instead.

**Reclaim the driver page pool before a big load.** After a large CUDA process exits, its memory
does not come back on its own: `free` reports it *used* while `AnonPages`, `Cached`, `Slab` and
nvmap's own accounting are all tiny. `sync; echo 3 > /proc/sys/vm/drop_caches` recovered 96 GiB
(22 GiB available -> 118). `scripts/run_full.sh` also raises its own `oom_score_adj` to 1000, so a
bad sizing costs a restart rather than the session.

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

## Decode, after the MoE kernel rewrite (OPTIMIZATION_LOG #10)

| | before | after | |
|---|---|---|---|
| AR decode, 45 layers, 512 ctx | 253.11 ms/step | **119.63 ms/step** | **2.12x** |
| | 3.94 tok/s | **8.36 tok/s** | |
| % of achievable bandwidth | 32% | **67%** | 0731 engine is at 68% |
| `ffn:moe` | 4038 ms / 13% BW | **857 ms / 82% BW** | 4.71x |

Every gate green throughout: `gate_kda` 15/15, `gate_moe` 3/3, `gate_layer` 12/12, `gate_mla` 4/4,
`gate_indexer` 15/15, `gate_mla_sparse` 3/3, `gate_stack` 4/4, `gate_batch` 13/13.

**Kernel efficiency is done.** Every remaining phase runs at 190-200 GB/s against a machine that
streams 235-247. The only lever left is `B_tok` itself — ROOFLINE §3, the NVFP4 conversion of the
13.32 GiB of bf16 dense weights, now correctly ordered *after* the MoE and worth ~1.5x
(~12.8 tok/s). It is a checkpoint change, not a kernel change, and it has not been started.

## Decode and prefill, after the NVFP4 dense overlay (OPTIMIZATION_LOG #11)

| | before | after | |
|---|---|---|---|
| AR decode, 45 layers | 119.63 ms/step | **84.32 ms/step** | **1.42x** |
| | 8.36 tok/s | **11.86 tok/s** | |
| prefill @ chunk 32 | 84.0 ms/tok | **51.7 ms/tok** | **1.63x** |
| `B_tok` | 19.761 G | **9.762 G** | -50.6% |
| resident | 98.15 GiB | 101.85 GiB | overlay is additive; the bf16 copies stay loaded |

All gates green, `gate_batch`'s M=1 bit-identity included.

**The finding that matters more than the numbers: activation traffic, not weight traffic, was the
binding term in every batch-1 and batched gemv in this engine.** A gemv reads 4 bytes of x per
weight — 2 bytes of x per byte of bf16 weight, but 7.1 per byte of NVFP4. That is why halving
`B_tok` first bought 1.4%, why the batched path cost exactly M times a gemv, and why prefill got
*worse* with wider chunks. Both are fixed by reusing x across output rows.

### What is left, in order

1. ~~**Expert-gathering in the MoE.**~~ **DONE** — OPTIMIZATION_LOG #12. Prefill now falls with
   width instead of rising, and the default chunk is 32.
   ~~The next laggard is `attn:mla`.~~ **ALSO DONE** — OPTIMIZATION_LOG #13 (sub-phase marks) and
   #14 (batching `absorb_q`/`expand_v`, which each streamed the whole of `kv_b` per token).
   `attn:mla` is 20.0% -> 15.3% of prefill at 38% -> 54% of achievable bandwidth; prefill at
   chunk 32 is **47.79 ms/tok**. The remaining MLA mass is `mla:sdpa`, which is CACHE traffic,
   not weight traffic — a different problem from every lever in the log so far.
2. **The NVFP4 accuracy decision.** cos 0.9972 over three KDA layers against the PyTorch oracle;
   uniform rel 0.088-0.100 per tensor, no family worse than another. Reverting a family is a
   re-run of `tools/requant_dense_nvfp4.py --families`; `GLM5_DENSE_NVFP4=0` reverts all of it
   with no file touched.
3. **Speculative decode** is unblocked in principle now that batch cost is sublinear in K, but
   ROOFLINE §4's curve should be re-measured against the fixed kernel before any head fine-tune.
4. Still owed from before: long-context correctness above 2051 on the full model, and the
   unprofiled >34-minute prefill (which was at least partly this — prefill was ~82 ms/tok, so
   3,400 tokens was ~4.6 minutes of kernel time, not 34).
