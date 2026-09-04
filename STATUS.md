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
| tokenizer / HTTP server | not started (ports from `0731`) |
| MTP + speculative decode | not started |

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

1. **Widen the stack gate to layer 3**, which covers MLA and MoE *inside the engine loop* — both
   are gated standalone but their wiring is not. Blocked only on memory: the oracle needs ~20 GiB
   for layer 3's experts and an unattended stage currently holds 64 GiB.
2. **Run the full 45-layer engine.** Needs ~98 GiB free; same blocker.
3. **Server**: tokenizer (GLM vocab 154 880, three EOS ids), HTTP/OpenAI, SSE — ports from `0731`.
4. **DSA indexer**, to go past 2048 context. k-pooling (`kpool` 4 + compress gate + APE) is new;
   `index_topk` is 2048, not 512.
5. **Then, and only then, speculation** — with the batch-cost curve measured first, because that
   is what decided it on GGUF.

## Two things the build has already changed

- **`B_tok` is the lever, not the kernels.** KDA runs at the machine's achievable bandwidth
  already (`OPTIMIZATION_LOG` #1). The 51% `B_tok` cut from quantising the bf16 dense weights
  (`ROOFLINE` §3) is worth more than any kernel work left on the table.
- **Gate on real weights or do not bother.** Both "misaligned address" faults (`OPTIMIZATION_LOG`
  #2) were invisible to synthetic fixtures and fatal on the checkpoint.
