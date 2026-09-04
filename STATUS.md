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
| kernels | not started |
| engine | not started |
| server | not started |

## The two findings that set the agenda

1. **KDA linear attention is 47.4% of per-token bandwidth** — nearly twice the entire MoE.
   REAP+NVFP4 quantised the experts (83% of disk, 24% of per-token bytes) and left 13.32 GiB of
   bf16 dense weights untouched (76% of per-token bytes).
2. **Quantising those to NVFP4 halves `B_tok`** (19.76 → 9.66 G) and doubles the AR wall
   (12.1 → 24.9 tok/s at the measured 240 GB/s). Worth more than every kernel optimisation
   combined — and it must be gated tensor-family by tensor-family, because KDA feeds a
   recurrence that accumulates rounding error.

## Next

Kernels bottom-up, each gated on real weights: elementwise → NVFP4 GEMV → KDA → HC → MoE →
MLA/DSA → engine → server. KDA first: it is the new surface and the largest single cost.
