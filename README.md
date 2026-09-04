# glm-5.3-flash-cuda-server

A pure-CUDA inference server for **GLM-5.3-Flash-REAP50-NVFP4** on Jetson AGX Thor.
No Python on the request path. Built for AR decode and MTP speculative decode.

## Why

The draft head is already good — **72.26% depth-1 acceptance un-fine-tuned**. Speculation still
*lost* on the GGUF build, because llama.cpp's batch cost is flat below 32 tokens, so a depth-K
draft always lands on the slow side of the cliff. **The batch-cost curve is what decides whether
speculation pays, and here it is ours to build.**

## What the checkpoint actually is

45 layers: **34 KDA linear-attention** + **11 MLA/DSA full-attention** (at 3, 7, 11, … 43), plus
one MTP block at index 45. 144 REAP-pruned experts, 8 per token, 1 shared. Manifold-constrained
hyper-connections carry **four** residual streams, not one. Attention is **pure NoPE** — no rotary
anywhere in the main path. A 24-block vision tower ships and is preserved.

## The finding that set the agenda

> **KDA linear attention is 47.4% of per-token bandwidth — nearly twice the entire MoE.**

REAP+NVFP4 quantised the *experts*: 83% of the bytes on disk, but only 24% of the bytes per token,
because 8 of 144 are read. Meanwhile **13.32 GiB of bf16 dense weights sit on the AR path
untouched** and account for 76% of `B_tok`. Quantising those halves `B_tok` (19.76 → 9.66 G) and
doubles the AR wall (12.1 → 24.9 tok/s at the 240 GB/s measured on this box). See `ROOFLINE.md`.

## State

| subsystem | % of `B_tok` | status |
|---|---|---|
| KDA linear attention | 47.4% | **gated cos 1.000000000** |
| MoE (routed + shared + router) | 28.9% | **gated cos 1.000000000** |
| MLA full attention | 13.1% | **gated cos 1.000000000** |
| dense MLP + mHC + norms | 3.5% | **gated cos 1.000000000** |
| lm_head | 6.4% | wired (gemv), gated via the engine |
| DSA indexer | 0.8% | not built — **provably unnecessary below 2048 context** |

Everything is gated against `transformers` **on real checkpoint weights**, never on synthetic
fixtures. That is not pedantry: both "misaligned address" faults in `OPTIMIZATION_LOG` #2 were
invisible to `cudaMalloc`-backed buffers and fatal on the real model.

## Layout

```
ROOFLINE.md          per-token bytes, measured; where the levers are
ARCH_DELTA.md        what ports from ~/deepseek-v4-flash-0731-cuda, what is new
OPTIMIZATION_LOG.md  every measurement, and the ones that were nearly logged wrong
CLAUDE.md            operating rules (detachment, gate discipline, hard constraints)
STATUS.md            where the build is
include/  kernels/  src/     the engine
ref/                 PyTorch oracles (run from ~/glm-5.3-reap's venv)
tests/               gates — the deliverable as much as the kernels are
tools/               roofline, bandwidth probe, benches
```

## Build and gate

```bash
bash scripts/build.sh
cd ~/glm-5.3-reap
for g in kda moe layer mla stack; do
  ./.venv/bin/python ~/glm-5.3-flash-cuda-server/ref/gen_$g.py
done
cd ~/glm-5.3-flash-cuda-server && bash scripts/gate.sh
```

## Next

1. Engine end-to-end on all 45 layers (needs ~98 GiB free; blocked while an unattended stage holds
   the box).
2. Tokenizer + HTTP/OpenAI server — ports from `~/deepseek-v4-flash-0731-cuda`.
3. DSA indexer, to go past 2048 context.
4. **Measure the batch-cost curve, then decide on speculation** — in that order, because the
   reverse order is what wasted the effort on GGUF.
