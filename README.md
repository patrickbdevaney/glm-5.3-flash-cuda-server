# glm-5.3-flash-cuda-server

A pure-CUDA inference server for **GLM-5.3-Flash-REAP50-NVFP4** on Jetson AGX Thor.
No Python on the request path. Built for AR decode and MTP speculative decode.

## Why, and how that changed

This began as a speculative-decode engine: the native MTP draft head is good (**72.3% acceptance,
un-fine-tuned**), and speculation lost on the GGUF build only because llama.cpp's batch cost is
flat below 32 tokens. The batch curve here is ours to build, so the plan was to build a better one.

**That plan was measured and it failed.** The curve was built, and speculation still loses — see
"Speculation: closed" below. What the engine turned out to be good for is plain autoregressive
decode, which it now does at 1.42x its own starting point. The MTP scaffolding
(`forward_batch`, state-slot rollback, `SPEC_DECODE.md`) is built, gated and unused.

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

## State — measured, on this box

| | |
|---|---|
| decode | **11.69 tok/s** (85.5 ms/step), 50% of a 23.5 tok/s roofline |
| prefill | **47.8 ms/tok** at chunk 32 |
| `B_tok` | **9.762 G/token** (19.761 before the NVFP4 dense overlay) |
| perplexity | **4.4996** vs 4.4305 bf16 — the overlay costs **+1.56%** |

All kernels are gated against `transformers` **on real checkpoint weights**, never on synthetic
fixtures. That is not pedantry: both "misaligned address" faults in `OPTIMIZATION_LOG` #2 were
invisible to `cudaMalloc`-backed buffers and fatal on the real model. `forward_batch` at M=1 is
asserted **bit-identical** to sequential `decode()`, not merely close.

### The NVFP4 dense overlay

A 3.70 GiB side-car (not a checkpoint rewrite) that converts the bf16 dense weights on the AR path
to NVFP4. Halves `B_tok`, buys **1.42x** on decode. It is not free, and the honest cost is not the
cosine:

| | |
|---|---|
| perplexity | +1.56% (47,195 tokens, identical ids both conditions) |
| top-1 agreement | 90.101% |
| **confident** disagreements (reference NLL < 0.5) | **0.138%** of tokens |

The flips sit where the model was already unsure — median reference NLL 2.2054 on disagreements
against 0.3547 on agreements. Per-tensor cosine was 0.9950-0.9972 and implied a much smaller
effect than the measurement found; **quote the perplexity, not the cosine.** `GLM5_DENSE_NVFP4=0`
reverts it with no file touched.

### Speculation: closed

The batch curve was built and then measured, in that order, which is the one thing the GGUF effort
got wrong. A width-2 verify costs **2.164** AR steps, so at 72.3% acceptance every draft depth is a
net loss (best 0.95x at depth 2), and the payoff is bounded at **1.27x even for a perfect drafter**.

The cause is structural, not an implementation defect: with 144 experts at top-8, M tokens touch
`144*(1-(1-8/144)^M)` distinct experts — still 7.4 per token at M=4 — so expert traffic is
near-linear in M exactly where speculation needs it flat. Attention batches fine (0.46-0.54 at
width 4); `ffn:moe` does not (0.82) and is ~45% of the step. **Fine-grained MoE is structurally
hostile to speculative decoding at small draft depths.** No draft head fixes this; do not fine-tune
one for this engine.

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

## Server

OpenAI-compatible, no Python on the request path.

| endpoint | |
|---|---|
| `POST /v1/chat/completions` | streaming (SSE), reasoning/`<think>` split out, tool calls, **images** |
| `POST /v1/completions` | raw prompt, no chat template |
| `POST /v1/embeddings` | last-token pooled hidden, L2-normalized, 4096-d |
| `POST /tokenize` / `/detokenize` | with `with_pieces` for per-token inspection |
| `GET /props` | context, vocab, resident GiB, capabilities |
| `GET /health` `/metrics` `/v1/models` `/` | Prometheus metrics; web UI at `/` |

Sampling: `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, `max_tokens`, `reasoning_effort`,
`logprobs`/`top_logprobs`. Prefix caching reuses the resident state whenever a request extends it,
which is the ordinary multi-turn case (`usage.prompt_tokens_details.cached_tokens` reports it).

`--ctx N` sets the KV/context length. Only 11 of 45 layers have a KV cache: the 34 KDA layers cost
a **fixed** 145.56 MiB regardless of N, and the latent cache is `11 * N * 512 * 4` bytes, so
context is far cheaper here than in a dense model of the same size.

**Images**: `data:` URIs only. A plain http(s) URL is refused on purpose — making an inference
server fetch arbitrary URLs for a caller is SSRF.

## Next

1. **Vision.** The checkpoint ships a 24-block vision tower (`model.visual.*`, patch embed, merger,
   downsample) and this server is **text-only** — there is no vision code in it at all. The GGUF
   build has it; this does not. Largest capability gap.
2. **Concurrent-request batching.** The server serialises on a global lock. The measured curve
   gives **1.69x aggregate throughput at width 16**, and the memory is affordable precisely because
   only 11 of 45 layers have a KV cache: per stream is 145.56 MiB of fixed KDA state plus
   `11 * ctx * 512 * 4` of latent cache — 322 MiB at 8k, so 16 streams cost 5 GiB.
3. **Long-context correctness above 2051.** The DSA sparse path is gated bit-exact *below*
   `DENSE_CTX_LIMIT`; above it there is no dense answer to compare against. `tools/perplexity.cu
   --concat` reports perplexity by position band, which is the available check.
4. Decode is at 50% of roofline and the remaining gap is spread thin — `ffn:moe` at 66% of
   bandwidth is the largest single block. Row tiling was swept (`NVFP4_R`): 5 is optimal, higher is
   up to 3x worse, because `x` is 16 KB and L2-resident.
