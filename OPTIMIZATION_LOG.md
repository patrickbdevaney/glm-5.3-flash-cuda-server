# OPTIMIZATION_LOG

Measurements only. Each entry says what was measured, under what conditions, and what it changed.

---

## #1 — KDA decode step, first cut (2026-09-04)

**Correctness first.** `tests/gate_kda.cu`, 15/15 checks against `transformers` on real layer-0
weights, **cosine 1.000000000 on every stage**, including the full 1 048 576-element recurrent
state tensor. Both dtypes gated: fp32 and the production bf16 path (max_rel 2.99e-6).

Stages gated separately so a regression names a stage, not a layer: `qkv+conv+silu`,
`conv_state`, `forget gate g`, `beta`, `output gate`, `l2norm q`, `l2norm k`, `recurrence out`,
`state S_out`, `gated o_norm`, `o_proj`, plus the fused `kda_decode_step` end-to-end.

**Bandwidth.** `tools/bench_kda.cu`, byte model 284.66 MB/layer (exact — counts only what the
kernels are obliged to move).

| | |
|---|---|
| median | 3.55 ms/layer (best 2.55, worst 4.39) |
| achieved | 80.2 GB/s |
| **achievable at that instant** | **76.7 GB/s** |
| efficiency | **104.5% of concurrent achievable** (33.4% of the 240 GB/s idle figure) |

### The measurement that nearly went in the log wrong

The first run reported **29.3% of achievable** — which reads as "this kernel is 3x off and needs a
grind". It was measured while an unattended `llama-embedding` trace-extraction stage was holding
the GPU at 96%. Running `tools/bw_probe.cu` under the same contention: a **pure streaming read
reached 82.8 GB/s**, not 240. The kernel was never 29% of anything; it was at the machine's
current ceiling.

`bench_kda` now measures a streaming probe **in the same process, immediately before the timing
loop**, and reports efficiency against that. Efficiency above 100% is expected and means the byte
model slightly overcounts: it charges full HBM traffic for small tensors (`conv1d` 393 KB,
`dt_bias`, `A_log`, `x`) that in fact stay resident in L2 across iterations.

**Conclusion: the KDA recurrence is not where the time is hiding.** One block per head, 64 KiB of
state staged in dynamic shared memory so it crosses HBM exactly once each way, is already at the
bandwidth ceiling. Optimisation effort should go to `B_tok` (ROOFLINE §3), not to this kernel.

**Owed:** a clean re-measure on an idle box. The absolute ms figures above are not usable for
planning; the efficiency ratio is.

---

## #2 — MoE block gated on real NVFP4 experts (2026-09-04)

`tests/gate_moe.cu`, layer 3, reading packed experts **straight out of the checkpoint shards**
(3.30 GiB resident) rather than from an fp32 fixture — so the 4-bit unpack, the fp8-e4m3 scales,
the per-tensor global scale and the loader are all on the hook.

| check | result |
|---|---|
| routing | **PASS** — same 8 experts {2, 13, 36, 38, 43, 76, 84, 134}, max weight delta 5.96e-08 |
| router logits | **PASS** cos 1.000000000, max_rel 1.62e-06 |
| moe output y | **PASS** cos 1.000000000, max_rel 3.72e-06 |

With KDA (#1) that is **76.3% of `B_tok` gated against `transformers` on real weights.**

### The alignment trap, twice

Both faults were "misaligned address", both were vector loads on checkpoint tensors, and both
would have passed any gate built on synthetic buffers.

**safetensors aligns tensors to 4 bytes, not 8 or 16.** Measured on this checkpoint: of the
`weight_packed` tensors in a shard, **777 of 1671 sit at offset 4 mod 8** (e.g. layer 45 expert 0
`down_proj` at blob offset 2 691 195 540). Anything from `cudaMalloc` is 256-byte aligned, so a
kernel that works perfectly on oracle fixtures dies on the real model.

1. `nvfp4_row_dot` read each 16-element group as one `uint2` (8 bytes). Now two `uint32` loads —
   same bytes, same coalescing, legal at 4-byte alignment.
2. `gemv` read weights as `float4` (16 bytes). Now `gemv()` measures alignment at launch and
   dispatches to a scalar variant when the vector path would be illegal. The check tests the row
   **stride** as well as the base: a 16-aligned base with an odd stride leaves row 1 misaligned
   even though row 0 is fine.

`nvfp4_check_align()` asserts the 4-byte floor at load time, so a checkpoint that breaks even that
is caught at startup rather than mid-request.

---

## #3 — MLA gated, and the DSA indexer proved unnecessary below 2048 context (2026-09-04)

`tests/gate_mla.cu`, layer 3, weights from the checkpoint: **4/4 cos 1.000000000**
(`q_resid`, `q`, the stored latent, and the attention output).

### Two findings that shrink the remaining work

**1. MLA here is pure NoPE.** `qk_rope_head_dim = 0` and `mla_use_nope = true`, so there is no
rotary embedding anywhere in the main attention path — no YaRN, no rope cache, no interleave.
Position reaches these 11 layers only through the 34 KDA layers below them. DeepSeek-V4's
`yarn.h` and the rope half of its MLA kernel are not needed at all.

**2. Below 2048 tokens of context the DSA indexer is a no-op**, and this is *verified*, not
assumed. The indexer pools keys in groups of `index_kpool` = 4 and selects
`min(index_topk / index_kpool, n_pools)` = `min(512, n_pools)`. At `n_pools <= 512` — i.e.
context <= 2048 — every pool is selectable, and `index_kpool_always_select_tail` appends the
trailing incomplete pool. `ref/gen_mla.py` calls the **real** `get_pooled_states` /
`get_visible_tokens` and checks the covered set against the full visible set:

    DSA at T=38: 9 pools, select_k=9, covers all 38 positions: True

So dense causal MLA is **exact**, not an approximation, up to 2048 context. The indexer (0.8% of
`B_tok`, and the most intricate kernel left) is only needed to go beyond that — which makes a
correct end-to-end engine reachable without it.

### The absorbed form

Decode folds `W_k` into the query rather than expanding the latent per head:

    qa[h]   = W_k[h]^T q[h]            [512]     kv_b read once
    s[h][t] = qa[h] . C[t] * scaling             attention runs on the latent itself
    ctx[h]  = sum_t a[h][t] C[t]       [512]
    o[h]    = W_v[h] ctx[h]            [256]     expand once, at the end

Algebraically identical — the oracle measures the difference at **7.45e-07** — and it is why only
the 512-wide latent is cached: **88 MiB at 8k context across all 11 layers**, against 1408 MiB for
the expanded form.

### Kernel layout notes

`k_scores` uses one block per tile of 8 cached tokens and loops all 64 heads, so **the latent
cache is read once per block**. The obvious layout — one block per (head, token) — re-reads it 64
times, which at 8k context is 268 MB per layer per token instead of 16 MB. `k_context` uses
head-groups of 16 for the same reason (4x amplification instead of 64x). Neither is measured yet;
both are structural choices made to avoid a known-bad access pattern, and the ladder stays open.

---

## #4 — The engine runs, and its wiring gates (2026-09-04)

`tests/gate_stack.cu` drives the **Engine itself** — not a kernel — over four successive decode
steps and compares the four mHC residual streams against `transformers`:

```
engine: 81 tensors, 10.26 GiB resident, 3 layers, max_ctx 2048
engine: 3 KDA layers (12.84 MiB state, context-independent), 0 full-attn layers
  step 0 tok  7321  PASS  cos 1.000000000  max_rel 1.694e-06
  step 1 tok 32959  PASS  cos 1.000000000  max_rel 1.710e-06
  step 2 tok 10320  PASS  cos 1.000000000  max_rel 1.847e-06
  step 3 tok 48987  PASS  cos 1.000000000  max_rel 1.384e-06
```

Four steps matter more than one: step 0 would pass even if every KDA layer shared a single
recurrent state, or if the conv window never advanced. Steps 1–3 only stay at cosine 1.0 if each
layer keeps its own state and advances it correctly.

**What this does NOT yet cover, and it should be said plainly:** layers 0–2 are all KDA with a
dense MLP, so this gate exercises neither MLA nor MoE *inside the engine* — both are gated
standalone (#2, #3) but their wiring into the layer loop is not. Extending the stack gate to layer
3 covers both. It is not run yet because the oracle must materialise layer 3's 144 NVFP4 experts
as fp32 (~20 GiB) and the box currently has 38 GiB available with an unattended trace-extraction
stage holding 64 GiB. Running it now risks OOM-killing a job that has been going for hours, which
is not a trade worth making for a gate that can run in an hour.

**Also owed:** the full 45-layer load (98 GiB) cannot be attempted until that stage finishes.

---

## #5 — The 51% `B_tok` lever converts (2026-09-04)

`tools/dense_nvfp4_probe.py` round-trips layer 0's bf16 dense weights through the NVFP4 grid the
checkpoint already uses (per-group-of-16 fp8-e4m3 scale, one fp32 global scale, e2m1 values) and
measures what it does to the **output**, not just to the weights.

```
tensor                      rel err          cos   MB saved/tok
kda q_proj                  0.09516    0.9954803         48.2
kda k_proj                  0.09584    0.9954364         48.2
kda v_proj                  0.09387    0.9955958         48.2
kda o_proj                  0.09496    0.9955014         48.2
kda f_b / g_b               0.0951     0.99547            1.5 each
dense gate / up / down      0.093      0.99566           72.4 each

layer output   rel err 0.00127   cos 0.9999992
KDA state      rel err 0.06819   cos 0.9976720   (after 64 tokens of accumulation)
```

**Two things this settles.**

1. **Error cancels, hard.** Individual weights lose ~9.5% relative accuracy, but the *layer
   output* comes back at **cosine 0.9999992** — three orders of magnitude better than the weights
   that produced it. Dot products over 4096 dimensions average independent quantisation noise
   away. This is with **every** dense weight in the layer quantised at once, not one family.

2. **The recurrence is NOT amplifying the error**, which was the specific worry — upstream warns
   KDA states are "susceptible to rounding errors", and `q/k/v_proj` feed a recurrence that
   accumulates over the whole sequence. After 64 tokens the state sits at cosine **0.9977**,
   *better* than the 0.9955 of the weights that drive it, and the gated RMSNorm after the
   recurrence absorbs most of what remains before it reaches the output.

So the plan in `ROOFLINE.md` §3 — quantise `lm_head` → `o_proj` → `q/k/v` and gate at each step —
is more conservative than the evidence requires. The evidence supports quantising the whole dense
set, with the per-step gating kept as a check rather than as a staged retreat.

**Owed before acting on it.** This is one layer and one 64-token prefill. Two things could still
bite: 45 layers of drift compounding, which only a perplexity run over the full stack will show;
and longer contexts, since the state was measured only at 64 tokens and the question is whether
0.9977 is a floor or a slope. Neither can run until the box frees up. **No checkpoint has been
modified and nothing has been written** — the probe is read-only by design.

### A process note worth keeping

Cleaning up the probe, `pkill -f "[d]ense_nvfp4_probe"` killed the calling shell. The bracket
trick prevents a pattern from matching *itself*, but it does nothing when the same command line
also contains the literal string elsewhere — here the path `tools/dense_nvfp4_probe.py`. The rule
is narrower than "use brackets": **do not pkill a pattern that appears anywhere in your own
command line.** Kill by PID, or run the pkill from a command that does not name the target.

---

## #6 — The server, and one bug only a live run could find

The tokenizer, chat encoder, sampler, stream splitter and OpenAI shaping are all gated on the CPU:
**256 checks, no GPU, no weights** beyond the checkpoint's own `tokenizer.json`. 170 of them are
id-exact against HF over adversarial strings, 42 are byte-exact against HF's own Jinja over 19
prompt fixtures. Those two are exact-match, not similarity, because a tokenizer or template that is
99% right is worse than one that is obviously broken: the model keeps producing fluent text from
subtly wrong ids and nothing in the stack reports an error.

Three things that would have been wrong if ported from `0731` by habit:

- **The pre-tokenizer is one stage here, not four.** GLM uses the standard GPT-4 alternation;
  DeepSeek-V4 splits digits and CJK first. And **`ignore_merges` is set**, which DeepSeek's is not:
  a pre-token that is already a vocab entry is emitted whole and BPE never runs on it.
- **Only 18 of the 36 added tokens are `special`.** `</think>`, `<tool_call>` and the `<arg_key>` /
  `<arg_value>` markers are added-but-not-special, so `skip_special_tokens` must not drop them —
  they are exactly what the stream splitter and tool parser exist to find.
- **`reasoning_effort` accepts only `low` and `high`.** Everything else, *including omitting it*,
  renders as `Max`. The default is the most expensive setting, and `medium` is silently `Max`.

Then the 3-layer smoke run found the bug the gates could not. With `--n-layer 3` the text is
meaningless but the plumbing is real, and the two response paths disagreed: the streaming splitter
knows generation starts inside a `<think>` block, but the non-streaming parser looked for a
`</think>` and, not finding one in a truncated generation, called the whole thing `content`. Same
tokens, same request, different answer depending on whether the client asked for a stream. Fixed by
telling the parser where it starts; gated both ways.

**Worth keeping:** a smoke test on a deliberately undersized load is not a lesser test. It cost
10 GiB and four minutes and caught a real divergence that 256 unit checks did not, because the bug
lived in the seam between two components that were each individually correct.

---

## #7 — The multi-token forward, bit-exact, and what it costs to verify a draft

`forward_batch` runs M tokens in one pass, reading each weight ONCE instead of M times. It is
gated **bit-exact** against M sequential `decode()` calls — equality of floats, not closeness — at
widths 1,2,3,4,5,8, at ragged widths, and interleaved with single-token decodes. That standard is
not fussiness: a speculative verify that merely approximates the AR path silently stops sampling
from the AR distribution, which no benchmark can falsify. It is achievable because the batched gemm
reduces in the same order as the gemv, and every non-gemm kernel is called per token with offset
pointers.

The gate runs at 4 layers, which reaches layer 3 — **the first MLA and MoE layer — so both are now
exercised inside the engine loop**, not only standalone. And it needs no PyTorch oracle at all: its
reference is the engine's own already-gated sequential path. That makes it the one whole-engine
gate that runs while the box is busy, which is most of the time.

Two decisions recorded with their arithmetic rather than their intuition:

- **The routed experts are deliberately NOT batched.** At the widths speculation uses, K tokens
  select almost disjoint expert sets — 29.4 distinct of a possible 32 at K=4 — so batching them
  would save **4.7%**. It reaches 1.9x at K=32, so it is worth doing for wide prefill chunks and
  not for verify. Until then the already-gated batch-1 path is reused.
- **Prefill now chunks through it**, which was the server's single largest cost: sequential prefill
  pays the full 19.76 GB weight read for every prompt token, where a chunk of C amortises 15.005 GB
  of it. 2.3x at C=4, 3.8x at C=32.

And the correction that matters most, in SPEC_DECODE.md: **the draft head's own `lm_head` is
1.269 G per drafted token**, which is larger than the entire rest of the MTP block. Folding it in
drops the predicted speculation win from 1.59x to 1.38x. It should be NVFP4, and doing so is
system-level lossless — a draft error costs a rejection, not a wrong output — which brings it to
1.49x. The largest single draft cost sits in the one place where reduced precision cannot hurt
quality.

---

## #8 — The DSA indexer, and three gates that had to be rewritten because they were wrong

The context ceiling is gone: the engine ran dense to 2051 and sparse above it, and a 4,073-token
prompt now serves end to end where anything past 2048 used to be refused.

**What the kernel had to get right, none of it guessable from the config.** `k_norm` is a LayerNorm
*with a bias* (eps 1e-6), not the RMSNorm used everywhere else in this model. The pool softmax is
per *channel* over the 4 tokens, not per token over the channels. A trailing incomplete pool is
never selectable but its tokens are appended raw — which is why the dense limit is 2051 and not
2048. And the head weights can be NEGATIVE, so `index_score` is a signed sum of relu terms that
routinely produces `-0.0`; a top-k comparing raw bits would order that below `+0.0` and silently
select different pools. The ported `topk_radix.h` already canonicalises it, which is most of why it
was worth porting rather than writing.

**Three times the gate reported a failure that was not one, and each taught something.**

1. *The oracle ran bf16, the kernel arm runs fp32.* Pool scores differed in the third decimal, and
   at lengths where the top-k actually excludes something the 512th-place pool flipped — 404 of
   3003 rows, each off by exactly one pool. It looked like a logic bug and was a dtype mismatch in
   the harness. bf16 weights widen to fp32 exactly, so an fp32 oracle uses identical weights with a
   wider accumulator, and the disagreement vanished entirely.
2. *Comparing the emitted ARRAY against the reference's array is not testable.* The oracle runs
   prefill, so its `select_k` and slot offsets differ from decode's even when the visible-key SET
   is identical. Attention consumes a set — it softmaxes over the selected keys — so the set is the
   invariant and the array is an artefact of how the reference was captured.
3. *Order cannot be compared on near-ties.* Our scores agree with the reference to 1.7e-6, and 2 of
   the 511 adjacent gaps in the T=2050 ranking are TIGHTER than that — rank 412 to 413 is 5.1e-7.
   Two correct fp32 implementations with different reduction orders swap that pair. The check
   became "is every chosen pool at or above the select_k-th best score", which keeps every real
   ordering bug and drops the artefact.

**And once, the gate's own bug.** At T=38 it walked past the pool region into the raw tail and read
a tail token as a pool id. The rule that keeps paying: when a gate fails, the gate is a suspect too.

**Two design choices worth keeping.**

- **The emit is sorted ASCENDING**, discarding the score order. Cache reads become sequential
  instead of a gather — and, more usefully, the sparse path becomes bit-identical to the dense one
  whenever the indexer selects everything, because the fp32 context sum then accumulates in the
  same order. That converts "sparse agrees with dense below 2051" into an EXACT test needing no new
  oracle, which is the only whole-path check available above the reference's reach.
- **The selected count stays on the device.** Reading it back to size the launch would mean a
  stream sync per full-attention layer per token — 11 pipeline stalls a step, to avoid launching
  blocks that exit in nanoseconds.

**One trap avoided by construction:** pool keys are built incrementally as each group of 4 tokens
completes, so `indexer_keys` must run from token 0 **even while attention is still dense**. A
version that started the indexer only once the context crossed 2051 would have no pool keys for the
first 512 pools — which is most of the context it then has to score.

**Below the limit the dense path is kept**, because the indexer provably selects everything there
and scoring every pool to conclude "all of them" is wasted work. `force_sparse` exists purely so
the gate can exercise the sparse kernels where a known answer exists; without it that gate would
have been comparing dense against dense and passing vacuously — which it briefly did.

---

## #9 — The decode gap is the MoE, and it inverts the optimisation order

`tools/bench_decode.cu`, 45 layers, 24 steps, 512 ctx, **246.8 GB/s measured in-process with the
model resident**. 3.94 tok/s against a 12.49 roofline = **32%**.

```
phase                        ms       %    calls     GB/tok     GB/s     %BW
ATTENTION               1638.11   27.0%     1080     11.950    175.1     71%
FFN                     4145.85   68.2%     1080      6.307     36.5     15%
lm_head                  142.94    2.4%       24      1.269    213.1     86%
  attn:kda              1260.73   20.8%      816      9.366    178.3     72%
    kda:qkv+conv         780.02   12.8%      816      6.850    210.8     85%
    kda:o_proj           260.41    4.3%      816      2.283    210.4     85%
    kda:gates            167.00    2.7%      816      0.500     71.9     29%
  ffn:moe               4037.94   66.5%     1008      5.401     33.9     14%
    moe:w13+act         2608.75   42.9%     1008      3.568     32.8     13%
    moe:w2+combine      1396.25   23.0%     1008      1.783     30.7     12%
TOTAL                   6074.75  (253.11 ms/step, 3.95 tok/s)
```

**The MoE holds 66.5% of the step while moving 27% of the bytes, at 13% of achievable
bandwidth.** Everything else is already close to the machine: the KDA projections and `lm_head`
run at 85–86%, KDA as a whole at 72%, MLA at 67%. There is no diffuse inefficiency to hunt —
one kernel family owns the entire gap.

At the 85% the KDA gemvs already demonstrate, `ffn:moe` would take 617 ms instead of 4038, and
the step would fall to 110.6 ms — **9.04 tok/s, a 2.29x speedup from one kernel family**.

### This reverses ROOFLINE §3

§3 calls NVFP4-ing the bf16 dense weights "worth more than every kernel optimisation combined",
on a −51% `B_tok`. That arithmetic silently assumes every phase converts bytes to time at the
same rate. **They do not**, and the phases §3 targets — KDA, MLA, `lm_head` — are exactly the
ones already at 72–86%. Halving their bytes saves ~890 ms of 6075:

| order | step | result |
|---|---|---|
| NVFP4 dense weights first | 6075 -> 5188 ms | 4.63 tok/s (**1.17x**) |
| fix the MoE kernels first | 6075 -> 2654 ms | 9.04 tok/s (**2.29x**) |
| then NVFP4 on top of that  | 2654 -> 1764 ms | 13.6 tok/s (1.50x more) |

Same two changes, and doing the cheap-looking one first buys 1.17x instead of 2.29x. **Fix the
MoE kernels first.** This also matches the recorded prior on this box (`dspark-decode-gap-research`:
"top lever = HW-unpack FP4 MoE GEMV") — arrived at there by a different route, on a different model.

### A wrong number the profile caught on its way past

`moe:router` reported **313 GB/s on a 247 GB/s machine**, and `mla:indexer` 180%. A phase cannot
beat the memory system, so an impossible row means the byte count on that row is wrong, not the
kernel. The router's was: `roofline.py` bucketed on `body.startswith('mlp.gate')`, which also
prefixes `mlp.gate_proj.weight` — the SwiGLU gate of the three DENSE MLPs, a `[12288, 4096]` bf16
that is 6x the real `[144, 4096]` router. `moe router` read 0.352 G instead of 0.051 G, and
`dense mlp` was under-reported by the same 0.302 G.

`B_tok` is unaffected at **19.761 G** — both buckets were already inside it, so this is a
reclassification, not a correction to the headline. Fixed with one character (`'mlp.gate.'`).
`mla:indexer`'s 180% is different and legitimate: 0.164 G is a long-context figure and this bench
runs at 512 ctx. Those cells now print a dash rather than a number that would invite a wrong lever.

### Secondary targets, once the MoE is fixed

`kda:gates` at 29% (167 ms) and the two `hc:` compose marks at 6% (119 ms combined) are together
4.7% of the step — worth having, worth nothing before the MoE.

---

## #10 — The MoE kernels, fixed: 2.12x on the whole step

#9 said the MoE owned 66.5% of the step while moving 27% of the bytes, and that fixing it was
worth 2.29x. It was worth 1.94x on its own and 2.12x with the follow-on work.

    253.11 ms/step  ->  119.63 ms/step        3.94  ->  8.36 tok/s
    ffn:moe  4038 ms -> 857 ms  (4.71x)       13% -> 82% of achievable bandwidth

Every change below is gated against the PyTorch oracle on real checkpoint weights, and
`gate_batch` (forward_batch at M=1 must be bit-identical to decode) passes throughout.

| # | change | why it was slow | gate | measured |
|---|---|---|---|---|
| a | **MoE: warp-per-row + HW FP4 unpack + ILP 4** | Three faces of "not enough bytes in flight". (1) Block-per-row: at BS=128 each thread ran TWO iterations, then paid a 5-step shuffle, a shared round trip and a `__syncthreads` — the reduction cost more than the work. (2) A `__constant__` LUT for the e2m1 nibble: constant memory broadcasts only on a uniform address, and every lane reads a different one, so all 16 lookups per group serialised up to 8 ways. (3) ILP=1. Replaced with warp-per-row (shuffle reduce only), `__nv_cvt_fp4x2_to_halfraw2`, and a 4-deep unroll issuing 8 loads before consuming any. `k_expert_down`'s serial 9-slot loop moved into the grid (9x the warps) with a fixed-order combine, so it stays deterministic. | `gate_moe` cos 1.000000000, max_rel 3.738e-06 (baseline 3.723e-06) | **ffn:moe 4038 -> 968 ms**, step 253.11 -> 130.16 |
| b | **Stage the MoE activation in shared** | Each of the 8 warps re-read all of `x` from L1 for both its gate and its up row: 32 bytes of fp32 activation fetched per 4 bytes of weight. | unchanged, 3.738e-06 | ffn:moe 968 -> 846 ms, step -> 125.26 |
| c | **`gemv`/`gemm` block size as a function of K** | Fixed BS=256 on the `[8192, 128]` KDA gate projections left 224 of 256 threads idle and the launch degenerated to latency. Now 32/64/256 by K, picked by the *same* rule in both so the batch invariant holds. | `gate_batch` 13/13 | **kda:gates 171 -> 80 ms (2.14x)** |
| d | **Warp-parallel Sinkhorn + split `k_hc_mix`** | 20 iterations x 8 reductions walked serially on thread 0 of a single block while 255 lanes waited. Now 16 lanes hold the 4x4, rows reduce over `__shfl_xor` bits 0-1 and columns over bits 2-3. `k_hc_mix` went from MIX=24 blocks (one per SM) to 96, with bf16x2 loads. | `gate_layer` post 6.773e-08, comb 8.353e-08 | **hc:pre 119 -> 56 ms (2.12x)** |
| e | **Ping-pong the hyper-connection streams** | `hc_apply` reads the pre-site streams and writes the post-site streams — they never alias, so the `cudaMemcpyAsync` snapshotting them into `resid_` was moving 64 KB twice per layer, 5.9 MB and 90 launches per token, to make a buffer the next kernel could have read in place. | `gate_stack` 4/4 | folded into (d) |

### Three things that were tried and did NOT work

Recorded so they are not retried. All three are cases where the obvious reasoning was right about
the mechanism and wrong about which mechanism binds.

1. **Inline `fp8e4m3()` instead of the shared scale LUT.** The LUT is indexed by the scale BYTE
   VALUE, which is effectively random across a warp — a textbook shared bank conflict, once per 8
   weights. Decoding arithmetically instead is six ALU ops and no memory. Measured:
   **w13+act 535 -> 637 ms, a 19% regression.** This kernel is instruction-bound, not
   shared-bandwidth-bound, so trading a conflicted LDS for eight more ALU ops is backwards. The
   LUT stays.

2. **half2 math in the MoE (`GLM5_MOE_HALF=1`, kept, default off).** Halves the instruction count
   per 8 codes (the e2m1 unpack already *produces* a half2, so consuming it as one removes the
   conversion) and halves the staged activation. The 0731 engine got **2.59x** from exactly this.
   Here, measured round-robin against the fp32 path so a contention spike lands on both:

   | | fp32 | half2 |
   |---|---|---|
   | run A | 8.33 tok/s | 8.17 |
   | run B | 8.29 | 8.16 |

   **Consistently ~1.6% slower**, and it is not the same function: `gate_moe` cos 0.999999913,
   rms_rel 5.738e-04. Two reasons not to ship it: no win, and the 0731 engine's identical change
   cost that engine its draft-head acceptance (3.12 -> 1.00 tokens/verify, its #9). We have an MTP
   head at 72.3% acceptance never fine-tuned against a perturbed target. The flag stays so the
   measurement is repeatable, not because it is a candidate.

3. **4-deep ILP unroll in `gemv`.** Threaded `acc` through helpers so the addition order is
   unchanged and the result stays bit-identical. **Measured neutral**: mla:o_proj 235.20 -> 235.82,
   kda:qkv+conv 823.63 -> 823.29, lm_head 151.11 -> 152.39. The compiler was already pipelining
   the rolled loop. Kept (it is free and bit-identical), but it is not a lever.

### CUDA graphs are NOT a lever here, and the profile says so

Worth capturing because it was the obvious next idea and cost nothing to rule out. Over 24 steps,
**wall 2978.18 ms against a dprof kernel total of 2967.30 — a 0.4% gap.** The GPU is essentially
never idle between launches, so there is no launch overhead for a graph to remove. The 0731 engine
got 1.17x from a full-step graph; that engine had a different launch profile. Do not port it.

### Where the remaining gap is

    phase              ms      % step   GB/s   note
    kda:qkv+conv     823.3     28.7%   199.7   at the machine
    ffn:moe          857.3     29.9%   151.2   fixed; the residue is instruction-bound FP4
    kda:o_proj       287.9     10.0%   190.3   at the machine
    mla:o_proj       235.8      8.2%     -     at the machine
    lm_head          152.4      5.3%   199.9   at the machine

**Kernel efficiency is now at parity with the 0731 engine**: we run at 67% of achievable
bandwidth, that engine at 68% (14.61 tok/s against its 21.42 roofline). The remaining absolute
difference in tok/s is not kernel quality, it is `B_tok` — 19.761 G/token here against ~11.2 G
there. No further kernel work moves it much; §3's lever does.

---

## #11 — The activations were the cost all along: NVFP4 dense weights, and prefill

Two levers in one finding. ROOFLINE §3 said quantising the 13.9 G/token of bf16 dense weights
would halve `B_tok` and be worth ~1.5x. It halved `B_tok` and was worth **1.4%**. Chasing that is
what produced the finding, and the finding then fixed prefill too.

### What the overlay does

`tools/requant_dense_nvfp4.py` converts every AR-path tensor that is read through `gemv`/`gemm`.
13.16 GiB of checkpoint becomes a **3.70 GiB overlay** — not a rewritten checkpoint. The base is
opened read-only, nothing is destroyed, and a family is disabled by not emitting it, so
`--families` IS the gate at zero runtime cost. `B_tok` 19.761 -> **9.762 G, -50.6%**.

`kv_b_proj` and the hyper-connection `fn` tensors stay bf16 on purpose: MLA reads `kv_b` strided
inside `k_absorb_q`, not through `gemv`, so converting it would need a second kernel for 1.7% of
`B_tok`. The MoE router stays bf16 too — 0.05 G, and it decides which experts run.

### Three kernels, and only the third is worth anything

| | | |
|---|---|---|
| uint32 loads, shared scale LUT | 8.36 -> **8.48** tok/s | halved the bytes, bought 1.4% |
| uint4 loads, hardware e4m3 | 8.48 -> **3.55** tok/s | 3.4x worse still |
| row-tiled, R=5 | 8.36 -> **11.53** tok/s | **1.38x** |

`tools/bench_gemv -DFP4_PROBE` stubs one term of the inner loop at a time. It settled in one run
what two engine-level A/Bs (20 minutes each) could not:

```
stub the e2m1 unpack   ->  17.1 GB/s     no change; cvt.rn.f16x2.e2m1x2 is free
stub the e4m3 scale    ->  17.1 GB/s     no change
stub the x reads       -> 323.7 GB/s     19x
```

**A gemv reads 4 bytes of activation per weight.** Against bf16 that is 2 bytes of x per byte of
weight; against NVFP4 it is **7.1**. bf16 sits at 88-96% of streaming DRAM and is at the right
wall. NVFP4 asking for 1.7 TB/s of x to match it is not at any wall worth being at. *Halving the
weight bytes cannot help while x is seven times the weight traffic* — which is why the first
version removed half of `B_tok` and changed nothing, and why the second, which made each load
wider, made it worse: a uint4 ties one lane to 32 contiguous weights, so at a fixed offset the
warp's 32 lanes touch 32 different 128-byte lines and use 4 bytes of each.

So each block owns R output **rows** and reads x once for all of them. R is measured, not
reasoned, and the peak is sharp:

| R | 2 | 3 | 4 | **5** | 6 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|
| GB/s (kda q/k/v) | 115 | 135 | 150 | **161** | 135 | 132 | 100 | 75 |

Final per-shape, against bf16: 1.74x to 3.19x, 112-170 GB/s.

### The same bug was in the batched path, and it was the prefill anomaly

Prefill had never been profiled. It got **worse** with wider chunks — 81.7 ms/tok at width 1,
68.7 at width 4, **84.0 at width 16** — which is backwards for a path whose entire purpose is to
amortise a weight read across tokens.

`tools/bench_gemv --m` shows why, and shows it in the **bf16** kernel too:

| M | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| bf16 weight GB/s | 210 | 111 | 56 | 28.7 | 14.2 |

Exactly 1/M. **A gemm reads W once and costs M times as much anyway**, because it reads M rows of
x per weight and x was already the binding term at M=1. At M=16 that is 40 GB of activation
traffic for lm_head alone — ~450 GB/s, which is the L2 ceiling this box actually has.

That invalidates ROOFLINE §4's premise as *implemented*: batch cost was not flat in K, it was
linear, so a multi-token forward saved nothing and speculative verification would have won
nothing no matter how good the draft head was. The curve is still ours to build; it just was not
built yet.

Fix: chunk M and tile rows against each other. Per weight the cost is
`0.5625 * (M/MCHUNK)` of weights plus `4 * MCHUNK / R` of activations, under a register budget of
roughly `R * MCHUNK`. MCHUNK=4, R=8 measured best (18.2 GB/s on kda o_proj at M=16 against 7.7).
R=12 and R=16 both lose to register pressure — 93 registers already.

### Batching the MoE over tokens

`forward_batch` ran the routed experts as M sequential `moe_forward` calls, and `ffn:moe` cost the
same 35.1 ms per token at width 16 as at width 1. The token is now a grid dimension
(`blockIdx.z`), so the M tokens go through in one set of launches. At `gridDim.z == 1` this is
exactly the kernel it was, which is why `gate_batch`'s bit-identity still holds.

It is worth 5%, not the 33% the expert-overlap arithmetic suggests, and the reason is worth
recording: at M=16 about 42 of the 128 expert selections are repeats, but each expert triple is
4.7 MB and 86 distinct experts is ~400 MB per layer — nothing like an L2 working set, so the
second read is not served from cache. **The win here is occupancy, not reuse.** Actually saving
those bytes needs true expert-gathering — one block per distinct expert, looping over its token
list with the weight row held in registers — and that is a new kernel, not a grid change.

### Where it landed

| | before | after | |
|---|---|---|---|
| decode | 8.36 tok/s | **11.86 tok/s** | 1.42x |
| prefill @ width 16 | 84.0 ms/tok | **61.2 ms/tok** | 1.37x |
| prefill @ width 1 | 81.7 ms/tok | 81.3 ms/tok | (unbatched, unchanged) |
| `B_tok` | 19.761 G | **9.762 G** | -50.6% |

Prefill is now flat from width 4 upward (61.3 / 61.2 / 61.8 at 4 / 16 / 32) rather than rising.
Flat, not falling, because `ffn:moe` is 48% of it and still pays full price per token.

### What this cost in accuracy, and who decides

NVFP4 is e2m1 with one fp8 scale per 16. `gate_nvfp4` reads both the bf16 tensor and the NVFP4
triple off disk and compares: **rel 0.088-0.100, cos 0.9950-0.9961**, uniform across all seven
families — the format's own band, not a property of any family. That band is the discriminator: a
layout bug lands at cos ~0, not at 0.995.

End to end over three KDA layers against the PyTorch oracle, `gate_stack` reports **cos 0.9972**.
That is a real change and it is the operator's call, not a gate's, so `gate.sh` runs `gate_stack`
on bf16 for exactness (4/4, cos 1.000000000) and prints the NVFP4 drift beside it as a report.
The experts in this checkpoint were already NVFP4; this extends 4 bits to the dense weights.
Reverting any family is a re-run of the requant script with a shorter `--families`, and
`GLM5_DENSE_NVFP4=0` reverts all of it without touching a file.

### Two traps

**A retained mmap map reads garbage across two loads.** `WeightStore` kept its shard-to-blob maps
as members. The mmaps die with each `load()`, so the overlay's mmaps landed at the same addresses
and overlay tensors resolved against base-checkpoint blobs. `gate_stack` came back at cos 0.0005
with activations of 4e21 — while every kernel gate stayed green, because none of them load two
directories. The maps are now local to one call.

**A dprof table can report a prefix of the run.** The first prefill profile said 58% of the time
was outside any kernel. It was not: `moe_forward` records 3 mark pairs per token per MoE layer, so
a 256-token prefill at width 16 wants ~130k events and the 65536 default captured 41% of them.
The giveaway was `lm_head` at 27224% of bandwidth. An impossible row is never a fast kernel — and
here it was not even a wrong byte count, it was a truncated recording.

---

## #12 — Expert gathering: prefill 84.0 -> 51.5 ms/tok

#11 made the token a grid dimension and got 5%. It could not get more, and the reason says what
this had to be: **a grid dimension does not change how many times a weight is READ.** M tokens x 8
slots is M*8 reads of an expert triple whether they run together or in sequence. At M=16 about 42
of those 128 selections are repeats, and they were not served from cache — 86 distinct experts is
~400 MB per layer, nothing like an L2 working set.

So the repeats are collapsed in the kernel. `k_build_work` (one block, ~1 KB of shared) counts the
pairs routed to each expert, lays them out contiguously, and emits one work item per
(distinct expert, tile of TT=4 pairs). `k_expert_act_gathered` and `k_expert_down_gathered` then
give one block to one work item: it reads that expert's rows **once** and applies them to every
token in the tile.

Expected reads fall from `M*8` to roughly `144*(1-(1-8/144)^M)`:

| M | 4 | 16 | 32 | 64 |
|---|---|---|---|---|
| unbatched reads | 32 | 128 | 256 | 512 |
| distinct experts | 27 | 86 | 121 | 140 |

### Two things the shape had to respect

**K is tiled, not fully staged.** The old kernel staged all of `x` in shared and indexed it by
absolute k. The gathered one holds TT tokens, so it tiles K at 1024 — chosen to land exactly on
the unrolled loop's 1024-element step, so the accumulation sequence is the same
`base = 0, 1024, 2048, ...` it always was. 16 KB of shared, unchanged occupancy.

**The accumulators are compile-time indexed.** `for (t = 0; t < TT; ++t) if (t < nt)` rather than
`t < nt`, because a runtime-indexed accumulator array spills to local memory and the kernel stops
being bandwidth-bound, which is the entire point of it.

**The grid is sized for the worst case**, since `n_work` is a device value and a host-side grid
cannot see it. Two bounds, tighter wins: every item holds at least one pair, and separately
`sum_e ceil(cnt_e/TT) <= M*KS/TT + distinct`. At M=128 that is 432 blocks rather than 1152.

### Bit-identity is structural, not lucky

Every (token, slot) dot product is independent, so grouping changes which block computes it, never
the value or the order of its accumulation. The pair order *within* a group comes from an atomic
and is therefore arbitrary — which is fine precisely because each pair writes its own slot of
`act` and `part`, and `k_down_combine` still sums the slots in a fixed order with no atomics.

`gate_batch` checks it the only way worth checking: 12 sequential decodes as reference, then
K = 1, 2, 3, 4, 5, 8 and ragged widths must match **bit for bit**. 13/13.

M=1 stays on the ungathered path. There is nothing to collapse at M=1, and it keeps decode on
exactly the kernels it was gated on.

### Result

| chunk | 4 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| ms/tok before #12 | 61.3 | 61.2 | 61.8 | — | — |
| ms/tok after | 57.5 | 52.8 | **51.7** | 51.5 | 52.1 |

Prefill now *falls* with width instead of rising, which is what a batched path was always supposed
to do. The default chunk is 32: past that it stops improving and only costs buffers.

**84.0 -> 51.5 ms/tok end to end, 1.63x**, decode unchanged at 11.60-11.86 tok/s.

### What is now the laggard

`attn:mla` is 23.8% of prefill at 35% of achievable bandwidth — it has become the worst phase by
efficiency, and its sub-phase marks do not exist in the batch path, so attributing it needs those
marks first. `ffn:moe` is 38.4% and its byte model now over-reports (the row reads >100%) because
`kBytes` still prices M*8 expert reads rather than the distinct count; that row is a wrong byte
count, in the direction that means the gathering is working.
