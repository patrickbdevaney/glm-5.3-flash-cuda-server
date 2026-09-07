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

---

## #13 — MLA sub-phase marks, and what they say

`mla_batch_step_dsa` had no sub-phase marks at all, and `mla_decode_step_dsa` was missing two —
`DP_M_QPROJ` and `DP_M_KV` existed only in the non-DSA `mla_decode_step`, which the engine never
calls. So `attn:mla`'s children summed to 217 of 278 ms in the decode table and 0 of 5405 in
prefill. Both paths are now marked identically, including which kernel goes in which bucket, so
the two tables can be read against each other. Children now account for **99.2%** of `attn:mla` in
decode and **99.4%** in prefill.

### The call counts are the finding

| row | ms | calls | |
|---|---|---|---|
| `mla:q_proj` | 681 | 198 | = 11 layers x 18 chunks — **batched** |
| `mla:kv` | 120 | 198 | **batched** |
| `mla:indexer` | 287 | 6336 | = 198 x 32 — **per token** |
| `mla:absorb_q` | 861 | 6336 | **per token** |
| `mla:sdpa` | 965 | 6336 | **per token** |
| `mla:o_proj` | 2460 | 6534 | 6336 `expand_v` + 198 batched gemm |

A row whose call count scales with M is a row that did not get batched. **The projections are 15%
of MLA and the per-token attention loop is 85%** (~4570 of 5405 ms at chunk 32).

### `kv_b` is read twice per token, and ROOFLINE counts it once

`k_absorb_q` and `k_expand_v` each stream the whole of `kv_b` — bf16 `[32768, 512]`, 33.55 MB per
MLA layer — and both run per token. From the decode table: `absorb_q` is 0.1425 ms/call, i.e.
**235 GB/s**, which is the machine; `expand_v` (o_proj's 0.561 ms/call less the ~0.337 ms the
NVFP4 gemm takes) is ~150 GB/s. Neither is a slow kernel. They are simply reading 67.1 MB per
token per layer.

Over 11 layers that is **0.738 G/token, 7.6% of `B_tok`** — and ROOFLINE §1 prices `kv_b` at
0.344 G because it counts one read, not two. **This corrects OPTIMIZATION_LOG #11**, which
dismissed converting `kv_b` as "1.7% of `B_tok` for a second kernel". It is 7.6%, and the second
kernel now has a much better case.

### What that makes the next MLA lever

Batching `absorb_q` and `expand_v` over the M tokens, so `kv_b` is read once per chunk instead of
M times. It is the same insight as #11 and #12 — the weight does not care who reads it — and it is
the last place in the engine where a per-token loop streams a whole tensor. Converting `kv_b` to
NVFP4 is worth 3.55x on top of that, but batching comes first and is worth ~M.

---

## 14. Batching `absorb_q` and `expand_v` over the chunk — prefill 50.66 → 47.79 ms/tok

The lever #13 named, built. `k_absorb_q` and `k_expand_v` each stream the whole of `kv_b`
(bf16 `[32768, 512]`, 33.55 MB per MLA layer) and both ran once per token. Now a block owns
(head, chunk of `MLA_MB` tokens) and reads each weight once for all of them, so `kv_b` traffic per
chunk falls by `min(M, MLA_MB)` rather than not at all.

Two structural changes made it possible:

- **`absorb_q` hoists out of the per-token loop entirely.** It reads only `q` and `kv_b` — not the
  cache, not the indexer state, not any other token's attention — so computing all M up front is
  free. `qa` becomes `[M, 64, 512]`.
- **`ctx` becomes per-token** (`[M, 64, 512]`), because `expand_v` can only batch if every token's
  context survives until the end of the loop. That is 4 MiB apiece at M=32, paid once.

`scores` stays single-token: the score/softmax/context chain is still serial, and one
`[64, max_ctx]` scratch is all it needs.

`MLA_MB = 8`. Larger spills `acc[]` out of registers at 512 threads, which costs more than the
extra reuse buys. The `if (i < nm)` guards sit inside `#pragma unroll` loops over a compile-time
bound for the same reason — a runtime bound forces the accumulator to local memory.

### The A/B

`-DMLA_MB=1` keeps the hoisted structure and removes only the per-block reuse, so it isolates the
traffic change rather than the launch-count change. Same widths, same prompt, round-robin, two
reps; the MB=8 run's own streaming probe read *lower* (227.7 vs 233.7 GB/s), so if anything this
understates the gain.

| row | MB=1 | MB=8 | |
|---|---|---|---|
| `mla:absorb_q` | 1762.07 ms | 320.44 ms | **5.50x** |
| `mla:o_proj` (incl. `expand_v`) | 4511.15 | 2818.15 | 1.60x |
| `attn:mla` | 11310.03 | 8190.14 | 1.38x |
| `ATTENTION` | 28507.00 | 25390.60 | 1.12x |
| TOTAL | 56461.08 | 53405.92 | 1.06x |

| width | MB=1 min ms/tok | MB=8 min ms/tok |
|---|---|---|
| 16 | 51.275 | 48.523 |
| 32 | 50.656 | **47.793** |

**The control is what makes this a measurement.** `mla:sdpa` and `mla:indexer` were not touched by
this change and must not move: sdpa reads 3015.68 → 3019.78 ms, 0.14%. A run where the untouched
rows drifted would not be readable at 6%.

The `calls` column closes the diagnosis #13 opened: `absorb_q` and `o_proj` now read 550, the same
as the batched projections, where they read 550xM before. Only `indexer` and `sdpa` still scale
with M, and both are inherently per-query.

### Why 6% and not ~M

Because `absorb_q` was only 3.1% of prefill to begin with. The kernel itself got 5.5x — the byte
model was right — but Amdahl caps what that is worth end to end. `attn:mla` is now 15.3% of
prefill (was 20.0%) at 54% of achievable bandwidth (was 38%), and inside it the remaining mass is
`sdpa` (3020 ms) and the `o_proj` gemm. **`sdpa` is now the largest MLA row**, and it is cache
traffic, not weight traffic — a different problem from every lever in this log so far.

Converting `kv_b` to NVFP4 is still worth 3.55x on what `absorb_q`/`expand_v` read, but that is
now 0.6% and 5.3% of prefill respectively, so it buys far less than it would have before this
change. It has moved from "next" to "probably not worth a second kernel" — the honest reversal of
what #13 predicted, in the direction that matters.

---

## 15. `k_context`: 5.2x, and two ways a benchmark lied on the way there

After #14, `mla:sdpa` was the largest MLA row and the only cost in the engine that GROWS WITH
CONTEXT — everything else costs the same at token 100 and token 3000. New level-3 dprof marks
(`sdpa:scores` / `sdpa:softmax` / `sdpa:context`) put the mass squarely in one kernel:

    mla:sdpa      39868 ms   17.0% of prefill      (prompt 2048, width 32)
      sdpa:scores  5770
      sdpa:softmax  364
      sdpa:context 30420      76% of sdpa, 13.1% of the whole prefill

At 256-token prompts sdpa is only 5.7% of prefill, so the earlier profile understated this badly.
**Sizing a context-dependent lever at one context length is how it stays invisible.**

### The two knobs

`k_context` launched `Hh/HG = 4` blocks. Two ways to get more:

- **HG (heads per block).** Lowering it multiplies blocks AND cache re-reads by the same factor.
- **NT (t-tiles), new.** Each tile owns a DISJOINT slice of the latent cache, so NT tiles read the
  cache once between them. Parallelism at no extra traffic, paid for with a partial buffer and a
  reduce pass. NT is compile-time and `grid.y` is always NT, tiles past the end writing zeros, so
  the dense and sparse twins partition `t` identically and stay bit-exact against each other
  (`gate_mla_sparse`).

### The result inverts the obvious model

`tools/bench_context.cu` sweeps the grid on the REAL kernels (extracted to `include/mla_context.cuh`
so the bench cannot drift from what ships). At n_tok=2048:

| HG/NT | blocks | cache reads | min us | GB/s |
|---|---|---|---|---|
| 16/1 (shipped) | 4 | 4 | 1012.2 | 16.8 |
| 16/16 | 64 | 4 | 260.2 | 80.6 |
| 8/8 | 64 | 8 | 173.9 | 205.0 |
| 4/1 | 16 | 16 | 162.2 | 415.4 |
| **4/16** | 256 | 16 | **116.8** | 610.3 |
| 2/16 | 512 | 32 | 157.7 | 877.7 |
| 1/4 | 256 | 64 | 208.9 | 1290.2 |

**HG=4 reads the cache 16 times per call and beats HG=16, which reads it 4 times, by 6.5x.** At
equal block count (64), HG=4/NT=4 is 1.8x faster than HG=16/NT=16. So cache re-reads were never
the constraint: the latent cache is L2-resident at these sizes — the winning point runs at
610 GB/s against a 237 GB/s streaming read — and the binding constraint is per-SM occupancy, which
`acc[HG]` destroys. Below HG=4 the traffic finally does bite: HG=1 saturates L2 at ~1290 GB/s.

This is the opposite of #11, #12 and #14, where widening the consumer to cut weight traffic was
always right. **Weight traffic goes to DRAM; this cache fits in L2.** Same shape of loop, opposite
lever, and the byte model gave the wrong answer for the first time in this log.

### In the engine (prompt 2048, width 32, reps 2)

| | before | HG=4 | HG=4/NT=16 |
|---|---|---|---|
| `sdpa:context` | 30420 ms | 7588 | **5829** (5.22x) |
| `mla:sdpa` | 39868 (17.0%) | 17028 (8.1%) | 15280 (7.4%) |
| prefill min | 52.272 ms/tok | 48.582 | 48.589 |

Control: `sdpa:scores`, untouched by any of this, 5770 -> 5614 -> 5590. Decode is unchanged at
11.55 tok/s (its contexts are short, so `k_context` is not where its time goes).

Note the honest split: **HG bought the 7.0% end-to-end, NT bought 1.30x more on the kernel and
under 1% end-to-end.** `sdpa` is now 7.4% of prefill with `scores` and `context` about equal, which
is the point where this stops being the largest lever.

### Two traps, both caught, both worth remembering

**1. A kernel that fails to launch looks infinitely fast.** The first HG sweep reported HG=32 as
6.3x faster than HG=16 and HG=64 faster still — 12x off the trend the valid points sat on. Both
were `too many resources requested for launch`: `acc[HG]` at 512 threads exceeds the register
budget, the kernel never ran, and dprof honestly timed an empty stream slot. Nothing in the
benchmark noticed; only `gate_mla` did. Launches are now checked by `KCHK` and name the kernel.
The impossible-row rule from `dprof.h` generalises: **a row far off its own trend line in the
"too good" direction is a broken measurement, not a fast kernel.**

**2. A correctness check can reject a correct kernel.** `bench_context` compares every grid point
against HG=4/NT=1, and the first two runs rejected all 10 NT>1 points at max_rel ~1e-2, grouped
perfectly by NT and stable across reps. Both causes were the test, not the kernel: `s` was
generated with random SIGNS, so a 2048-term sum cancels to near zero, and max-per-element relative
error then divides a 1e-7 absolute error by a near-zero output. Fixed by generating `s` as an
actual softmax row (non-negative, summing to 1) and scoring with relative L2. Reassociation from
t-tiling is real and unavoidable; the metric has to be able to tell it from a bug.

**3. A failed build left gates that passed.** `scripts/build.sh` has `set -e`, so a compile error
in `kernels/mla.cu` stopped the run — after the earlier targets were already written. The stale
`gate_mla`, `gate_mla_sparse` and `gate_batch` from the previous revision then ran and reported
13/13 green on code that did not compile. `build.sh` now deletes every target before building.
CLAUDE.md §2 says a gate that passes against a dead engine is worse than no gate; this is how one
gets created by accident.

---

## 16. The byte model stops being a model

`ffn:moe` printed **105% of bandwidth** and `lm_head` **11399%**, and by `dprof.h`'s own rule a row
over 100% is a wrong byte count, never a fast kernel. Both were. The cause was structural, not a
typo: `kBytes` is a **per-decoded-token weight model**, and in prefill a weight read serves M
tokens. Every gemm-backed row was priced ~`M/ceil(M/MCHUNK)` too high — about 4x at width 32.

MoE was worse than wrong-by-a-factor. Since the expert-gathering kernel (#12), each distinct expert
is read once per chunk rather than once per (token, slot), so the true count depends on **how many
distinct experts M tokens happened to route to**. That is data-dependent. No constant can express
it, and the old `M*(KS+1)` figure was 2.62x high at width 32.

### So the launch sites report what they actually read

- `dprof_bytes(b)` credits every mark that is **currently open**. Marks nest, so a parent row
  becomes the exact sum of its children for free, with nobody maintaining that relationship.
  (`dprof_end` now clears `g_open`, which it never did — the flag had meant "has ever been opened".)
- `gemv`/`gemm` report their own weight bytes, counting **one pass per chunk of the M loop**, at
  the site where the chunking actually happens rather than modelling it elsewhere.
- MoE reports from a **device counter**: `k_build_work` atomically accumulates its work-item count,
  and a flush hook converts it to bytes at report time. Reading it per layer would mean a stream
  sync per layer — the exact stall dprof exists to avoid. The hook takes a `credit` flag so
  `dprof_reset` discards a warm-up's work instead of billing it to the timed run.
- `k_absorb_q`/`k_expand_v` report `kv_b`, the MLA kernels report latent-cache passes, and the KDA
  recurrence reports its state — none of these are weights, so no gemm would have counted them.

A measured row prints `*`. A row with no measurement falls back to the old constant, which is
still right at width 1.

### The validation: at M=1 the old model was correct, so the new one must reproduce it

That is a comparison whose sign was known in advance (CLAUDE.md §6.4), and it is the only reason to
believe the counters. Decode, measured vs the hand-derived ROOFLINE §1 constants:

| row | modelled | measured | |
|---|---|---|---|
| `ffn:moe` | 5.401 | 5.4004 | 0.01% |
| `moe:w13+act` | 3.568 | 3.5673 | 0.02% |
| `moe:w2+combine` | 1.783 | 1.7836 | 0.03% |
| `moe:router` | 0.050 | 0.0495 | 1.0% |
| `lm_head` | 0.357 | 0.3568 | 0.06% |
| `kda:qkv+conv` | 1.927 | 1.925 | 0.1% |
| `ATTENTION` | 3.990 | 3.9247 | 1.6% |

An independent device counter landing on four significant figures of a figure derived by hand from
safetensors headers is as strong a cross-check as this repo has. Where they disagree slightly
(`kda:gates`, modelled 0.141 vs measured 0.065) the measurement is from the actual gemv shapes and
the constant was the guess.

### What the prefill table says now (width 32, prompt 2048)

| row | was | now |
|---|---|---|
| `ffn:moe` | 5.401 G/tok, **105%** | 2.061 G/tok, **39%** |
| `FFN` | 6.307, 105% | 2.126, 38% |
| `ATTENTION` | 3.990, 65% | 1.312, 23% |
| `lm_head` | 0.357, **11399%** | 0.0003, 53% |

Prefill is **weight-cheap** — that is what batching bought — and the time is going to activations
and occupancy, not to streaming weights. The old table hid that behind numbers over 100%.

### The impossible-row rule needed refining, not defending

`sdpa:context` now prints **204%**, and it is *correct*. That row is the MLA latent cache, which #15
established is L2-resident and measured at 487-610 GB/s against a 237 GB/s streaming read; `%BW`'s
denominator models DRAM and does not know about L2. So the rule now has two branches, told apart by
asking whether the tensor fits in L2 — and since weights never do, a weight-dominated row over 100%
is still a bug in the count. Rows over 100% are flagged `!` rather than left to look absurd.

Two build consequences: `gemv.cu` now references dprof, so `gate_indexer` and `gate_nvfp4` link
`kernels/dprof.cu` (they failed to link before this was noticed).

---

## 17. Gate 0 for the draft head: speculation cannot win on this architecture

The DFlash/MTP draft head was the stated goal. It is **blocked, and not by the head** — the same
verdict as the GGUF path reached, but for a different and more fundamental reason.

### The batch curve (45 layers, idle box, streaming probe 220.8 GB/s, min/med within 3%)

| K | min ms | vs K=1 | ceiling if every draft accepted |
|---|---|---|---|
| 1 | 84.22 | 1.000 | 1.00x |
| 2 | 182.23 | 2.164 | 0.92x |
| 3 | 185.12 | 2.198 | 1.36x |
| 4 | 222.00 | 2.636 | 1.52x |
| 8 | 419.16 | 4.977 | 1.61x |
| 16 | 798.19 | 9.477 | 1.69x |

With the measured 72.3% acceptance of the native un-fine-tuned MTP block, expected tokens per
verify is `(1-p^(K+1))/(1-p)` against a cost of `curve[K+1] + 0.08K`:

| draft depth | E[tokens] | cost | speedup |
|---|---|---|---|
| 1 | 1.72 | 2.24 | 0.77x |
| 2 | 2.25 | 2.36 | **0.95x** |
| 3 | 2.62 | 2.88 | 0.91x |
| 7 | 3.34 | 5.54 | 0.60x |

**Every depth is a loss.** And the payoff is boundable for ANY head, which settles whether a
fine-tune could rescue it: at 85% acceptance the best depth gives 1.02x, at 95% 1.13x, and a
*perfect* drafter caps at 1.27x. **No draft head is worth building against this curve.**

### Why: fine-grained MoE is structurally hostile to speculation

Profiling widths 1/2/4 separately (dprof rows, ratio to width 1; 0.25 would be perfect at width 4):

| row | w4/w1 | |
|---|---|---|
| `mla:absorb_q` | 0.25 | at floor (OPTIMIZATION_LOG #14) |
| `moe:router` | 0.34 | good |
| `mla:o_proj` | 0.43 | good |
| `attn:mla` | 0.46 | good |
| `attn:kda` | 0.54 | good |
| `moe:w13+act` | 0.75 | weak |
| **`ffn:moe`** | **0.82** | **flat, and ~45% of the step** |
| `moe:w2+combine` | 0.99 | flat |
| `mla:sdpa`, `mla:indexer` | 0.90, 0.96 | flat (inherently per-token) |

Attention batches. The MoE does not, and it dominates. That is not an implementation defect:
with 144 experts and top-8 routing, M tokens touch `144*(1-(1-8/144)^M)` DISTINCT experts —
8.0 per token at M=1, 7.8 at M=2, 7.4 at M=4, 6.6 at M=8, and only 5.4 at M=16. Expert traffic is
near-linear in M exactly where speculation needs it to be flat. The measured 0.82 is slightly
BETTER than this model's 0.86, so the expert-gathering kernel (#12) is already beating the naive
floor. **There is nothing left to win here.**

Note the direction REAP-50 pushed this: 144 experts saturate sooner than the original 288 would,
so pruning made speculation *less* bad, not more.

### The transferable rule

**Measure the serving-side batching curve before building or capturing for a draft head.** Twice
now — GGUF and NVFP4 — the head was fine and the target's batch behaviour was the blocker, and
both times the head was the thing that looked like the work. On GGUF the cause was
`op_offload_min_batch_size = 32`; here it is expert routing, which no amount of engineering
removes.

---

## 18. The NVFP4 accuracy call, settled with perplexity

The dense overlay (#13) halved `B_tok` and bought 1.42x on decode, and every gate said "close":
`gate_nvfp4` 12/12 at cos 0.9950-0.9961 per tensor, `gate_stack` cos 0.9972 over three layers.
None of that answers whether the model got worse, so the decision sat open. It is now measured.

`tools/perplexity.cu`, 32 sequences / **47,195 tokens** from the preserved capture corpus
(`artifacts/iq3m_mtp_capture/corpus`) — already tokenized with this model's tokenizer, so both
conditions index byte-identical input. Every sequence is under `DENSE_CTX_LIMIT`, so dense MLA is
exact and no DSA approximation can masquerade as a quantization effect. The only variable is
`GLM5_DENSE_NVFP4`.

| | PPL | mean NLL |
|---|---|---|
| bf16 (reference) | 4.430457 | 1.48850273 |
| NVFP4 overlay | 4.499673 | 1.50400474 |
| **delta** | **+0.069216 (+1.56%)** | **+0.01550** |

**The degradation is real, not noise.** Naive standard error on the mean per-token delta is
0.00176, so +0.0155 is **8.8 sigma**; and NVFP4 was worse at all four running checkpoints
(8/16/24/32 sequences), a clean 4-of-4 sign test. Tokens within a sequence are correlated so the
effective N is below 47,195, but the direction is not in question.

### Perplexity alone would have been the wrong number to stop at

| | |
|---|---|
| top-1 agreement | **90.101%** (4,672 of 47,195 tokens differ) |
| median per-token NLL delta | +0.000475 — essentially zero |
| p5 / p95 | -0.485 / +0.541 |
| tokens moving >0.1 nats | 42.3% |

So the *typical* token is unchanged, and the mean is a small positive drift on top of a wide,
nearly symmetric perturbation. A 10% top-1 flip rate looks alarming until you ask WHERE the flips
are: on disagreements the reference's median NLL is **2.2054** (it was giving its own top choice
about 11% probability), against **0.3547** where they agree. **Confident disagreements — reference
NLL < 0.5, i.e. the bf16 model was >60% sure — are 65 tokens, 0.138% of the total.** The overlay
reshuffles the model's coin-flips and leaves its convictions alone. At the shipped generation
config (temperature 1.0, top_p 0.95) those coin-flips were being sampled anyway.

### The call

**Keep the overlay on.** +1.56% perplexity for 1.42x decode throughput, with 99.86% of
high-confidence predictions preserved, is a good trade on a box where decode is the binding
constraint. It is not free, and this is the number to quote — not the 0.9972 cosine, which implied
a much smaller effect than 1.56% and 90% top-1.

`GLM5_DENSE_NVFP4=0` still reverts everything with no file touched. Untested and available if the
1.56% ever matters: reverting **`lm_head` alone** to bf16 (re-run `requant_dense_nvfp4.py` with a
shorter `--families`). It is the one converted tensor that sets the logits directly, so it likely
owns a disproportionate share of the top-1 flips, and it costs +0.912 G/token of `B_tok` — about
-8.5% decode — to put back. Whether that trade is better has not been measured.

---

## 19. Long-context correctness above `DENSE_CTX_LIMIT`, finally checked

This was owed for a long time. `gate_mla_sparse` asserts the sparse DSA path is **bit-identical**
to dense for all 2051 steps at or below the limit — but above it there is no dense answer to
compare against, so the regime the sparse path actually exists for was never validated. Shipped,
unverified.

The available check is **positional perplexity**: with more context a correct model must not get
worse, and a broken sparse path spikes exactly where the engine switches over. Known sign.
`tools/perplexity.cu --concat` glues corpus sequences into 6144-token streams and buckets NLL into
512-token bands.

6 streams, 36,858 tokens, NVFP4 overlay active:

| position | ppl | |
|---|---|---|
| 0–511 | 7.8121 | |
| 512–1023 | 7.3949 | |
| 1024–1535 | 4.7479 | |
| 1536–2047 | 4.5286 | dense MLA |
| **2048–2559** | **3.6925** | **switchover — DSA sparse from here** |
| 2560–3071 | 4.8354 | |
| 3072–3583 | 5.9126 | |
| 3584–4095 | 6.6543 | |
| 4096–4607 | 4.5799 | |
| 4608–5119 | 4.1412 | |
| 5120–5631 | 3.6866 | |
| 5632–6143 | **2.9068** | |

**No discontinuity at the boundary.** The switchover band is the *lowest* perplexity up to that
point, and the run ends at 2.91 — the model is using long context productively, which is the
opposite of what a broken indexer or a mis-gathered KV would produce. The bump at 3072–4095 is
content, not position: concatenated streams glue unrelated documents together, so a document
boundary landing in a band raises it. Position bands are the same width so those are comparable.

**What this does and does not establish.** It establishes that the sparse path is not broken and
that long context helps. It does not establish bit-exactness against a dense reference, which is
not computable above the limit — that is why the check is statistical. Combined with
`gate_mla_sparse`'s bit-exact equality below the limit, the sparse path is now covered on both
sides of the switchover.

---

## 20. The vision tower

The checkpoint ships a complete 24-block vision tower — 347 tensors under `model.visual.` — and
this server never loaded one byte of it. The GGUF build has vision; this did not. Dropping vision
capability is on the operator's escalate-first list, so a text-only server was a capability gap,
not a missing optimisation.

Built, and gated against `transformers`' own `Glm5NextVisionModel`.

### It is the one part of this checkpoint that PyTorch can load

Every other oracle in `ref/` streams a layer at a time, because the full model is not instantiable
here. The tower is the exception: ~0.5 B params in plain bf16, 1.17 GiB, so it loads whole and the
gate compares against transformers proper rather than against a reimplementation of it.

### Shape

One 448x448 image is 32x32 patches of 14 -> **1024 rows of 1176** (`channels x temporal_patch x
patch x patch`), through patch_embed to 1024-wide, 24 blocks, post_layernorm, a 2x2 spatial
downsample to **256 rows of 4096**, then the merger. Those 256 rows are what splice into the
language model at the image-token positions.

Two things that look like convolutions and are not: `patch_embed` is a `Conv3d` whose kernel
EQUALS its stride, so it is a plain gemm over flattened patches; the downsample is a `Conv2d` in
the same position, so it is a gather of each 2x2 neighbourhood followed by a gemm.

### Four ways it differs from the language model, all of which fail quietly

- MLPs use a **clamped** SwiGLU — gate clamped above at 10.0, up clamped both ways.
- The merger uses **LayerNorm with bias** and a **GELU**, not RMSNorm and SiLU.
- `q_norm`/`k_norm` are RMSNorm over `head_dim=64`, per head, **before** rope.
- Attention is **bidirectional**. No causal mask anywhere in the tower.

### The bug, and the metric that hid a non-bug

`k_rmsnorm_heads` launched **64 threads — two warps — and reduced with `__shfl_down_sync`, which
is warp-local.** Half the sum of squares was silently dropped, rescaling every head by about
sqrt(2). The tower came back at cos 0.690: wrong, but far enough from zero to look like a numerics
problem rather than a bug. It is now launched with exactly one warp.

Then, with that fixed, the gate still failed — and the failure was the **oracle**. The reference
ran the whole tower in bf16 (`model.to(torch.bfloat16)`), while this engine keeps fp32 activations
with bf16 weights, so the reference is the LESS precise of the two and its rounding compounds
across 24 blocks:

| stage | vs bf16 oracle | vs fp32 oracle |
|---|---|---|
| patch_embed | cos 0.999998578 | **cos 1.000000000**, relL2 4.34e-07 |
| block 0 | cos 0.999994162 | **cos 1.000000000**, relL2 7.36e-07 |
| all 24 blocks | cos 0.997160027 | **cos 1.000000000**, relL2 1.07e-05 |
| merged | cos 0.997822164 | **cos 1.000000000**, relL2 1.19e-05 |
| position ids | — | exact, relL2 0.0 |
| rope cos / sin | — | relL2 1.80e-08 / 1.70e-08 |

`GLM5_VIS_DTYPE=fp32` regenerates the oracle upcast, and the gate defaults to it. **A gate can
fail because the reference is wrong, and "compare against something more precise than the thing
under test" is not automatic when the reference is a bf16 model.** Reported with relative L2 rather
than max-per-element, for the reason established in #16.

Position ids are **not raster order** — patches are visited in 2x2 blocks so that the rope
agrees with what the downsample later folds together. Getting that wrong scrambles an image
spatially while still producing fluent captions, which is why it is gated separately from the
tower rather than folded into one number.

### What is not done

The tower is an encoder with a gate. Wiring it to the request path — image decode and
preprocessing to 1176-wide patch rows, splicing the 256 embeddings at `image_token_id` (154854),
and the text-side mrope positions — is not built. `vision_forward` is the hard, verifiable half.

### 20b. The rest of the vision path

`vision_forward` was the encoder; this is what feeds it and what consumes it.

**Preprocessing, Python-free** (`src/vision_preproc.cpp`, `tests/gate_vision_preproc.cpp`).
Decode (stb_image) -> `smart_resize` -> content fit -> zero pad -> rescale 1/255 -> CLIP normalize
-> patchify. Gated against `Glm5NextImageProcessor` on two real images:

| | small (233x311) | large (1701x2203) |
|---|---|---|
| canvas | 252x336, matches | 1708x2212, matches |
| grid | 18x24 = 432 patches | 122x158 = 19,276 patches |
| patchify | relL2 1.32e-07 | 1.49e-07 |
| full path from PNG bytes | **relL2 1.13e-07** | **1.28e-07** |

Both halves are gated separately on purpose: patchify is exact rearrangement, while an antialiased
bicubic is implementation-defined, so a single number could not say which moved. In the event
neither moved — `max_abs` is **4.768e-07 on every check, which is exactly 2^-21, one fp32 ulp**.
The first run failed at a 1e-7 threshold, and the failure was the threshold: the oracle's patches
come from the processor's normalize and the reference pixels from this script's, and two different
float32 op orders cannot agree more closely than an ulp. Demanding bit-equality across them is a
gate that can only fail.

Note that both test images take the **pad** path — the token budget does not bite until ~12.5 M
pixels — so the bicubic downscale branch is implemented but only lightly exercised. Said here
rather than left for someone to discover.

**Splicing** (`Engine::set_image_embeds`). The 256 rows per image replace the token embedding at
their absolute positions, broadcast into all four hyper-connection streams exactly as
`k_embed_broadcast` does.

**There is no mrope, and that is not an omission.** This model's language side is pure NoPE —
`qk_rope_head_dim == 0`, `mla_use_nope == true`, no rotary anywhere in its attention — so an image
contributes nothing to position encoding beyond occupying n consecutive slots, and splicing is
exactly an embedding swap. Position reaches the language model through the 34 KDA layers. Every
other VLM needs 3D mrope here; this one does not.

Still open: an HTTP surface that accepts an image. The pieces below it are gated.


### 20c. The vision HTTP surface, and three bugs that each looked like something else

`POST /v1/chat/completions` now accepts OpenAI `image_url` content parts. End to end on the test
image (red horizontal gradient + green vertical gradient + blue checkerboard), the model's own
reasoning reads *"It's a checkerboard pattern (alternating squares). The colors vary across the
image, forming a gradient."* — 134 prompt tokens (108 image + text), 12.06 tok/s.

`data:` URIs only. **A plain http(s) URL is refused deliberately**: making an inference server
fetch arbitrary URLs for a caller is server-side request forgery, and a box holding a 100 GiB
checkpoint on someone's LAN is not where to add one.

The chat encoder already emitted one `<|image|>` per image part; `expand_image_tokens` replaces
each with the N the tower produced and records where, and `Engine::set_image_embeds` splices there.
`max_image_tokens` defaults to 1024, far below the processor's 8000, because `k_vis_attn` holds
the score row in shared memory — the processor's default would allow 32k patches and 128 KB of
shared, which does not launch.

**Three bugs, and none of them presented as itself.**

1. **`hasVision()` read a lazily-set flag**, so it was false until the first image had already been
   encoded — every image request was rejected with "no vision tower resident" on a server that had
   1.17 GiB of vision weights loaded. Now probed at construction.
2. **A shared size counter meant a buffer was never allocated.** `vis_cos_` and `vis_sin_` shared
   `vis_rope_n_`; after cos grew, the sin allocation saw the size already satisfied and returned
   without allocating, so the memcpy ran against a null pointer. Presented as
   `cuda invalid argument`, three frames away from the cause.
3. **A deadlock that answered `/metrics` cheerfully.** The image path took `g_lock` with a
   `unique_lock` and never released it; generation then took the same non-recursive mutex and the
   request hung forever while the metrics endpoint — which does not take the lock — kept replying
   `requests_total 0`. A health check would have said the server was fine.

**And one that produced a fluent, plausible, wrong answer:** `reset()` cleared `img_spans_`, and
`generate()` calls `reset()` for any request that cannot reuse the resident prefix — which is every
first turn. The embeddings were wiped microseconds after being registered, and the model reported,
politely and in perfect prose, that no image was attached. **That is the failure mode to fear here:
not a crash, but a coherent answer to a question the model was never actually shown.** The spans are
request-scoped and the caller owns them; `reset()` no longer touches them.

Lifetime note worth keeping: the SSE path hands generation to a chunked content provider that
captures by value and runs *after* the handler returns, so the image buffers are held by a
`shared_ptr` both paths share rather than a scope guard, which would have freed them mid-stream.
