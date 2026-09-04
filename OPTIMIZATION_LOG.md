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
