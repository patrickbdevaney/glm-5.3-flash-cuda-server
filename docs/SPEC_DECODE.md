# Speculative decode: the design, and the one thing that makes it hard here

Nothing in this file is implemented yet. It exists because the pieces it depends on are now built
and gated, and because two of the findings below change what should be built — recording them
before writing the code is cheaper than discovering them inside it.

## Why this is worth doing at all

The draft head is already good: **72.26% depth-1 acceptance, un-fine-tuned**, measured on this
checkpoint (`glm53-nvfp4-mtp-gate2`). Speculation still *lost* on the GGUF build, for a reason that
had nothing to do with the head — llama.cpp's batch cost is flat below 32 tokens, so a depth-K
draft always landed on the slow side of the cliff. Here the kernels are ours. `forward_batch` is
built and gated bit-exact against the sequential path, so the cliff is gone by construction.

## The MTP block, from the checkpoint

Layer 45. Structurally an ordinary decoder layer plus three tensors, and **no `hc_*` at all** — it
runs a plain residual, not the four-stream mHC the backbone uses.

| tensor | shape |
|---|---|
| `enorm.weight`, `hnorm.weight` | [4096] bf16 |
| `eh_proj.weight` | [4096, 8192] bf16 |
| `self_attn.*` | MLA + a DSA indexer, same shapes as layers 3..43 |
| `mlp.*` | 144 routed NVFP4 experts + 1 shared, same as the backbone |
| `shared_head.norm.weight` | [4096] bf16 |
| lm_head | **shared with the main model**, not its own |

The forward, taken from the oracle that produced the 72.26% figure
(`~/glm-5.3-reap/scripts/nvfp4_mtp_gate.py`) rather than from `transformers`, which drops layer 45
on load and has no implementation to read:

```
x = concat( rms(embed[next_token], enorm), rms(h_prev, hnorm) ) @ eh_proj^T
x = x + self_attn(input_layernorm(x))
x = x + mlp(post_attention_layernorm(x))
draft_logits = rms(x, shared_head.norm) @ lm_head^T
```

**Open question, and it is not cosmetic.** In that oracle `h_prev` is the hidden state *after* the
final RMSNorm — the vector that feeds `lm_head` — because that is what the llama.cpp trace
extraction captured. DeepSeek's MTP feeds the state from *before* the final norm. 72.26% acceptance
says post-norm is at least workable, but it does not say it is what the block was trained on, and
the difference is one `rmsnorm` call. The engine has both vectors in hand at that point
(`pooled_` before and after the norm), so this is a two-line A/B once the full model loads. It
should be run before any fine-tuning, because fine-tuning against the wrong input would bake the
mistake in and still look like it was working.

## The hard part: a recurrence cannot be rewound

This is the finding that shapes the implementation.

A verify step runs `forward_batch` over K drafted tokens. If only j < K are accepted, the engine
must return to the state at position j. For an ordinary transformer that is free — you truncate the
KV cache. **34 of 45 layers here are KDA, which is a recurrence**: token m's state is token m−1's
output, in place. There is nothing to truncate and no inverse to apply.

Three ways out, and the third is nearly free:

1. **Snapshot before, re-run after.** Copy the 145.56 MiB state, restore it on partial rejection,
   re-run `forward_batch` over the j accepted tokens. The copy is cheap (~0.7 ms); the *second
   forward* is not — it pays the full weight read again and cancels the entire win.
2. **Snapshot per position.** Keep S₀..S_K, then "accept j" is a pointer move. Costs K extra state
   copies, ~2% of a 4-wide forward.
3. **Ping-pong the recurrence across slots.** Have the recurrence kernel *read* slot m and *write*
   slot m+1 instead of updating in place. Identical bandwidth — same volume read, same volume
   written, just to a different address — so the snapshots cost **nothing at all**. Memory is
   (K+1) × 145.56 MiB: 728 MiB at K=4, 1.31 GiB at K=8.

Option 3 is the design. It needs `k_recurrence` and `k_conv_silu` to take separate in/out state
pointers, which is a change of two signatures and no arithmetic; the existing in-place behaviour is
exactly the case where both pointers are equal, so the current gates stay valid.

The MLA latent cache needs none of this — rewinding it is just moving the position back. And the
MTP block's own state is a latent cache only, since it has no KDA.

## What it should actually be worth

From ROOFLINE §4, corrected for the cost the table there explicitly excluded: the draft head's own
forward, which is one MLA+MoE layer (0.371 G) plus `eh_proj` (0.067 G) plus **`lm_head`, at 1.269 G
by far the largest term**, per drafted token.

| depth | verify | draft (bf16 head) | total | tokens | **speedup** |
|---|---|---|---|---|---|
| 1 | 1.227 | 0.086 | 1.314 | 1.72 | 1.31x |
| 2 | 1.442 | 0.173 | 1.615 | 2.24 | **1.39x** |
| 3 | 1.645 | 0.259 | 1.904 | 2.62 | 1.38x |
| 4 | 1.836 | 0.345 | 2.182 | 2.89 | 1.32x |

**The draft head's `lm_head` should be NVFP4, and that is system-level lossless.** A draft error is
not an output error — it costs a rejection, which the verify step catches by construction. So the
one place in this model where reduced precision cannot hurt quality is precisely its largest single
draft cost. Quantising it cuts the draft from 1.707 G to 0.755 G per token:

| depth | total | tokens | **speedup** |
|---|---|---|---|
| 1 | 1.266 | 1.72 | 1.36x |
| 2 | 1.518 | 2.24 | 1.48x |
| **3** | **1.759** | **2.62** | **1.49x** |
| 4 | 1.989 | 2.89 | 1.45x |

So: **depth 2–3, expect ~1.4x with a bf16 draft head and ~1.5x with an NVFP4 one**, before any
fine-tuning of the head. The optimum is flat across depths 2–4, so tuning the depth is not worth a
session; quantising the draft `lm_head` is.

Both tables assume independent expert routing across the drafted tokens, which is the pessimistic
end — adjacent tokens share more experts than chance, so the verify column is if anything cheaper
than shown.

## Build order

1. In/out state pointers on `k_recurrence` and `k_conv_silu`; (K+1) state slots in the engine.
   Gate: unchanged results when in == out, which the existing KDA and batch gates already assert.
2. Load layer 45; implement the draft step above. Gate against `nvfp4_mtp_gate.py`'s own numbers on
   the same sequences — the acceptance rate is the gate.
3. Resolve the pre-norm / post-norm question for `h_prev` before anything else is tuned.
4. The draft-verify loop, with the accept rule matching the target's argmax.
5. NVFP4 the draft `lm_head`.

Steps 2–4 need the full 45-layer load (~98 GiB) and are blocked on the box until the trace
extraction finishes. Step 1 is not blocked.
