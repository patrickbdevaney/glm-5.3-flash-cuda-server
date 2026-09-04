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
