# ARCH_DELTA.md — what ports, what is new, and from where

## 0. The honest framing

This box already holds a **188k-line pure-CUDA engine for a model with the same DNA**:
`~/deepseek-v4-flash-0731-cuda` (DeepSeek-V4-Flash-0731-REAP). It runs MLA, a DSA lightning
indexer, manifold-constrained hyper-connections with Sinkhorn, NVFP4 MoE and an embedded
multi-token-prediction head — every one of them gated bit-exact or cosine-1.0 against a PyTorch
oracle, on this same Thor.

GLM-5.3-Flash shares four of those five subsystems, **including the hyper-connection constants
exactly** (`hc_mult` 4, `hc_sinkhorn_iters` 20, `fn` `[24, 16384]`, `base` `[24]`, `scale` `[3]`).

**So this is not a from-scratch kernel build. It is one new attention family, three retargets,
and a server.** The single genuinely new surface is **KDA (Kimi Delta Attention)**, which is 34
of 45 layers and — per `ROOFLINE.md` §1 — 47.4% of per-token bandwidth. That is where the
engineering goes.

---

## 1. Model geometry (read from the checkpoint, not the config)

`~/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2` — 57 604 tensors, 98.15 GiB, 62 shards.

| | |
|---|---|
| hidden | 4096, vocab 154 880, `tie_word_embeddings` false |
| layers | **45** = 34 KDA linear + 11 MLA/DSA full, **+ 1 MTP block at index 45** |
| full-attn layers | 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43 (every 4th) |
| KDA | 64 heads x 128, short conv k=4, `gate_lower_bound` -5.0, low-rank (128) f/g gates |
| MLA | `q_lora` 1536, `kv_lora` 512, `qk_nope` 256, **`qk_rope` 0 — pure NoPE**, `v_head` 256, 64 heads |
| DSA indexer | 32 heads x 128, `index_topk` **2048**, `kpool` 4 + compress, rope-interleave |
| MoE | 144 routed (REAP-50), 8/tok, 1 shared, inter 2048, **sigmoid** + `noaux_tc`, scale 2.5, `norm_topk_prob` |
| dense layers | 0, 1, 2 (`first_k_dense_replace` 3), inter 12288, swiglu clamp 10.0 |
| mHC | `hc_mult` 4, sinkhorn 20, eps 1e-6, on all 45 backbone layers (**not** on MTP) |
| vision | 24-block ViT, hidden 1024, patch 14, img 448 → 4096, **present and preserved** |
| quantised | **only MoE experts + shared experts** are NVFP4; everything else is bf16 |

**NoPE is a real simplification.** `qk_rope_head_dim = 0` and `mla_use_nope = true` mean the
full-attention layers apply **no rotary embedding at all** — no YaRN, no rope cache, no
interleave in the main attention path. Position information reaches those layers only through
the 34 KDA layers below them (and the indexer's own rope). The 0731 engine's `yarn.h` and the
rope half of `mla_attn.cu` are simply **not needed**.

---

## 2. Port verdict, per subsystem

**PORT** = existing gated CUDA applies directly · **RETARGET** = existing kernel, new shapes ·
**NEW** = write from scratch.

| subsystem | source | verdict | note |
|---|---|---|---|
| Hyper-connections + Sinkhorn | `0731 kernels/hc.cu`, `hc_sinkhorn.cu` | **PORT** | constants identical; `fn [24,16384]`, hcd 16384 the same. Only delta: GLM's final `HyperHead` is an **unweighted mean**, DSV4's was weighted. |
| safetensors mmap loader | `0731 include/safetensors.h` | **PORT** | same format, same 8-byte-offset alignment trap |
| weight store / resident planner | `0731 include/weight_store.h` | **PORT** | |
| NVFP4 dense GEMM/GEMV | `0731 kernels/nvfp4_dense.cu`, `fp4_gemm.cu`, `cutlass_moe.cu` | **PORT** | same `weight_packed` U8 + `weight_scale` fp8-e4m3 per-16 + `weight_global_scale` layout, verified bit-identical by `glm-5.3-reap/scripts/nvfp4_dequant_check.py` |
| MoE router + dispatch | `0731 kernels/moe.cu`, `tc_moe_gemm.cu` | **RETARGET** | 144/8/1 vs 160/6/1; **scoring `sigmoid` not `sqrtsoftplus`**; `norm_topk_prob` on; `routed_scaling_factor` 2.5 |
| MLA attention | `0731 kernels/mla_attn.cu`, `mla_decode.cu` | **RETARGET** | new shapes; **drop rope entirely** (NoPE); no `o_lora`/`o_groups` in GLM |
| DSA lightning indexer | `0731 kernels/indexer.cu` | **RETARGET** | 32 heads not 64, topk **2048** not 512, and **k-pooling (`kpool` 4 + `compress_gate`/`ape`) is new** |
| top-k selection | `0731 include/topk_radix.h` | **PORT** | topk 2048 of N is the same radix problem |
| HTTP / OpenAI API / SSE / webui | `0731 include/openai_api.h`, `stream_parse.h`, `webui.h`, `server/` | **PORT** | model-agnostic |
| tokenizer | `0731 include/tokenizer.h` (engine) | **RETARGET** | GLM vocab 154 880, 3 EOS ids `[154820, 154827, 154829]`, new merge tables |
| bandwidth probe / profiler | `0731 tools/bw_probe.cu`, `include/dprof.h` | **PORT** | |
| **KDA linear attention** | — | **NEW** | 34 layers, 47.4% of `B_tok`. §3. |
| **KDA recurrent state cache** | — | **NEW** | 145.56 MiB, context-flat. §3. |
| MTP block (layer 45) + spec decode | `0731 src/draft.cu` (shape only) | **RETARGET** | GLM's MTP is one ordinary DSA+MoE layer + `enorm`/`hnorm`/`eh_proj`, **no hyper-connections**, output pre-`shared_head.norm`. Far simpler than DSV4's 3-stage DSpark chain. |
| vision tower | — | **NEW (deferred)** | 24-block ViT; text AR path does not touch it |

---

## 3. The new surface: KDA

Reference: `transformers/models/glm5_next/modeling_glm5_next.py`,
`Glm5NextTextLinearAttention` + `Glm5NextTextForgetGate` + `recurrent_kimi_delta_attention`.

Per layer, per token, 64 independent heads of `k_dim = v_dim = 128`:

```
qkv    = concat(q_proj, k_proj, v_proj) @ x                   [3 * 8192]
qkv    = silu(depthwise_conv1d_k4(qkv, conv_state))           per channel, rolling window
q,k,v  = split(qkv) -> [64, 128] each
g      = -5.0 * sigmoid(exp(A_log)[h] * (f_b(f_a(x)) + dt_bias))   [64, 128]   in (-5, 0)
beta   = sigmoid(b_proj(x))                                   [64]
q,k    = l2norm(.)   (sqrt(sum + 1e-6), FLA convention, NOT max(.,eps))
q     *= 1/sqrt(128)

per head, state S [128 k, 128 v] fp32:
   S      = S * exp(g)[:, None]        # decay per (head, k-dim), broadcast over v
   kv_mem = k^T S                      # [128 v]
   delta  = (v - kv_mem) * beta        # [128 v]
   S     += outer(k, delta)
   out    = q^T S                      # [128 v]

gate   = g_b(g_a(x))                   [64, 128]
out    = rmsnorm_fp32(out, o_norm.w) * sigmoid(gate)
out    = o_proj(out)
```

**Checkpoint splits the fused conv1d** into `q_conv1d`/`k_conv1d`/`v_conv1d`, each `[8192, 1, 4]`,
bias-free; the reference has one `[24576, 1, 4]`. Concatenate on load.

**Why this maps well to CUDA.** One block per (layer, head): 64 KiB of fp32 state, read once and
written once, three `[128]`-vector ops and two rank-1 updates. No cross-head communication, no
context-length dependence, no KV scan. It is a bandwidth problem with a tiny arithmetic core —
which is exactly the regime where the 0731 grind showed the difference between 37% and 80% of
achievable bandwidth.

**Numerical note.** Upstream computes the recurrence in fp32 deliberately ("states are more
susceptible to rounding errors"). We keep fp32 state. `ROOFLINE.md` §4 shows bf16 state would
save 0.7% of `B_tok` — not worth it.

---

## 4. What is deliberately NOT built

- **YaRN / rope for main attention** — `qk_rope_head_dim = 0`. Does not exist in this model.
- **KV compressor (ratio 4/128)** — 0731's `compressor.cu` serves a model with `compress_ratios`;
  GLM-5.3 has no such config key. MLA latent KV is 88 MiB at 8k context; there is nothing to compress.
- **FP8 block GEMM** — this checkpoint has **zero** FP8 weight blocks (the `F8_E4M3` tensors are
  NVFP4 *scales*, 18 705 of them, not weights).
- **3-stage DSpark head** — GLM's MTP is a single ordinary layer.
- **Vision tower** — deferred, not dropped. Weights stay resident-capable; the text AR path
  simply never reads them.

---

## DSA indexer — semantics read from the module, and a correction to the dense limit

Previously recorded as "dense MLA is exact below 2048". **That is off by three.** Measured against
the real `Glm5NextTextIndexer` in `ref/gen_indexer.py`:

| context | pools | select_k | tail | distinct tokens the last query sees | |
|---|---|---|---|---|---|
| 2048 | 512 | 512 | 0 | 2048 of 2048 | dense |
| 2050 | **513** | 512 | 2 | 2050 of 2050 | **still dense** |
| 2051 | 513 | 512 | 3 | 2051 of 2051 | **still dense** |
| 2052 | 513 | 512 | 0 | 2048 of 2052 | first sparse length |

A trailing **incomplete** pool is never selectable — `pool_valid` requires all `index_kpool` tokens
to be present — but `append_visible_tail` appends its tokens raw anyway. So 513 pools with only 512
selected is still complete coverage whenever the 513th is a fragment. The limit is
`floor(T / index_kpool) <= index_topk / index_kpool`, i.e. **`DENSE_CTX_LIMIT = IDX_TOPK +
IDX_KPOOL - 1 = 2051`**, and the engine's guard now uses that rather than 2048.

Three more facts a kernel will need, none of them guessable from the config:

- Output width is **constant at 2051** (`index_topk + index_kpool - 1`), at every context length,
  padded with -1.
- Pooling starts at the **first valid key**, not at slot 0, so with left padding the pool grid is
  offset. At decode with no padding that offset is zero.
- `k_norm` is a **LayerNorm with a bias**, eps 1e-6 — not the RMSNorm used everywhere else in this
  model. Reusing the model's own norm here would be wrong in a way that still produces plausible
  scores.

Pool keys are a softmax-weighted average over each complete pool of
(`index_kpool_compress_gate @ hidden` + `index_kpool_compress_ape`), so **they are fixed once a
pool's 4 tokens exist** — the kernel should compute each pool key once on completion and cache it,
not recompute 512 of them per decode step.
