# A local daily-driver for agentic coding — Thor selection & measurement plan

*Working doc. Goal: the local model that runs my agentic coding loops and recovers most of
Claude Opus's usefulness (Fable/Astra being intractable locally), on **Thor** (Jetson AGX
Thor, 128 GB unified LPDDR5X), clearing four gates at once: usable speed-in-loop, tool-call
reliability, long-context coherence, and frontier reasoning/knowledge.*

**Revision 2 — fortified with real specs, evals, decode physics, and REAP data (web,
Sept 2026).** Numbers now come from published sources (cited in §10); items still unknown
are tagged `[verify]`. The framework and measurement harness (§6) are unchanged in spirit:
*your loop is the oracle; a model card is a claim.* But the quantitative grounding flips one
of the earlier conclusions — see §0.

---

## 0. The bet, and the one fact that reshapes it

Original bet: reap a bigger, higher-capability base **down to ~180 B total** so it fits
Thor, wagering that reaped experts from a deep base beat a native-180 B model.

**The fact that changes the ranking:** on Thor, decode is **memory-bandwidth-bound at
273 GB/s**, and **REAP prunes *total* experts (memory) while leaving *active* params per
token unchanged (speed).** MiniMax-M2 reaped 230 B→172 B stays **A10B active**;
Qwen3-Coder reaped 480 B→246 B stays **A35B active**. So:

- **Memory fit** is set by *total* params × quant → decides *if it loads*.
- **Decode speed** is set by *active* params × quant / bandwidth → decides *if it's usable*,
  and **reap does not help it.**

Consequence, scoped precisely: reaping the **full** 745 B / A44B GLM-5.3 down to a Thor-fitting
size is (a) a **~76 % prune, past REAP's validated ≤50 % frontier → heal-required**, and (b)
*still* A44B active → ~10 tok/s. But your **GLM-5.3-Flash is a distinct ~321 B base**, and
reaping *it* **50 % → ~160 B** is a *different animal*: it fits Thor and sits **inside** the
validated near-lossless zone. Its only cost is the **~3× active params** vs native-180 B Qwen —
a **deliberate, priced trade**: accept slower decode to keep deeper experts from a frontier-
lineage base firing more per forward pass. That trade is the real thesis of this doc, and it's
sound (§4.5). The practical decision is a **4-way** (§4): **Qwen3.8-Flash-Next** (no reap, fast
floor) · **GLM-5.3-Flash-reap-50** (validated, the heavy — your bet) · **GLM-5.3 progressive
reap/heal** (ceiling moonshot, heal-required) · **DeepSeek-V4.1-Flash progressive reap/heal**
(long-context specialist, heal-required). Full reasoning in §4.5.

---

## 1. The box everything must fit in (Thor) — measured

| Resource | Figure | Source |
|---|---|---|
| Total unified memory | 128 GB LPDDR5X | NVIDIA / ServeTheHome |
| Memory bandwidth | **273 GB/s** (256-bit @ 4266 MHz) | NVIDIA spec; confirmed on GB10 |
| Realized bandwidth (MBU) | **~85 %** → **~232 GB/s effective** | Measured on DGX Spark GB10 (12.32 of 14.54 tok/s theoretical) |
| Usable memory for weights+KV+draft | **~108–112 GB** `[verify on your OS image]` | ~16–18 GB reserved for OS/runtime/harness |
| FP4 compute | ~2070 FP4 TFLOPS | Yahboom/NVIDIA — irrelevant to decode (bandwidth-bound), matters for prefill + spec-decode verification |

**Thor ≡ DGX Spark for LLM decode.** Both are GB10-class, 128 GB LPDDR5X, **273 GB/s**. Every
Spark decode benchmark transfers to Thor directly — a large, growing body of real data (§10).

### 1.1 Memory sizing — the *total*-param / fit axis

Weight bytes ≈ `total_params × bpw / 8`. Usable ≈ 110 GB. At ~4-bit (0.5 GB per 1 B params):

| Candidate (total after reap) | @4-bit weights | Fits 110 GB w/ KV+draft? |
|---|---|---|
| Qwen3.8-Flash-Next ~180 B (no reap) | 90 GB | ✅ ~20 GB left |
| **GLM-5.3-Flash 321 B → reap-50 → ~160 B** | **80 GB** | ✅ ~30 GB left |
| GLM-5.3 full 745 B → reap/heal ~160–180 B | 80–90 GB | ✅ **only post-heal** |
| DeepSeek-V4.1-Flash ≫284 B → reap/heal to fit | `[verify]` | ✅ only post-heal |

**Reap-ratio (the quality-risk axis):** **GLM-5.3-Flash 321 B→160 ≈ 50 %** (validated
near-lossless, §3) — the low-risk fit. **GLM-5.3 full 745 B→~170 ≈ 77 %** ⚠ and
**DeepSeek-V4.1-Flash** (≫284 B → fit) ⚠ are **past the validated ≤50 % frontier → heal-required**,
with real, uneven quality risk. Qwen needs no reap. This is exactly the low-risk / high-risk
split that structures the 4-way (§4).

### 1.2 Speed sizing — the *active*-param / usable axis (the one reap can't fix)

```
decode tok/s  ≈  232 GB/s (effective)  /  (active_params × bpw/8  +  KV read/token)
```

Weight-dominated estimate at ~4-bit, before speculative decoding:

| Active params | Example model | GB/token | **base tok/s** |
|---|---|---|---|
| ~3 B | Qwen3-Next / Qwen3.8-Flash-Next | 1.5 | **~155** |
| **10 B** | **MiniMax-M2-REAP-172B-A10B** | 5.0 | **~46** |
| 13 B | DeepSeek-V4-Flash | 6.5 | **~36** |
| 21 B | Hunyuan Hy3 | 10.5 | **~22** |
| 32 B | GLM-4.6 / GLM-4.7-REAP | 16.0 | **~14** |
| 44 B | GLM-5 / GLM-5.3 (reaped) | 22.0 | **~10.5** |
| 49 B | Hunyuan Hy4-preview | 24.5 | **~9.5** |

**Speculative decoding — real GB10 coding numbers (this is where your use case wins):** on
DGX Spark, a 27 B model went **12.32 → 65.02 tok/s (5.28×)** on *coding* workloads with a
well-tuned drafter (DFlash2: 12-token draft budget, 7.48 avg acceptance length). MTP/EAGLE-
style gave 41.57 tok/s (~3.4×); prose/chat only ~26–28 tok/s (~2.1–2.3×). **Coding gets the
biggest speedup because code is the most predictable → longest accepted drafts.** DeepSeek's
own MTP reports **85–90 % first-token acceptance → ~1.8×** as a conservative floor.

Applying a conservative **×2–3 (up to ×5 best-case coding)** to the base column:

| Model (active) | base | coding w/ spec-decode | verdict on Gate 1 (≥~20 t/s) |
|---|---|---|---|
| Qwen3.8-Flash-Next (A3B) | 155 | (bandwidth-capped, >>50) | ✅✅ blazing |
| MiniMax-M2-REAP (A10B) | 46 | **~90–140** | ✅✅ |
| DeepSeek-V4-Flash (A13B) | 36 | **~70–120** | ✅✅ |
| Hunyuan Hy3 (A21B) | 22 | ~45–70 | ✅ |
| GLM-4.x-REAP (A32B) | 14 | ~28–45 | ✅ (marginal without spec-decode) |
| GLM-5.3 reaped (A44B) | 10.5 | ~21–35 | ⚠ needs aggressive spec-decode to clear the floor |
| Hunyuan Hy4 (A49B) | 9.5 | ~19–32 | ⚠ borderline, and doesn't fit memory anyway |

`[verify]`: exact per-token KV read at your context length (subtract from the 232 GB/s — MLA
and linear/hybrid attention make this small; full-MHA at 100 k makes it large — §5), and your
drafter's real acceptance length on *your* code.

---

## 2. The four gates (your hard floor, made measurable)

Out if it fails any one. Mapped to harness legs in §6.

1. **Speed-in-loop** — sustains ≥~20 t/s decode *with* spec-decode at working context, prefill
   doesn't stall. **Governed by active params (§1.2), not reap.**
2. **Tool-call / structured-output reliability** across a *long* multi-step loop — the
   capability REAP is most likely to nick (rare-format precision). Adversarial long-horizon
   test, not one-shot.
3. **Long-context coherence** (target ≥100 k `[verify your real ceiling]`) at a KV cost that
   fits Thor — decided by attention family (§5): MLA / DSA / GDA / KDA.
4. **Frontier reasoning + world knowledge + novel synthesis** — the capability wide-MoE param
   count buys, and what the reap gambit bets it preserves.

---

## 3. The reap gambit — with the real degradation curve

**REAP** (Router-weighted Expert Activation Pruning, Cerebras, **ICLR 2026**, arXiv 2510.13999):
one-shot prune of whole experts by a saliency criterion combining router gate-values and
expert activation norms; router kept intact (unlike merging, which blurs specializations).

**Published degradation (coding, generative):**
- **~25 % prune: near-lossless** — ~0.2–1.9 % mean accuracy drop.
- **~50 % prune: still strong** — ~1.2 % up to ~6.9 % mean drop depending on task; **<2 % even
  at 50 %** on the big fine-grained-expert models (Qwen3-Coder-480B, Kimi-K2 1T) for code-gen
  and tool-calling.
- **>50 %: off the published map.** Degradation climbs and is uneven; this is where your
  "progressive reap/heal 75 %" lives. It requires **healing (continued-pretrain/distill)** to
  be viable, and heal quality is unproven per-model — treat 75 % as a research bet, not a
  known-good.

**Two things the curve doesn't show, and both bite on Thor:**
1. **Reap cuts total, not active** (§0/§1.2). A 76 %-reaped GLM-5 is still A44B → ~10 t/s.
   You pay the quality risk of extreme pruning *and* keep the slow decode. Worst of both.
2. **Degradation is capability-uneven** — it protects aggregate benchmarks while nicking the
   long tail: rare-API coding, exact tool-format, multi-step state-tracking (Gates 2 & 4).
   This is the manuk lesson exactly: *an umbrella metric can't go red*. Measure the specific
   capabilities pruning steals, on your corpus (§6) — never an aggregate.

**Concrete existing REAP checkpoints (proof the method ships, not just a paper):**
`cerebras/MiniMax-M2-REAP-172B-A10B` and `-162B-A10B` (25–30 %, near-lossless) ·
`cerebras/Qwen3-Coder-REAP-246B-A35B-FP8` (~49 %) · `cerebras/GLM-4.7-REAP-268B-A32B-FP8` ·
`cerebras/Kimi-Linear-REAP-35B-A3B` (KDA). Community NVFP4-**GB10** quants exist
(`saricles/MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10`) — i.e. someone already targeted Thor's
silicon.

---

## 4. The real 4-way

Per your scoping, the practical decision is **four candidates** — no MiniMax (no role here),
no Hunyuan (Hy4 < GLM-5.3; Hy3 rejected), vision only as an aside. Two axes structure it:
**reap risk** (validated/none vs. heal-required) and **depth↔speed** (active-param count).

| Candidate | Base → Thor total / **Active** | Reap path | Risk | @4-bit fit | Role |
|---|---|---|---|---|---|
| **Qwen3.8-Flash-Next** | ~180 B / **low (≈⅓ of GLM-flash)** `[verify]` | **none** | **lowest** — production-assured | ✅ ~90 GB | **Fast floor.** Best tool-format reliability + speed; the control the others must beat. |
| **GLM-5.3-Flash (reap-50)** ⭐ | **321 B → ~160 B** / **~3× Qwen** | **50 % — VALIDATED near-lossless zone** | **low** | ✅ ~80 GB | **The risk-adjusted heavy (your bet).** Deeper experts from a 321 B frontier-lineage base + 3× active = more reasoning/forward-pass. Slower, priced trade. Fits cleanly. |
| **GLM-5.3 (progressive reap/heal)** | 745 B / **A44B** → ~160–180 B | **~76–78 % + heal** ⚠ | **high** — heal must be executed & proven | post-heal only | **Ceiling moonshot.** Deepest experts, *most* active (~10 t/s base) — "the pro and con, more intense." Only if flash-reap-50 under-delivers on Gate 4. |
| **DeepSeek-V4.1-Flash (progressive reap/heal)** | **≫ 284 B unpruned** `[verify]` / A `[verify]` | **reap/heal to fit** ⚠ | **high** | post-heal only | **Long-context specialist.** MLA + DSA = uniquely cheap 100 k+ context on Thor (GQA GLM can't match). Pick if Gate 3 binds. |

*Aside — DeepSeek-V4-Flash-Vision-Exp:* same DeepSeek economics + vision; only if the loop must
ground on screenshots/GUI/diagrams. Outside the core 4.

**Attention (why it matters for the two heals):** GDA = Gated DeltaNet linear hybrid (Qwen,
O(1) state, cheap long-ctx + fast decode) · GQA + QK-Norm + MoE-MTP head (GLM line — heaviest
KV, MTP head helps spec-decode) · MLA = Multi-head Latent Attention (DeepSeek, ~10–20× KV
compression — the Thor long-context superpower) · DSA = DeepSeek Sparse Attention (lightning
indexer + top-K selector, ~1.5–2× cheaper long-ctx, stacks with MLA).

## 4.5 Fortified performance read (grounded, per-model)

> Confidence (★) reflects how much rests on **shipped data** vs. extrapolation. `%-of-Opus` =
> estimated recovered daily-driver usefulness for *agentic coding* at the Thor operating point;
> still a prior to falsify in §6, but now anchored to real evals.

**Qwen3.8-Flash-Next — the production-assured fast floor. ★★★**
Lineage: the Qwen3-Next hybrid line (Gated DeltaNet + Gated Attn 3:1, ultra-sparse MoE, MTP
built in) — the most *reliable* open agent (top tool-format precision, strong coding/math).
Qwen3.8-27B already **beats DeepSeek-V4-Pro on SWE-bench Pro** and hit **65 t/s on a single
Spark** with spec-decode. At ~180 B with **low active (≈⅓ of GLM-flash's)** it decodes fast on
Thor and fits with headroom. Gates 1 & 2: **best in class**; no reap → **lowest risk of the
four.** Weak spot: shallower single-pass reasoning than a 3×-active deep-base model on the hard
tail — the exact gap the GLM-flash bet targets. **%-of-Opus ~75–85 % routine · ~55–68 % hardest
synthesis.** *If it clears your corpus, you may be done — but the whole exercise is testing
whether GLM-flash's depth beats it where it's weakest.*

**GLM-5.3-Flash (reap-50) — the deep-expert bet, on VALIDATED footing. ★★½**
Base is **~321 B** (a distinct Flash model, *not* a reap of the 745 B full GLM-5.3). Reaping it
**50 % → ~160 B** lands cleanly in Thor's box (~80 GB @4-bit) and **inside** REAP's validated
near-lossless zone (Qwen3-Coder-480B, Kimi-K2 <2 % on code + tool-calling at 50 %). The thesis:
surviving experts from a 321 B frontier-lineage base (GLM-5 family ≈ Sonnet-4 agentic,
LiveCodeBench 82.8 %) are individually richer than a native-180 B Qwen's, and **~3× active
params fire per forward pass** → more
reasoning engaged per token. **Where it wins:** Gate 4 (deep reasoning / novel synthesis /
world knowledge) — active-param width *is* single-pass reasoning depth; ultra-sparse routing
trades against it, so a 3×-active deep-base model should out-reason native-180 B on the hard
tail. **Where it's exposed:** Gate 2 — REAP's degradation is uneven and nicks the long tail
(rare-API / exact tool-format), the one axis Qwen leads natively; it can be *smarter and
flakier* at once. **Cost:** ~3× active compounds over every reasoning step *and* tool
round-trip in the loop; spec-decode (3.4–5.3× on coding) claws most of it back, landing it at
the **low end of your 10–50 band — slower, a chosen trade, not disqualified.** ⇒ the
**call-in-the-heavy**, not an all-turns default. **%-of-Opus: highest of the shortlist on the
hard tail if the deep-expert edge is real; measure it (below).**

**GLM-5.3 (full, progressive reap/heal) — highest ceiling, highest execution risk. ★½**
The "pro *and* con, more intensely" tier you named: deeper experts still, *and* more active
still (A44B → ~10 t/s base). But reaching a Thor-fitting size from 745 B is ~76 % — **past
REAP's validated frontier**, so it needs a **heal** (continued-pretrain/distill) you must
execute and prove. High-ceiling / high-effort. Only worth it if GLM-5.3-Flash-reap-50 clearly
under-delivers on Gate 4 and the heal is feasible for you offline.

**DeepSeek-V4.1-Flash — corrected: much bigger than 284 B unpruned. ★★**
The 284 B/A13B figure was DeepSeek-*V4*-Flash; **V4.1-Flash unpruned is much larger** `[verify
total/active]`, so it too needs reap/heal to fit — not the easy 37 % fit rev 2 claimed. If it
lands, MLA+DSA still gives the best long-context/token economics on Thor (128 k at ~½ cost).
Position it as an *alternative* reasoning heavy to GLM-5.3-Flash, decided on the long-context
leg specifically.

**Hunyuan — deprioritized per your read. ★**
Hy4-preview (770 B / A49B) is, by your assessment, **inferior to GLM-5.3**, and its size +
high active make it the worst Thor fit regardless. **Hy3 (295 B / A21B): rejected — "garbage".**
Hunyuan drops off the shortlist.

**DeepSeek-V4-Flash-Vision-Exp — only if you need eyes. ★½**
Same 284 B/A13B economics plus vision; pull it *only* if the loop must ground on
screenshots/GUI/diagrams. Otherwise the vision tax (speed + a likely text-quality notch) isn't
worth it over text-only V4.1-Flash.

### The synthesized answer (grounded)

1. **The 4-way splits cleanly by risk.** Two **low-risk** (Qwen no-reap; GLM-5.3-Flash-reap-50,
   a validated 50 % of a 321 B base → ~160 B) and two **high-risk** (full GLM-5.3 and
   DeepSeek-V4.1-Flash, both **progressive reap/heal past the validated zone** — a heal you must
   execute and prove). Resolve the low-risk pair *first*; escalate to a heal only for a gap it
   can't close.
2. **The recommended rig = Qwen-180B fast floor + GLM-5.3-Flash-reap-50 heavy.** Your bet —
   GLM-flash's deeper experts (from a 321 B frontier-lineage base) + ~3× active beat native Qwen
   on the hard tail, at the cost of speed — is well-founded and on validated REAP footing. It's
   the *heavy, not the default*, because the 3× active tax compounds across a long loop; Qwen
   carries the routine ~80 %. **Escalation targets:** if that pair leaves a **Gate-4 reasoning**
   gap → **full GLM-5.3 (reap/heal)**; a **Gate-3 long-context** gap → **DeepSeek-V4.1-Flash**
   (MLA+DSA is the only architecture here built for cheap 100 k+ context). Rule: **50 %-reap a
   right-sized base before you 76 %-reap-and-heal a giant.**
3. **The decisive measurement.** Give **Qwen-180B and GLM-5.3-Flash-reap-50** the hard/novel-
   synthesis subset and count **completed hard tasks, net of speed and tool-format derails,
   priced in wall-clock.** If GLM-flash *finishes hard tasks Qwen can't*, it earns the heavy slot
   regardless of t/s. If its only edge is a few rubric points on tasks both complete, Qwen's
   speed + reliability wins. That's the whole gambit, made falsifiable.
4. **Where the gambit pays vs. doesn't:** most plausibly beats native on **G4 reasoning/knowledge
   depth** (deep base, kept-smart experts, 3× active = more reasoning per pass); least plausibly on
   **G2 tool-format reliability** (pruning nicks rare-format precision; Qwen leads there natively —
   it can be *smarter and flakier at once*). Which wins = which gate binds you.
5. **Opus recovery, grounded:** ~**78–88 %** on the routine ~80 % of agentic-coding work looks
   genuinely achievable (GLM-4.6≈Sonnet-4, Qwen3.8-27B>DeepSeek-V4-Pro on SWE-bench Pro, REAP
   near-lossless at ≤50 %). The gap that stays open is the **hardest novel-synthesis / longest-
   horizon** tail — an honest ~**55–70 %** until proven — which is what Opus/Astra actually buy.

---

## 5. Attention is a *fitting constraint*, not a feature bullet

On Thor the KV cache competes with weights for the same 110 GB, and per-token KV read is
subtracted from the 232 GB/s in §1.2 (so it costs both memory *and* speed at long context):
- **MLA (DeepSeek V4-Flash/Vision)** — latent KV compression ~10–20×; the most Thor-friendly
  for large context. Biases you to DeepSeek if Gate 3 (≥100 k) binds.
- **DSA (DeepSeek V3.2+)** — lightning-indexer + top-K token selection; ~1.5–2× cheaper
  attention on long sequences, stacks with MLA.
- **GDA / KDA (Qwen / Kimi linear-hybrid)** — O(1) recurrent state in the linear layers; KV
  grows only in the periodic full-attention layers → cheap long-ctx *and* faster decode.
- **Plain MHA/GQA (GLM line)** — heaviest KV; at 100 k it can force lower quant or shorter ctx.

**Rule:** fix your real context ceiling first, compute its KV footprint per candidate, subtract
from the §1.1 headroom *and* the §1.2 bandwidth, *then* pick quant. KV, not weights, usually
kills long context on unified memory.

---

## 6. The measurement harness — your tri-sweep, ported to models

Rank on *your* frozen agentic corpus, not public benchmarks (*a number is only as good as the
corpus it was taken on*).

**Corpus (once, reuse forever):** 30–60 real tasks from your loops — a manuk tick, an
archipelago feature, a multi-file refactor, a "read this repo & answer", a "plan→implement→
self-verify" long-horizon task. Include **rare-API / obscure-language** tasks and **long-context**
(big repo paste) tasks — these are the reap-detectors. Freeze and version it.

| Leg | Gate | Metric | Method |
|---|---|---|---|
| Speed | 1 | decode t/s + prefill latency, at real ctx, spec-decode on | Measure **on Thor**, 3 quant levels, ≥3 runs, report the **band** (*one run refuses nothing*). |
| Loop reliability | 2 | valid-tool-call %, format-error %, loops completed w/o derail | Full end-to-end runs in your real harness; count derailments/malformed calls over long sessions. |
| Long-context | 3 | success vs ctx (10k→50k→100k→ceiling) | Same task, padded with real repo; report the **degradation knee**, not just max ctx. |
| Reasoning/knowledge | 4 | pass-rate + rubric on the hard/novel/rare subset | Rubric or strong-model judge + human spot-check. Where the gambit lives or dies. |

**Controls (or the numbers lie):** (a) **Opus anchor** — run the corpus through Opus to fix the
100 % you're recovering a fraction of. (b) **Native-180B control** — Qwen must be *beaten*, not
tied. (c) **Reap-delta** — score pre- and post-reap on the *same* corpus; *diff the failing task
names* on the rare/long subset, not the totals. (d) **Quant-delta** — 4.0 vs 3.5 vs 3.0 bpw; find
*your* quant knee. **Report:** per model, `PASS/FAIL each gate · t/s band · %-of-Opus · reap-delta
on rare subset · quant`. Ties break to the **simpler artifact.**

---

## 7. Decision path (grounded ordering)

1. **Stand up Qwen-180B (Qwen3.8-Flash-Next), no reap — the production-assured floor.** Lowest
   ops risk, fastest, MTP built in. It's the control GLM-flash must *beat*, and possibly the
   answer for the routine ~80 %.
2. **Bring up your GLM-5.3-Flash-reap-50 — the deep-expert heavy (your bet).** Run it head-to-head
   with Qwen on the **hard/novel-synthesis subset** and score **completed tasks net of speed +
   tool-format derails, in wall-clock** (§4.5 decisive test). This is the experiment the whole
   exercise exists to run.
3. **Escalate to a heal-required option ONLY for a gap steps 1–2 can't close.** If Qwen +
   GLM-flash leave a **Gate-4 (deep reasoning)** gap and you can run the heal offline → **full
   GLM-5.3 (progressive reap/heal ~77 %)**. If they leave a **Gate-3 (long-context)** gap →
   **DeepSeek-V4.1-Flash (reap/heal)** — MLA+DSA is the only architecture here built for cheap
   100 k+ context. Both are training-rig projects (§8); don't start one before the low-risk pair
   proves it's needed.
4. **Vision variant only if the loop needs eyes.**

Likely outcome: a **two-model daily rig** — Qwen-180B fast floor + GLM-5.3-Flash-reap-50 heavy —
with a heal-required option added only if a specific gate gap survives the head-to-head.

---

## 8. Practical gotchas

- **Heal ≠ Thor.** Progressive reap/**heal** (needed past ~50 % prune) is a *training-rig* job.
  Either download a community-healed checkpoint or do reap+heal offline and ship only the final
  quantized inference checkpoint to Thor. Thor is inference-only here.
- **Reap × quant compound.** Measure the *combined* post-reap-**and**-post-quant artifact on your
  corpus; don't trust a reap eval done at BF16 and a quant eval done un-reaped.
- **Spec-decode drafter must fit and be family-matched.** DFlash2/MTP/EAGLE gave 3.4–5.3× on
  *coding* on GB10; "DSpark" underperformed (58.6 % saturation). Budget the drafter's memory
  before choosing weight quant, and **tune the draft budget on your code** (12-token budget won
  the coding benchmark; throughput isn't linear in acceptance length).
- **`config.json` is truth for §1/§2.** total & active params, expert/top-k counts, attention
  type, native ctx — read them, don't trust prose.
- **Router health after reap.** A mis-scaled router post-prune under-routes to survivors and
  looks like "the model got dumb." Suspect the router before the experts.

---

## 9. What's now known vs. still `[verify]`

**Grounded (this revision):** Thor = 273 GB/s / ~85 % MBU / GB10-class (Spark benchmarks
transfer). REAP: near-lossless ≤25 %, <2–7 % at 50 %, past-50 % needs heal. Active-param decode
table (A3B→A49B). Spec-decode coding 3.4–5.3× on GB10. Real specs: Qwen3-Next 80B/A3B; MiniMax-M2
230→172B/A10B; DeepSeek-V4-Flash 284B/A13B & V4-Pro 1.6T/A49B; GLM-5 745B/A44B; Hunyuan Hy3
295B/A21B & Hy4 770B/A49B.

**Still verify:** exact **Qwen3.8-Flash-Next** total/active (assumed A3B-class); whether a
**natively small GLM-5.3-Flash** exists and its size/active; per-token **KV read** at your ctx per
attention family; your real **context ceiling**; your drafter's **acceptance length on your code**;
inference-stack (vLLM/SGLang/TensorRT-LLM/llama.cpp) support for each attention + quant + spec-
decode on Blackwell; and the current **MiniMax-M2.5/.1-REAP** benchmark retention (some cards read
"TBD").

---

## 10. Sources

- REAP (method, curve, ICLR 2026): [arXiv 2510.13999](https://arxiv.org/html/2510.13999v3) · [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap) · [Cerebras REAP collection](https://huggingface.co/collections/cerebras/cerebras-reap)
- REAP checkpoints: [MiniMax-M2-REAP-172B-A10B](https://huggingface.co/cerebras/MiniMax-M2-REAP-172B-A10B) · [MiniMax-M2.5-REAP-172B-A10B](https://huggingface.co/cerebras/MiniMax-M2.5-REAP-172B-A10B) · [MiniMax-M2-REAP-162B-A10B (MarkTechPost)](https://www.marktechpost.com/2025/11/15/cerebras-releases-minimax-m2-reap-162b-a10b-a-memory-efficient-version-of-minimax-m2-for-long-context-coding-agents/) · [Qwen3-Coder-REAP-246B-A35B](https://huggingface.co/cerebras/Qwen3-Coder-REAP-246B-A35B-FP8) · [GLM-4.7-REAP-268B-A32B](https://huggingface.co/cerebras/GLM-4.7-REAP-268B-A32B-FP8) · [NVFP4-GB10 quant](https://huggingface.co/saricles/MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10)
- Thor / GB10 hardware: [Jetson AGX Thor specs (ServeTheHome)](https://www.servethehome.com/nvidia-jetson-agx-thor-developer-kit-blackwell-for-robotics/2/) · [NVIDIA Jetson Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)
- GB10 decode physics + spec-decode: [DGX Spark bandwidth-ceiling 85 % (ai-muninn)](https://ai-muninn.com/en/blog/dgx-spark-bandwidth-ceiling-85-percent) · [Spark-DGX-Benchmark](https://github.com/rossingram/Spark-DGX-Benchmark) · [llama.cpp DGX Spark report](https://github.com/DandinPower/llama.cpp_bench/blob/main/dgx_spark/report.md)
- Qwen3-Next: [vLLM blog](https://vllm.ai/blog/2025-09-11-qwen3-next) · [Qwen3-Next-80B-A3B](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- DeepSeek: [V3.2-Exp (arXiv 2512.02556)](https://arxiv.org/pdf/2512.02556) · [V3.2-Exp HF](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) · [V3 report / MTP (arXiv 2412.19437)](https://arxiv.org/pdf/2412.19437) · [V4 specs (morphllm)](https://www.morphllm.com/deepseek-v4) · [V4 (NxCode)](https://www.nxcode.io/resources/news/deepseek-v4-release-specs-benchmarks-2026)
- GLM: [GLM-5 (arXiv 2602.15763)](https://arxiv.org/html/2602.15763v2) · [GLM-5 deep dive](https://medium.com/data-science-in-your-pocket/glm-5-deep-dive-745b-moe-beast-crushing-swe-bench-code-benchmarks-5757a3022251) · [GLM-4.6 docs](https://docs.z.ai/guides/llm/glm-4.6)
- Hunyuan: [Hy3 (ai-tldr)](https://ai-tldr.dev/models/hunyuan-hy3/) · [Hy4-preview (ai-tldr)](https://ai-tldr.dev/models/hunyuan-hy4-preview/) · [Hunyuan-Large (arXiv 2411.02265)](https://arxiv.org/pdf/2411.02265)

*Bottom line: fit the box with *total* params; understand that decode speed rides on *active*
params (reap doesn't buy speed) — so the deeper-expert / 3×-active GLM-5.3-Flash-reap-50 is a
**deliberate, priced trade**, not a mistake, and it sits on validated 50 %-reap footing. The
rig the evidence supports: **Qwen-180B as the production-assured fast floor + GLM-5.3-Flash-reap-50
(321 B base, validated 50 %-reap) as the deep-reasoning heavy** — with **full GLM-5.3** or
**DeepSeek-V4.1-Flash** (both progressive reap/heal) added only if a specific Gate-4 or Gate-3
gap survives the head-to-head. Then let
your own frozen corpus on Thor settle the one question that matters: does GLM-flash's depth
finish hard tasks Qwen can't, net of its slower speed? The family data is real evidence; this
generation's checkpoints are not — measure them.*
