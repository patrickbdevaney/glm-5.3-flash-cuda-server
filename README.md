# glm-5.3-flash-cuda-server

A pure-CUDA inference engine and OpenAI-compatible server for
[`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2) on one
NVIDIA Jetson AGX Thor. That checkpoint is a 50% expert-pruned GLM-5.3-Flash with text and image input, produced by
the companion pipeline [glm-5.3-reap](https://github.com/patrickbdevaney/glm-5.3-reap). The request path has no
Python: it runs hand-written kernels for `sm_110a`, a C++ tokenizer and chat encoder, and an HTTP server in one
binary.

## Status (2026-10-26)

This is a research release from a single maintainer. It serves text and images, single stream, at up to 128K
tokens of context.

| Component | State |
|---|---|
| Engine: all 45 layers, MTP block, vision tower | Running; every kernel gated against `transformers` on real checkpoint weights |
| Server ([`server/server.cpp`](server/server.cpp)) | OpenAI chat and completions, streaming, tool calls, images, embeddings |
| NVFP4 dense-weight overlay | Generated locally by [`tools/requant_dense_nvfp4.py`](tools/requant_dense_nvfp4.py); not published |
| Speculative decoding (MTP) | Built and gated, then measured as a net loss; disabled. See [Findings](#findings) |
| Concurrent-request batching | Not built; requests are serialized |

## Results

All numbers were measured on one Jetson AGX Thor with the model resident. Benchmarks ran on an otherwise idle box,
interleaving old and new builds round-robin and reporting the minimum
([`CLAUDE.md`](CLAUDE.md) §6).

**Speed**

| Metric | Value |
|---|---|
| Decode, batch 1 | **11.9 tok/s** (83.9 ms/step); ~50% of the bandwidth roofline for this checkpoint |
| Prefill, short prompts | 47.8 ms/token at chunk 32, about 21 tok/s |
| Prefill, long prompts | ~27 min for 32K tokens, ~57 min for 64K, ~2.1 h for 128K |
| Resident memory | 101.9 GiB with the overlay; 105.0 GiB with a 132K context allocated |

**Quality**

| Check | Result |
|---|---|
| Gates | 16 gate programs in [`scripts/gate.sh`](scripts/gate.sh), all passing after the latest kernel change, plus `gate_vision_preproc` |
| Overlay cost, perplexity over 47,195 tokens | 4.4996 vs 4.4305 without it (**+1.56%**) |
| Overlay cost, top-1 agreement with the bf16 dense path | 90.10%; **0.138%** of tokens are confident flips (reference NLL < 0.5) |
| Long-context retrieval (needle in a haystack) | **9/9** found at 32K, 64K and 128K × depth 0.1/0.5/0.9 ([`results/`](results/niah_nvfp4_dsa.json)) |
| Positional perplexity, 0–6K | Falls from 7.81 to 2.91, with no discontinuity where the sparse attention path takes over at 2,051 tokens |
| HumanEval pass@1 | 91.5% (164 problems, `reasoning_effort=high`, 4,096-token budget, 7 truncated) |

The overlay's flips sit where the model was already unsure: the median reference NLL is 2.21 on disagreements and
0.35 on agreements. Per-tensor cosine (0.995–0.997) predicted a smaller effect than perplexity measured, so quote
perplexity. `GLM5_DENSE_NVFP4=0` turns the overlay off without touching any file.

HumanEval is a single benchmark run against this server. It is evidence that the served model works, not a quality
claim for the checkpoint. The checkpoint's own evaluation against the unpruned model is in the
[glm-5.3-reap README](https://github.com/patrickbdevaney/glm-5.3-reap).

## Findings

**1. Linear attention, not the MoE, dominated per-token bytes.** The checkpoint has 45 layers: 34 KDA
(linear-attention) layers, 11 MLA/DSA full-attention layers, 144 routed experts with 8 active plus one shared, and
four hyper-connection residual streams.
- REAP and NVFP4 compressed the experts. Those are 83% of the bytes on disk but only 24% of the bytes read per
  token, because only 8 of 144 experts are read.
- The other 13.32 GiB, bf16 dense weights, were 76% of the per-token bytes. KDA alone was 47.4%, nearly twice the
  whole MoE.
- An NVFP4 overlay of those dense weights cut bytes per token from 19.76 G to 9.76 G and decode time by 1.42×
  ([`docs/ROOFLINE.md`](docs/ROOFLINE.md)).

**2. Activation traffic, not weight traffic, bound every gemv.** A gemv reads 4 bytes of activation per weight.
That is 2 bytes per byte of bf16 weight, but 7.1 per byte of NVFP4. This explains three early results:
- halving the weight bytes first bought only 1.4%;
- the batched path cost exactly M gemvs;
- prefill got slower as chunks got wider.

Reusing each activation across several output rows fixed all three
([`docs/OPTIMIZATION_LOG.md`](docs/OPTIMIZATION_LOG.md) #11–12).

**3. Speculative decoding cannot win on this architecture.** The native MTP head accepts 72.3% of drafts without
fine-tuning, so the rest of the pipeline was built: multi-token forward bit-identical to sequential decode, and KDA
state rollback through spare state slots.
- **The verify cost decided it.** A width-2 verify costs 2.16 decode steps. The best draft depth reaches 0.95×, and
  even a perfect drafter is bounded at 1.27×.
- **The cause is expert union.** With 144 experts and top-8 routing, M tokens touch about 144·(1−(1−8/144)^M)
  distinct experts, which is still 7.4 per token at M=4. Expert traffic grows almost linearly in M exactly where
  speculation needs it flat.
- **No draft head fixes this.** Fine-grained MoE at small draft depths is hostile to speculation, so do not train a
  head for this engine ([`docs/SPEC_DECODE.md`](docs/SPEC_DECODE.md), [`docs/OPTIMIZATION_LOG.md`](docs/OPTIMIZATION_LOG.md) #17).

**4. Launch overhead and kernel efficiency are separate terms.** KDA's gate projections ran at 7% of achievable
bandwidth because they issued 238 kernel launches per token.
- Folding the gemvs that share an input into one launch, bit-identically, gave +4.8% end to end (11.36 → 11.91
  tok/s). Two untouched sub-ops served as controls, at 1.002× and 0.988×.
- The projection had been +16%. Removing launches recovers only what the launches cost; the remaining gap is the
  kernels' efficiency on tall, thin shapes.

## Engineering

- **Gates on real weights.** Every kernel is checked against `transformers` on the checkpoint's own tensors, never
  synthetic fixtures. Both "misaligned address" faults in the log were invisible with `cudaMalloc`-backed test
  buffers and fatal on the real model.
- **Bit-identity where speculation needs it.** The batched forward at M=1 is asserted bit-identical to sequential
  decode. Rollback is gated by accepting 0, 1, 3 or 5 of 5 drafts and comparing with a run that never speculated.
- **Sparse attention covered on both sides of the switch.** The DSA sparse path is bit-exact against dense for all
  2,051 positions below the limit. Above it no dense answer exists, so coverage there comes from positional
  perplexity and NIAH.
- **Vision path.**
  - The 24-block vision tower is gated at cosine 1.000000000 against an fp32 oracle; a bf16 oracle was the less
    precise side and failed the gate.
  - Preprocessing from PNG bytes matches `Glm5NextImageProcessor` to one fp32 ulp.
  - The language side is pure NoPE, so splicing image embeddings needs no mrope.
- **Exact text front end.** The tokenizer matches HF on 170/170 cases, id for id. The chat encoder matches HF's own
  Jinja template on 42/42 cases, byte for byte.
- **Memory safety on unified memory.** On Thor, `cudaMalloc` draws from system DRAM, so an over-allocation causes a
  system OOM, not an error return. Loads are sized from safetensors headers, never probed.
  [`scripts/run_full.sh`](scripts/run_full.sh) makes the server the preferred OOM victim.

## Getting started

Requirements:
- Jetson AGX Thor with CUDA for `sm_110a`.
- About 105 GiB free memory. Run nothing else GPU-heavy alongside; on Thor, a second large process can force a hard
  reboot.
- The NVFP4 v2 checkpoint (98.2 GiB).

```bash
huggingface-cli download patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2 \
  --local-dir ~/models/glm-5.3-flash-reap50-nvfp4-v2

# Optional, recommended: the 3.7 GiB NVFP4 dense overlay (1.42x decode, +1.56% perplexity).
# The engine looks for it next to the checkpoint, or at $GLM5_NVFP4_OVERLAY.
python3 tools/requant_dense_nvfp4.py --model ~/models/glm-5.3-flash-reap50-nvfp4-v2 \
  --out ~/models/glm-5.3-flash-dense-nvfp4-overlay

bash scripts/build.sh                         # engine, server, gates and benches into build/
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches   # reclaim the driver page pool before a ~100 GiB load
nohup setsid scripts/run_full.sh --ckpt ~/models/glm-5.3-flash-reap50-nvfp4-v2 --ctx 32768 --port 8080 \
  > server.log 2>&1 < /dev/null &
```

The load takes about 80 s. Then:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Write a Python function that reverses a linked list."}],
  "reasoning_effort": "high", "max_tokens": 2048}'
```

**API**

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | Streaming (SSE); reasoning returned separately; tool calls; images as `data:` URIs |
| `POST /v1/completions` | Raw prompt, no chat template |
| `POST /v1/embeddings` | Last-token pooled hidden state, L2-normalized, 4,096-d |
| `POST /tokenize`, `/detokenize` | `with_pieces` for per-token inspection |
| `GET /props`, `/health`, `/metrics`, `/v1/models`, `/` | Capabilities, Prometheus metrics, web UI |

- **Sampling fields:** `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, `max_tokens`, `reasoning_effort`,
  `logprobs`/`top_logprobs`.
- **Always set `reasoning_effort` to `low` or `high`.** The default is Max. At Max, HumanEval generations ran past
  8,192 tokens without terminating, either in repetition loops or in unbounded reasoning.
- **Prefix caching:** a request that extends the resident context reuses it; `cached_tokens` reports how much.
- **Images:** plain http(s) image URLs are refused on purpose, because fetching arbitrary URLs for a caller is
  server-side request forgery.

**Context memory.** Only the 11 MLA layers keep a KV cache. Each stream costs a fixed 145.56 MiB of KDA state plus
`11 × ctx × 2,048` bytes of latent cache: 322 MiB in total at 8K, about 2.9 GiB at 128K.

**Rebuilding the gates.** The oracle fixtures (741 MB) are not in git. Regenerate them from the checkpoint, then run
the gates:

```bash
cd ~/glm-5.3-reap      # the oracles run in the companion repo's environment
for g in kda moe layer mla stack vision; do ./.venv/bin/python ~/glm-5.3-flash-cuda-server/ref/gen_$g.py; done
cd ~/glm-5.3-flash-cuda-server && bash scripts/gate.sh
```

## Compatibility

| Platform | State |
|---|---|
| Jetson AGX Thor (`sm_110a`), 128 GB | Tested |
| Other Blackwell GPUs | Not supported. Kernels and memory sizing target Thor's unified memory |
| Other checkpoints | GLM-5.3-Flash REAP50 NVFP4 v2 only. v1 and v3 were not tested; v3 also quantizes the KDA attention projections to NVFP4 |

## Limitations

- **Single stream.** A global lock serializes requests.
- **Long-prompt prefill is slow.** 128K tokens take about two hours. The NIAH result proves retrieval, not
  interactive long-context use.
- **Above 2,051 tokens, sparse attention is verified statistically, not bit-exactly.** The checks are positional
  perplexity and NIAH.
- **Thin quality evidence.** One benchmark plus perplexity and agreement; no agentic or multi-turn tool-use suite
  has been run against this server.
- **The overlay changes outputs.** About 10% of top-1 tokens differ from the bf16 dense path, almost all of them
  low-confidence.
- **Hard-coded paths.** Oracle scripts and some defaults assume the author's directory layout.

## Roadmap

1. **Concurrent-request batching.** The measured batch curve gives 1.69× aggregate throughput at width 16, and 16
   streams at 8K cost about 5 GiB.
2. **Faster prefill at long context**, the main barrier to using the 128K window interactively.
3. **Row tiling for `kda:o_proj` and `mla:o_proj`** (15% of decode at ~100 GB/s); neither can be fused.
4. **v3 checkpoint support**, which also quantizes the KDA attention projections to NVFP4.

Closed: speculative decoding (Finding 3) and the overlay accuracy decision (kept).

## Repository map

| Path | Contents |
|---|---|
| [`kernels/`](kernels/) | KDA, MoE, MLA, DSA indexer, gemv, vision tower, profiling marks |
| [`src/engine.cu`](src/engine.cu) | Forward pass, chunked prefill, multi-token forward, state slots |
| [`server/server.cpp`](server/server.cpp), [`include/`](include/) | HTTP server, tokenizer, chat encoder, sampler, stream parser, image preprocessing |
| [`tests/`](tests/) | The gates: one program per kernel family, plus tokenizer, encoder, sampler and API |
| [`ref/`](ref/) | PyTorch oracle generators |
| [`tools/`](tools/) | Roofline model, bandwidth probe, benches, perplexity, NVFP4 overlay builder, eval harnesses |
| [`docs/`](docs/) | [`ROOFLINE`](docs/ROOFLINE.md), [`ARCH_DELTA`](docs/ARCH_DELTA.md) (what was ported from the DeepSeek server), [`SPEC_DECODE`](docs/SPEC_DECODE.md), [`OPTIMIZATION_LOG`](docs/OPTIMIZATION_LOG.md) (every measurement) |
| [`results/`](results/) | NIAH grid |
| [`dev/archive/`](dev/archive/) | Superseded status notes and planning documents |
| [`CLAUDE.md`](CLAUDE.md) | Operating rules for agents working in this repository: detachment, gating, measurement on a shared box |

## Contributing and contact

Issues are welcome for reproducibility problems and wrong numbers. Pull requests are not being accepted yet.

## License

- **Code in this repository:** MIT ([`LICENSE`](LICENSE)).
- **Third-party code keeps its own license:** `include/third_party/httplib.h` (MIT), `json.hpp` (MIT) and
  `stb_image.h` (public domain or MIT).
- **Weights:** MIT, inherited from [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash). This
  work is not affiliated with or endorsed by Z.ai.

## Citation

```bibtex
@misc{devaney2026glm53cuda,
  title  = {glm-5.3-flash-cuda-server: a pure-CUDA server for REAP-pruned GLM-5.3-Flash on Jetson AGX Thor},
  author = {Devaney, Patrick},
  year   = {2026},
  url    = {https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server}
}
```

Related: [glm-5.3-reap](https://github.com/patrickbdevaney/glm-5.3-reap), the pruning pipeline;
[deepseek-v4-flash-0731-cuda](https://github.com/patrickbdevaney/deepseek-v4-flash-0731-cuda), whose MLA and MoE
kernels this engine ported.
