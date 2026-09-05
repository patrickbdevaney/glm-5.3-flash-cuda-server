#!/usr/bin/env python3
"""Oracle for the DSA indexer — the one subsystem that stands between this server and a context
longer than 2048 tokens.

    cd ~/glm-5.3-reap && ./.venv/bin/python ~/glm-5.3-flash-cuda-server/ref/gen_indexer.py

Builds ONLY `Glm5NextTextIndexer` plus the two tensors needed to feed it (`q_a_proj`,
`q_a_layernorm`). That is ~16 MB of weights, not the 20 GiB a whole layer costs once its 144 NVFP4
experts are materialised — the mistake made once already with gen_mla.py.

Two jobs:
  1. Dump inputs and reference outputs so a CUDA kernel can be gated exactly.
  2. Answer, from the real module rather than from reading it, what the indexer actually does on
     either side of the 2048 boundary. `ref/gen_mla.py` already established that below 2048 it
     selects every pool and the engine's dense MLA is therefore EXACT. This checks what changes
     above it, which is the thing the kernel has to reproduce.
"""
import json, os, sys
import numpy as np
import torch

SRC = os.path.expanduser("~/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2")
OUT = os.path.expanduser("~/glm-5.3-flash-cuda-server/ref")
LAYER = 3                      # the first full-attention layer
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# fp32, NOT the checkpoint's bf16. The gate's fp32 arm must be compared against an fp32 reference:
# with a bf16 oracle, q and the pool keys carry bf16 rounding that an fp32 kernel does not, the pool
# scores differ in the 3rd decimal, and at contexts where the top-k actually EXCLUDES something the
# 512th-place pool flips. That showed up as 404 of 3003 rows differing by exactly one pool each --
# a precision artefact wearing the costume of a logic bug. bf16 weights widen to fp32 exactly, so
# this uses identical weights, only a wider accumulator.
DT = torch.float32


def load(names):
    """Pull a handful of tensors out of the shards without loading anything else."""
    from safetensors import safe_open
    idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
    out = {}
    by_file = {}
    for n in names:
        if n not in idx:
            raise KeyError(f"{n} not in the index")
        by_file.setdefault(idx[n], []).append(n)
    for f, ns in by_file.items():
        with safe_open(os.path.join(SRC, f), framework="pt") as s:
            for n in ns:
                out[n] = s.get_tensor(n)
    return out


def main():
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer, Glm5NextTextRMSNorm

    cfg = AutoConfig.from_pretrained(SRC)
    tcfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    P = f"model.language_model.layers.{LAYER}."
    idx_names = [P + "self_attn.indexer." + n for n in
                 ("wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias",
                  "weights_proj.weight")]
    idx_names += [P + "self_attn." + n for n in
                  ("indexer.index_kpool_compress_ape", "indexer.index_kpool_compress_gate",
                   "q_a_proj.weight", "q_a_layernorm.weight")]
    idx_names += ["model.language_model.embed_tokens.weight"]
    W = load(idx_names)

    ind = Glm5NextTextIndexer(tcfg, layer_idx=LAYER).to(DEV).to(DT)
    sd = {k.split("indexer.")[-1]: v for k, v in W.items() if ".indexer." in k}
    missing = ind.load_state_dict({k: v.to(DEV).to(DT) for k, v in sd.items()}, strict=False)
    print("indexer loaded; missing:", list(missing.missing_keys), "unexpected:", list(missing.unexpected_keys))

    print(f"config: index_topk={tcfg.index_topk} index_kpool={tcfg.index_kpool} "
          f"always_select_tail={tcfg.index_kpool_always_select_tail} "
          f"n_heads={ind.n_heads} head_dim={ind.head_dim} scale={ind.softmax_scale:.6f}")

    emb = W["model.language_model.embed_tokens.weight"]
    q_a_w = W[P + "self_attn.q_a_proj.weight"].to(DEV).to(DT)
    q_a_n = W[P + "self_attn.q_a_layernorm.weight"].to(DEV).to(DT)
    qnorm = Glm5NextTextRMSNorm(q_a_w.shape[0], eps=tcfg.rms_norm_eps).to(DEV).to(DT)
    qnorm.weight.data.copy_(q_a_n)

    # Real embedding rows, not noise: the KDA oracle taught this the hard way — driving a gate with
    # N(0, 0.02) puts it in a regime the model never sees, where relative error is dominated by
    # cancellation and the numbers mean nothing.
    g = torch.Generator().manual_seed(7)
    dumps = {}
    # Sizes chosen so the TAIL path is exercised on both sides of the boundary. A first pass used
    # only multiples of index_kpool=4, where every pool is complete and append_visible_tail never
    # fires — so the one branch that makes the output 2048+3 wide instead of 2048 went untested.
    # The boundary is NOT at index_topk. A trailing INCOMPLETE pool is not selectable (pool_valid
    # requires all index_kpool tokens present) but its tokens are appended raw by
    # append_visible_tail — so T=2051 has 513 pools, selects only 512, and is still fully dense.
    # Dense iff floor(T / index_kpool) <= index_topk / index_kpool, i.e. T <= 2051. 2052 is the
    # first sparse length. These sizes straddle that.
    for T in (38, 512, 2048, 2050, 2051, 2052, 3001, 3003):
        ids = torch.randint(0, 154820, (T,), generator=g)
        h = emb[ids].to(DEV).to(DT).unsqueeze(0)                       # [1, T, 4096]
        q_resid = qnorm(torch.nn.functional.linear(h, q_a_w))          # [1, T, 1536]
        mask = torch.ones(1, T, dtype=torch.bool, device=DEV)

        with torch.no_grad():
            topk = ind(hidden_states=h, q_resid=q_resid, attention_mask=mask, past_key_values=None)

            # The per-pool index_scores for the LAST query, recomputed through the module's own
            # get_pooled_states so the gate can check the arithmetic directly rather than inferring
            # it from which pools survived a top-k. A set difference at the selection boundary is
            # only acceptable if the scores there are a near-tie, and that cannot be judged without
            # the scores themselves.
            kk = ind.k_norm(ind.wk(h)).view(1, T, -1, ind.head_dim).squeeze(2)
            gs = torch.nn.functional.linear(h, ind.index_kpool_compress_gate)
            packed = torch.cat([kk, gs, mask.to(kk.dtype)[..., None]], dim=-1)
            pool_keys, _, _ = ind.get_pooled_states(packed_states=packed)
            qq = ind.wq_b(q_resid).view(1, T, -1, ind.head_dim)
            sc = torch.nn.functional.relu(
                torch.matmul(qq.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
                * ind.softmax_scale)
            ww = ind.weights_proj(h.to(ind.weights_proj.weight.dtype)).float() * (ind.n_heads ** -0.5)
            iscores = torch.matmul(ww.unsqueeze(-2), sc).squeeze(-2)      # [1, T, P]
            last_scores = iscores[0, -1].float().cpu().numpy()

        # What the LAST query can actually see, which is the decode-time question.
        last = topk[0, -1]
        sel = last[last >= 0]
        uniq = torch.unique(sel)
        n_pools = (T + tcfg.index_kpool - 1) // tcfg.index_kpool
        select_k = min(tcfg.index_topk // tcfg.index_kpool, n_pools)
        dense = uniq.numel() >= T
        tail = T % tcfg.index_kpool
        print(f"  T={T:5d}  pools={n_pools:4d}  select_k={select_k:4d}  tail={tail}  "
              f"width={topk.shape[-1]:5d}  distinct visible to last query = {uniq.numel():5d} of {T:5d}"
              f"   {'DENSE (indexer is a no-op)' if dense else 'SPARSE'}")

        # Dump the indexer's ACTUAL INPUTS (h and q_resid), not just the token ids. Recomputing
        # them in the gate would mean the gate also tests q_a_proj/q_a_layernorm, so a mismatch
        # there would be reported as an indexer bug. The kernel still derives k, gate_scores, q and
        # the head weights from h itself — those ARE the indexer.
        if T in (38, 2050, 3003):
            dumps[T] = dict(scores=last_scores,
                            ids=ids.numpy().astype(np.int32),
                            h=h[0].detach().float().cpu().numpy(),
                            q_resid=q_resid[0].detach().float().cpu().numpy(),
                            topk=topk[0].to(torch.int32).cpu().numpy())

    os.makedirs(OUT, exist_ok=True)
    for T, d in dumps.items():
        for k, v in d.items():
            v.tofile(os.path.join(OUT, f"indexer_{k}_{T}.bin"))
    wdump = {
        "wq_b":          W[P + "self_attn.indexer.wq_b.weight"],
        "wk":            W[P + "self_attn.indexer.wk.weight"],
        "k_norm_w":      W[P + "self_attn.indexer.k_norm.weight"],
        "k_norm_b":      W[P + "self_attn.indexer.k_norm.bias"],
        "weights_proj":  W[P + "self_attn.indexer.weights_proj.weight"],
        "compress_ape":  W[P + "self_attn.indexer.index_kpool_compress_ape"],
        "compress_gate": W[P + "self_attn.indexer.index_kpool_compress_gate"],
    }
    for k, v in wdump.items():
        v.float().cpu().numpy().tofile(os.path.join(OUT, f"indexer_w_{k}.bin"))
        print(f"  weight {k:15s} {tuple(v.shape)}")

    meta = dict(layer=LAYER, index_topk=int(tcfg.index_topk), index_kpool=int(tcfg.index_kpool),
                always_select_tail=bool(tcfg.index_kpool_always_select_tail),
                n_heads=int(ind.n_heads), head_dim=int(ind.head_dim),
                softmax_scale=float(ind.softmax_scale),
                output_width=int(tcfg.index_topk + (tcfg.index_kpool - 1
                                 if tcfg.index_kpool_always_select_tail else 0)),
                dumped=[int(t) for t in dumps])
    json.dump(meta, open(os.path.join(OUT, "indexer_meta.json"), "w"), indent=1)
    print("wrote", OUT + "/indexer_{ids,topk}_*.bin and indexer_meta.json")


if __name__ == "__main__":
    main()
