#!/usr/bin/env python3
"""gen_mla.py - PyTorch oracle for the MLA (full-attention) sublayer, and a verification of the
claim the decode path depends on:

    at context <= index_topk (2048), the DSA indexer selects EVERY visible token,
    so MLA is exactly dense causal attention and the indexer can be skipped.

That is not an assumption here. The oracle calls the real indexer's get_pooled_states /
get_visible_tokens and checks the selected set against the full visible set.

MLA in this model is pure NoPE (qk_rope_head_dim == 0): there is no rotary embedding anywhere in
the main attention path.
"""
import argparse, os, sys
import torch
import torch.nn.functional as F

REAP = os.path.expanduser('~/glm-5.3-reap')
sys.path.insert(0, os.path.join(REAP, 'scripts'))


def w(p, t): t.detach().to(torch.float32).cpu().numpy().ravel().astype('<f4').tofile(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(REAP, 'output/glm-5.3-flash-reap50-nvfp4-pass2'))
    ap.add_argument('--layer', type=int, default=3)
    ap.add_argument('--ctx', type=int, default=37, help='tokens already in the cache')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'mla'))
    ap.add_argument('--seed', type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention
    from stream_saliency import ShardReader

    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True).text_config
    assert cfg.layer_types[a.layer] == 'deepseek_sparse_attention'
    assert cfg.qk_rope_head_dim == 0, 'this oracle assumes pure NoPE'
    R = ShardReader(a.model)

    # Build ONLY the attention module. Going through _build_layer would also materialise the
    # layer's 144 NVFP4 experts as fp32 - ~21 GiB and minutes of CPU - for an oracle that never
    # touches the MoE.
    dev = torch.device('cpu')
    A = Glm5NextTextAttention(cfg, a.layer).to(torch.float32).eval()
    pfx = 'model.language_model.layers.%d.self_attn.' % a.layer
    sd = {}
    want = dict(A.named_parameters())
    want.update(dict(A.named_buffers()))
    missing = []
    for name in want:
        try:
            sd[name] = R.get(pfx + name).to(torch.float32)
        except Exception:
            missing.append(name)
    if missing:
        print('  note: %d params not in checkpoint: %s' % (len(missing), missing[:6]))
    A.load_state_dict(sd, strict=False, assign=True)

    T = a.ctx + 1                       # cache tokens + the one we decode
    emb = R.get('model.language_model.embed_tokens.weight')
    g_ = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(0, cfg.vocab_size, (T,), generator=g_)
    x = emb[ids].to(torch.float32).to(dev).unsqueeze(0)          # [1, T, H]
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)

    H, Dq, Dv, Lkv = cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank

    with torch.no_grad():
        q_resid = A.q_a_layernorm(A.q_a_proj(x))                 # [1, T, 1536]
        q = A.q_b_proj(q_resid).view(1, T, H, Dq)                # [1, T, 64, 256]
        c_kv = A.kv_a_layernorm(A.kv_a_proj_with_mqa(x))         # [1, T, 512]  (no rope split)

        # ---- verify the dense claim with the REAL indexer helpers ----
        k_idx = A.indexer.k_norm(A.indexer.wk(x)).view(1, T, -1, cfg.index_head_dim).squeeze(2)
        gate_scores = F.linear(x, A.indexer.index_kpool_compress_gate)
        valid = torch.ones(1, T, 1, device=dev)
        packed = torch.cat([k_idx, gate_scores, valid], dim=-1)
        pool_keys, pool_indices, pool_valid = A.indexer.get_pooled_states(packed)
        visible = A.indexer.get_visible_tokens(valid.squeeze(-1).bool(), T, T)   # [1, T, T]
        n_pool = pool_keys.shape[1]
        select_k = min(A.indexer.index_topk // A.indexer.index_kpool, n_pool)
        # last query row: which tokens would be reachable if every selectable pool were chosen?
        pool_end = pool_indices[..., -1].clamp(0, T - 1)
        cand = visible[0, -1].gather(0, pool_end[0]) & pool_valid[0]
        reachable = set()
        for p in range(n_pool):
            if bool(cand[p]):
                reachable |= {int(i) for i in pool_indices[0, p] if int(i) >= 0}
        # the tail (index_kpool_always_select_tail) adds the trailing incomplete pool
        tail = set(range(len(reachable), T)) if A.indexer.index_kpool_always_select_tail else set()
        covered = reachable | tail
        dense = (select_k >= int(cand.sum())) and covered == set(range(T))

        # ---- dense causal MLA for the LAST position ----
        kv = A.kv_b_proj(c_kv).view(1, T, H, Dq + Dv)
        k_nope, v = torch.split(kv, [Dq, Dv], dim=-1)            # [1, T, 64, 256] each
        ql = q[0, -1]                                             # [64, 256]
        scores = torch.einsum('hd,thd->ht', ql, k_nope[0]) * A.scaling
        attn = scores.softmax(-1)
        ctx = torch.einsum('ht,thd->hd', attn, v[0])              # [64, 256]
        y = A.o_proj(ctx.reshape(1, -1))[0]

        # absorbed form, which is what the kernel implements: fold W_k into q, attend over the
        # 512-wide latent directly, expand only at the end through W_v.
        kvb = A.kv_b_proj.weight.view(H, Dq + Dv, Lkv)
        Wk, Wv = kvb[:, :Dq, :], kvb[:, Dq:, :]                   # [64,256,512] each
        qa = torch.einsum('hd,hdl->hl', ql, Wk)                   # [64, 512]
        s2 = torch.einsum('hl,tl->ht', qa, c_kv[0]) * A.scaling
        a2 = s2.softmax(-1)
        ctx2 = torch.einsum('ht,tl->hl', a2, c_kv[0])             # [64, 512]
        o2 = torch.einsum('hl,hdl->hd', ctx2, Wv)                 # [64, 256]
        absorb_err = (o2 - ctx).abs().max().item()

    o = a.out
    w(os.path.join(o, 'x.bin'), x[0, -1])
    w(os.path.join(o, 'c_kv_cache.bin'), c_kv[0])                 # [T, 512] the whole cache
    w(os.path.join(o, 'q_resid.bin'), q_resid[0, -1])
    w(os.path.join(o, 'q.bin'), q[0, -1])
    w(os.path.join(o, 'scores.bin'), scores)
    w(os.path.join(o, 'ctx.bin'), ctx)
    w(os.path.join(o, 'y.bin'), y)
    with open(os.path.join(o, 'meta.txt'), 'w') as f:
        f.write('layer=%d\nT=%d\nheads=%d\nqk_nope=%d\nv_head=%d\nkv_lora=%d\nq_lora=%d\n'
                'scaling=%s\nn_pool=%d\nselect_k=%d\ndense=%s\nseed=%d\n'
                % (a.layer, T, H, Dq, Dv, Lkv, cfg.q_lora_rank, A.scaling, n_pool, select_k, dense, a.seed))

    print('oracle -> %s   layer %d, T=%d' % (o, a.layer, T))
    print('  DSA at T=%d: %d pools, select_k=%d, covers all %d positions: %s'
          % (T, n_pool, select_k, T, dense))
    print('  absorbed-vs-expanded max abs diff: %.3e  (they are algebraically identical)' % absorb_err)
    print('  |q|=%.4f |ctx|=%.4f |y|=%.4f' % (q[0, -1].norm(), ctx.norm(), y.norm()))


if __name__ == '__main__':
    main()
