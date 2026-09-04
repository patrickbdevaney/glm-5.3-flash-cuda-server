#!/usr/bin/env python3
"""gen_moe.py - PyTorch oracle for the MoE block (router + NVFP4 experts + shared expert).

Dumps only activations and routing decisions, NOT weights: tests/gate_moe.cu reads the packed
NVFP4 experts straight out of the checkpoint shards, so the gate exercises the real loader and
the real 4-bit layout rather than a convenient fp32 copy.

The dequantisation here is the hand-rolled formula, already proven bit-identical to
compressed-tensors' NVFP4PackedCompressor.decompress (glm-5.3-reap/scripts/nvfp4_dequant_check.py,
0.000e+00 on every sampled tensor).
"""
import argparse, os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

REAP = os.path.expanduser('~/glm-5.3-reap')
sys.path.insert(0, os.path.join(REAP, 'scripts'))

KE2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32)


def dequant(pk, sc, gs, out_f, in_f):
    """NVFP4 -> fp32. pk uint8 [out, in/2] (low nibble first), sc fp8 [out, in/16], gs f32 [1]."""
    flat = pk.reshape(-1)
    low, high = flat & 0x0F, (flat >> 4) & 0x0F
    comb = torch.stack((low, high), dim=1).reshape(out_f, in_f).long()
    vals = KE2M1[comb & 7] * torch.where((comb & 8).bool(), -1.0, 1.0)
    out = vals.reshape(out_f, in_f // 16, 16) * (sc.float().reshape(out_f, in_f // 16, 1) / gs)
    return out.reshape(out_f, in_f)


def w(path, t):
    t.detach().to(torch.float32).cpu().numpy().ravel().astype('<f4').tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(REAP, 'output/glm-5.3-flash-reap50-nvfp4-pass2'))
    ap.add_argument('--layer', type=int, default=3, help='must be a sparse (MoE) layer')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'moe'))
    ap.add_argument('--seed', type=int, default=99)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoConfig
    from stream_saliency import ShardReader
    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True).text_config
    assert cfg.mlp_layer_types[a.layer] == 'sparse', 'layer %d is dense' % a.layer

    R = ShardReader(a.model)
    P = 'model.language_model.layers.%d.mlp.' % a.layer
    H, I, E, K = cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_routed_experts, cfg.num_experts_per_tok

    # a realistic activation: a real embedding row, rms-normalised the way post_attention_layernorm
    # would leave it. Random noise would route differently from anything the model ever sees.
    emb = R.get('model.language_model.embed_tokens.weight')
    g = torch.Generator().manual_seed(a.seed)
    x = emb[torch.randint(0, cfg.vocab_size, (1,), generator=g)[0]].to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean() + cfg.rms_norm_eps)

    # ---- router ----
    gw = R.get(P + 'gate.weight').to(torch.float32)                 # [E, H]
    bias = R.get(P + 'gate.e_score_correction_bias').to(torch.float32)
    logits = F.linear(x.unsqueeze(0), gw)[0]                        # [E]
    scores = logits.sigmoid()
    choice = scores + bias
    # n_group == topk_group == 1, so the group mask selects everything and is a no-op.
    assert cfg.n_group == 1 and cfg.topk_group == 1
    idx = torch.topk(choice, k=K, dim=-1, sorted=False)[1]
    wts = scores[idx]
    if cfg.norm_topk_prob:
        wts = wts / (wts.sum() + 1e-20)
    wts = wts * cfg.routed_scaling_factor

    # ---- experts ----
    lim = cfg.swiglu_limit

    def expert_mlp(pfx, inter):
        def dq(n, o, i):
            return dequant(R.get(pfx + n + '.weight_packed'), R.get(pfx + n + '.weight_scale'),
                           R.get(pfx + n + '.weight_global_scale').float(), o, i)
        gp, up, dn = dq('gate_proj', inter, H), dq('up_proj', inter, H), dq('down_proj', H, inter)
        gt = (x @ gp.T).clamp(max=lim)
        u = (x @ up.T).clamp(-lim, lim)
        return (F.silu(gt) * u) @ dn.T, gt, u

    routed = torch.zeros(H)
    acts = []
    for s in range(K):
        e = int(idx[s])
        y, gt, u = expert_mlp(P + 'experts.%d.' % e, I)
        acts.append(F.silu(gt) * u)
        routed += y * wts[s]
    shared, _, _ = expert_mlp(P + 'shared_experts.', I * cfg.n_shared_experts)
    out = routed + shared

    o = a.out
    w(os.path.join(o, 'x.bin'), x)
    w(os.path.join(o, 'logits.bin'), logits)
    w(os.path.join(o, 'scores.bin'), scores)
    w(os.path.join(o, 'topk_w.bin'), wts)
    idx.to(torch.int32).cpu().numpy().astype('<i4').tofile(os.path.join(o, 'topk_idx.bin'))
    w(os.path.join(o, 'act0.bin'), acts[0])          # silu(gate)*up for routing slot 0
    w(os.path.join(o, 'routed.bin'), routed)
    w(os.path.join(o, 'shared.bin'), shared)
    w(os.path.join(o, 'y.bin'), out)
    with open(os.path.join(o, 'meta.txt'), 'w') as f:
        f.write('layer=%d\nhidden=%d\ninter=%d\nn_expert=%d\ntopk=%d\nscale=%s\nlimit=%s\nseed=%d\n'
                % (a.layer, H, I, E, K, cfg.routed_scaling_factor, lim, a.seed))

    print('oracle -> %s' % o)
    print('  layer %d  experts %s' % (a.layer, sorted(int(i) for i in idx)))
    print('  weights  %s' % ' '.join('%.4f' % v for v in wts))
    print('  |x|=%.4f |routed|=%.4f |shared|=%.4f |y|=%.4f' % (x.norm(), routed.norm(), shared.norm(), out.norm()))


if __name__ == '__main__':
    main()
