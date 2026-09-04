#!/usr/bin/env python3
"""dense_nvfp4_probe.py - does the 51% B_tok lever actually CONVERT?

ROOFLINE.md §3: 13.32 GiB of bf16 dense weights sit on the AR path untouched and are 76% of
B_tok. Quantising them to NVFP4 halves B_tok and doubles the AR wall. That is an arithmetic
claim about bytes. This asks the question that decides whether it is usable:

    what does NVFP4 do to the OUTPUT, tensor family by tensor family?

It matters most for KDA. Upstream says explicitly that KDA states are "susceptible to rounding
errors", and q/k/v_proj feed a recurrence that accumulates over the whole sequence - so an error
that is harmless in a feed-forward projection is not obviously harmless there. This measures the
drift in the layer output AND in the recurrent state after a real prefill, per family, so the
families can be quantised in order of what survives rather than all at once.

Writes nothing. No checkpoint is modified.
"""
import argparse, os, sys
import torch

REAP = os.path.expanduser('~/glm-5.3-reap')
sys.path.insert(0, os.path.join(REAP, 'scripts'))

KE2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])


def quant_nvfp4(w, group=16):
    """Round-trip a tensor through the NVFP4 grid the checkpoint uses: per-group-of-16 fp8-e4m3
    scale, one fp32 global scale per tensor, e2m1 values. Returns the dequantised tensor."""
    out_f, in_f = w.shape
    assert in_f % group == 0
    g = w.float().reshape(out_f, in_f // group, group)
    amax = g.abs().amax(-1, keepdim=True)
    # global scale chosen as the reference quantiser does: map the tensor max onto the fp8 range
    gs = (448.0 * 6.0) / w.float().abs().amax().clamp(min=1e-12)
    scale = (amax / 6.0 * gs).clamp(min=1e-12)
    scale = scale.to(torch.float8_e4m3fn).float()          # scales are stored fp8-e4m3
    eff = scale / gs
    q = (g / eff.clamp(min=1e-30))
    lut = KE2M1.to(w.device)
    idx = (q.abs().unsqueeze(-1) - lut).abs().argmin(-1)   # nearest e2m1 magnitude
    deq = lut[idx] * torch.sign(q) * eff
    return deq.reshape(out_f, in_f).to(w.dtype)


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-20)).item()


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(REAP, 'output/glm-5.3-flash-reap50-nvfp4-pass2'))
    ap.add_argument('--layer', type=int, default=0, help='a KDA layer: the hard case')
    ap.add_argument('--prefill', type=int, default=64)
    ap.add_argument('--seed', type=int, default=3)
    a = ap.parse_args()

    from transformers import AutoConfig
    from stages.s03_saliency import _build_layer
    from stream_saliency import ShardReader

    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True).text_config
    assert cfg.layer_types[a.layer] == 'linear_attention'
    R = ShardReader(a.model)
    layer = _build_layer(cfg, a.layer, R, torch.float32).eval()
    dev = next(layer.parameters()).device
    A = layer.self_attn

    emb = R.get('model.language_model.embed_tokens.weight')
    g_ = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(0, cfg.vocab_size, (a.prefill,), generator=g_)
    x = emb[ids].to(torch.float32).to(dev).unsqueeze(0)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)

    print('=' * 78)
    print('NVFP4 round-trip on the bf16 dense weights of layer %d (KDA), prefill %d'
          % (a.layer, a.prefill))
    print('=' * 78)
    print('%-22s %12s %12s %12s' % ('tensor', 'rel err', 'cos', 'MB saved/tok'))
    print('-' * 78)

    FAM = {
        'kda q_proj':  A.q_proj, 'kda k_proj': A.k_proj, 'kda v_proj': A.v_proj,
        'kda o_proj':  A.o_proj,
        'kda f_b':     A.forget_gate.f_b_proj, 'kda g_b': A.g_b_proj,
        'dense gate':  layer.mlp.gate_proj if hasattr(layer.mlp, 'gate_proj') else None,
        'dense up':    layer.mlp.up_proj if hasattr(layer.mlp, 'up_proj') else None,
        'dense down':  layer.mlp.down_proj if hasattr(layer.mlp, 'down_proj') else None,
    }
    saved_ratio = 1.0 - (0.5 + 1 / 16) / 2.0
    orig = {}
    for name, mod in FAM.items():
        if mod is None:
            continue
        w = mod.weight.data
        q = quant_nvfp4(w)
        orig[name] = (mod, w.clone())
        mb = w.numel() * 2 * saved_ratio / 1e6
        print('%-22s %12.5f %12.7f %12.1f' % (name, rel(q, w), cos(q, w), mb))

    # ---- what it does to the LAYER, and to the recurrent state ----
    def run(with_quant):
        for name, (mod, w0) in orig.items():
            mod.weight.data = quant_nvfp4(w0) if with_quant else w0.clone()
        H = cfg.hc_mult
        h = x.unsqueeze(2).expand(-1, -1, H, -1).contiguous()

        class C:
            def __init__(s):
                s.layers = [type('L', (), {'conv_states': [None], 'recurrent_states': [None]})()
                            for _ in range(a.layer + 1)]
                s._h = [False] * (a.layer + 1)
            def has_previous_state(s, i): return s._h[i]
            def update_conv_state(s, m, i, k):
                st = s.layers[i].conv_states[0]
                if st is None:
                    st = torch.zeros(m.shape[0], m.shape[1], k - 1, device=m.device, dtype=m.dtype)
                o = torch.cat([st, m], dim=-1)
                s.layers[i].conv_states[0] = o[:, :, -(k - 1):]
                return o
            def update_recurrent_state(s, st, i):
                s.layers[i].recurrent_states[0] = st
                s._h[i] = True

        c = C()
        with torch.no_grad():
            out, _ = layer(h, attention_mask=None, past_key_values=c)
        return out, c.layers[a.layer].recurrent_states[0]

    ref_out, ref_state = run(False)
    q_out, q_state = run(True)
    for name, (mod, w0) in orig.items():
        mod.weight.data = w0                       # leave the module as we found it

    print('-' * 78)
    print('layer output   rel err %.5f   cos %.7f' % (rel(q_out, ref_out), cos(q_out, ref_out)))
    print('KDA state      rel err %.5f   cos %.7f   (after %d tokens of accumulation)'
          % (rel(q_state, ref_state), cos(q_state, ref_state), a.prefill))
    print()
    print('Read this as: if the state cosine degrades much faster than the per-tensor cosines,')
    print('the recurrence is amplifying quantisation error and q/k/v_proj should stay bf16 even')
    print('though o_proj and the MLP can go. Quantise in the order lm_head -> o_proj -> q/k/v,')
    print('gating perplexity at each step (ROOFLINE.md §3).')


if __name__ == '__main__':
    main()
