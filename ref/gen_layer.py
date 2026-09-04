#!/usr/bin/env python3
"""gen_layer.py - PyTorch oracle for a COMPLETE decoder layer: hyper-connections at both sites,
the sublayer, and the residual mix. Layer 0 = KDA + dense MLP + 2x mHC.

This is the gate that catches wiring bugs the per-kernel gates cannot: stream ordering, the
comb-transpose in the residual mix, which norm feeds which site, and the fact that GLM's mHC
carries FOUR residual streams rather than one.
"""
import argparse, os, sys
import torch

REAP = os.path.expanduser('~/glm-5.3-reap')
sys.path.insert(0, os.path.join(REAP, 'scripts'))


def w(path, t):
    t.detach().to(torch.float32).cpu().numpy().ravel().astype('<f4').tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(REAP, 'output/glm-5.3-flash-reap50-nvfp4-pass2'))
    ap.add_argument('--layer', type=int, default=0)
    ap.add_argument('--prefill', type=int, default=9)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'layer'))
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoConfig
    from stages.s03_saliency import _build_layer
    from stream_saliency import ShardReader

    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True).text_config
    R = ShardReader(a.model)
    layer = _build_layer(cfg, a.layer, R, torch.float32).eval()
    dev = next(layer.parameters()).device   # _build_layer places the layer on the GPU

    torch.manual_seed(a.seed)
    emb = R.get('model.language_model.embed_tokens.weight')
    g_ = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(0, cfg.vocab_size, (a.prefill + 1,), generator=g_)
    rows = emb[ids].to(torch.float32).to(dev)

    H = cfg.hc_mult
    # four residual streams, as the model carries them
    streams = rows.unsqueeze(1).repeat(1, H, 1).unsqueeze(0)     # [1, S, H, D]
    streams = streams * torch.linspace(0.7, 1.3, H, device=dev).view(1, 1, H, 1)   # break stream symmetry

    with torch.no_grad():
        # prefill to build real KDA state, then the single decode step we gate
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextRMSNorm
        pre = streams[:, :a.prefill]
        x1 = streams[:, a.prefill:]

        class Cache:
            """Minimal stand-in exposing exactly what Glm5NextTextLinearAttention touches."""
            def __init__(self, nl):
                self.layers = [type('L', (), {'conv_states': [None], 'recurrent_states': [None]})()
                               for _ in range(nl + 1)]
                self._has = [False] * (nl + 1)
            def has_previous_state(self, i): return self._has[i]
            def update_conv_state(self, mixed, i, conv_kernel_size):
                st = self.layers[i].conv_states[0]
                if st is None:
                    st = torch.zeros(mixed.shape[0], mixed.shape[1], conv_kernel_size - 1,
                                     device=mixed.device, dtype=mixed.dtype)
                out = torch.cat([st, mixed], dim=-1)
                self.layers[i].conv_states[0] = out[:, :, -(conv_kernel_size - 1):]
                return out
            def update_recurrent_state(self, s, i):
                self.layers[i].recurrent_states[0] = s
                self._has[i] = True

        cache = Cache(a.layer)
        layer(pre, attention_mask=None, past_key_values=cache)
        conv_in = cache.layers[a.layer].conv_states[0].clone()
        S_in = cache.layers[a.layer].recurrent_states[0].clone()

        # the step, with every intermediate captured
        residual = x1
        post_a, comb_a, coll_a = layer.attn_hc(x1)
        n_a = layer.input_layernorm(coll_a)
        sub_a = layer.self_attn(hidden_states=n_a, cache_params=cache, attention_mask=None)
        mid = post_a.unsqueeze(-1) * sub_a.unsqueeze(-2) + torch.matmul(comb_a.transpose(-1, -2), residual)

        residual2 = mid
        post_f, comb_f, coll_f = layer.ffn_hc(mid)
        n_f = layer.post_attention_layernorm(coll_f)
        sub_f = layer.mlp(n_f)
        out = post_f.unsqueeze(-1) * sub_f.unsqueeze(-2) + torch.matmul(comb_f.transpose(-1, -2), residual2)

    o = a.out
    w(os.path.join(o, 'streams_in.bin'), x1[0, 0])       # [H, D]
    w(os.path.join(o, 'conv_state_in.bin'), conv_in[0])
    w(os.path.join(o, 'S_in.bin'), S_in[0])
    w(os.path.join(o, 'post_a.bin'), post_a[0, 0]); w(os.path.join(o, 'comb_a.bin'), comb_a[0, 0])
    w(os.path.join(o, 'coll_a.bin'), coll_a[0, 0]);  w(os.path.join(o, 'n_a.bin'), n_a[0, 0])
    w(os.path.join(o, 'sub_a.bin'), sub_a[0, 0]);    w(os.path.join(o, 'mid.bin'), mid[0, 0])
    w(os.path.join(o, 'post_f.bin'), post_f[0, 0]);  w(os.path.join(o, 'comb_f.bin'), comb_f[0, 0])
    w(os.path.join(o, 'coll_f.bin'), coll_f[0, 0]);  w(os.path.join(o, 'n_f.bin'), n_f[0, 0])
    w(os.path.join(o, 'sub_f.bin'), sub_f[0, 0]);    w(os.path.join(o, 'out.bin'), out[0, 0])
    with open(os.path.join(o, 'meta.txt'), 'w') as f:
        f.write('layer=%d\nhc_mult=%d\nhidden=%d\nprefill=%d\nseed=%d\nblock=%s\nmlp=%s\n'
                % (a.layer, H, cfg.hidden_size, a.prefill, a.seed,
                   cfg.layer_types[a.layer], cfg.mlp_layer_types[a.layer]))
    print('oracle -> %s   layer %d (%s / %s)' % (o, a.layer, cfg.layer_types[a.layer],
                                                 cfg.mlp_layer_types[a.layer]))
    print('  |streams_in|=%.4f |sub_a|=%.4f |mid|=%.4f |sub_f|=%.4f |out|=%.4f'
          % (x1.norm(), sub_a.norm(), mid.norm(), sub_f.norm(), out.norm()))
    print('  post_a=%s' % ' '.join('%.4f' % v for v in post_a[0, 0]))
    print('  comb_a row sums=%s col sums=%s'
          % (' '.join('%.4f' % v for v in comb_a[0, 0].sum(-1)),
             ' '.join('%.4f' % v for v in comb_a[0, 0].sum(-2))))


if __name__ == '__main__':
    main()
