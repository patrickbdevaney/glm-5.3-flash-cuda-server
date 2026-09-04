#!/usr/bin/env python3
"""gen_stack.py - PyTorch oracle for a STACK of decoder layers, to gate the engine's wiring.

The per-layer gates prove each kernel; this proves the thing between them — that the four mHC
residual streams propagate correctly from layer to layer, that the KDA state and conv window are
kept per layer and not shared, and that the embedding is broadcast across all four streams.

Default is 3 layers (0,1,2: KDA + dense MLP), which is cheap. --layers 4+ pulls in layer 3, whose
144 NVFP4 experts cost ~20 GiB and minutes of CPU to materialise as fp32.
"""
import argparse, os, sys
import torch

REAP = os.path.expanduser('~/glm-5.3-reap')
sys.path.insert(0, os.path.join(REAP, 'scripts'))


def w(p, t): t.detach().to(torch.float32).cpu().numpy().ravel().astype('<f4').tofile(p)


class Cache:
    """Minimal stand-in exposing exactly what Glm5NextTextLinearAttention touches."""
    def __init__(self, nl):
        self.layers = [type('L', (), {'conv_states': [None], 'recurrent_states': [None]})()
                       for _ in range(nl)]
        self._has = [False] * nl

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(REAP, 'output/glm-5.3-flash-reap50-nvfp4-pass2'))
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--steps', type=int, default=4, help='decode steps to run')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'stack'))
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoConfig
    from stages.s03_saliency import _build_layer
    from stream_saliency import ShardReader

    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True).text_config
    R = ShardReader(a.model)
    layers = [_build_layer(cfg, i, R, torch.float32).eval() for i in range(a.layers)]
    dev = next(layers[0].parameters()).device

    emb = R.get('model.language_model.embed_tokens.weight')
    g_ = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(0, cfg.vocab_size, (a.steps,), generator=g_)

    cache = Cache(a.layers)
    outs = []
    with torch.no_grad():
        for t in range(a.steps):
            x = emb[ids[t]].to(torch.float32).to(dev).view(1, 1, -1)
            # the model broadcasts one embedding across all hc_mult residual streams
            h = x.unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
            for i, L in enumerate(layers):
                h, _ = L(h, attention_mask=None, past_key_values=cache)
            outs.append(h[0, 0].clone())

    ids.to(torch.int32).cpu().numpy().astype('<i4').tofile(os.path.join(a.out, 'ids.bin'))
    for t, o in enumerate(outs):
        w(os.path.join(a.out, 'streams_%d.bin' % t), o)
    with open(os.path.join(a.out, 'meta.txt'), 'w') as f:
        f.write('layers=%d\nsteps=%d\nhc_mult=%d\nhidden=%d\nseed=%d\n'
                % (a.layers, a.steps, cfg.hc_mult, cfg.hidden_size, a.seed))
    print('oracle -> %s   %d layers, %d steps' % (a.out, a.layers, a.steps))
    for t, o in enumerate(outs):
        print('  step %d  token %6d  |streams|=%.5f' % (t, int(ids[t]), o.norm()))


if __name__ == '__main__':
    main()
