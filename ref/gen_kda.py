#!/usr/bin/env python3
"""gen_kda.py - PyTorch oracle for the KDA (Kimi Delta Attention) decode step.

Builds layer 0 of the REAL checkpoint with the upstream `transformers` module, runs a prefill
then a single-token decode, and dumps every input, intermediate and output as raw fp32 for
tests/gate_kda.cu to load.

Deliberately CPU-only by default: the GPU on this box is often busy with an unattended stage, and
an oracle that races it produces slow, noisy runs. Pass --device cuda to override.

Outputs (ref/kda/*.bin, all little-endian fp32 unless noted):
  meta.txt            key=value geometry, so the gate cannot silently disagree about shapes
  x.bin               [HIDDEN]            decode-step input hidden state
  conv_state_in.bin   [3*QKV][3]          rolling conv window BEFORE the step (channel-major)
  S_in.bin            [H][Dk][Dv]         recurrent state BEFORE the step
  qkv_conv.bin        [3*QKV]             after depthwise conv + silu
  g.bin               [H][Dk]             forget gate (already -5*sigmoid(...), i.e. log-space)
  beta.bin            [H]
  q_n.bin, k_n.bin    [H][Dk]             after l2norm (q also after 1/sqrt(Dk) scaling)
  core_out.bin        [H][Dv]             recurrence output, before o_norm
  S_out.bin           [H][Dk][Dv]         recurrent state AFTER the step
  gate.bin            [H][Dv]             g_b(g_a(x))
  normed.bin          [H][Dv]             o_norm(core_out, gate)
  y.bin               [HIDDEN]            final o_proj output
  w_*.bin                                 the layer's weights, fp32, in the exact layout the
                                          kernel indexes (so the gate needs no loader)
"""
import argparse, json, os, sys, struct
import numpy as np
import torch

REAP = os.path.expanduser('~/glm-5.3-reap')
sys.path.insert(0, os.path.join(REAP, 'scripts'))


def w(path, t):
    a = t.detach().to(torch.float32).cpu().numpy().ravel()
    a.astype('<f4').tofile(path)
    return a.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(REAP, 'output/glm-5.3-flash-reap50-nvfp4-pass2'))
    ap.add_argument('--layer', type=int, default=0, help='must be a KDA (linear_attention) layer')
    ap.add_argument('--prefill', type=int, default=17, help='tokens of prefill before the decode step')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'kda'))
    ap.add_argument('--seed', type=int, default=1234)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoConfig
    from stages.s03_saliency import _build_layer
    from stream_saliency import ShardReader

    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
    tcfg = cfg.text_config
    assert tcfg.layer_types[a.layer] == 'linear_attention', \
        'layer %d is %s, not KDA' % (a.layer, tcfg.layer_types[a.layer])

    dev = torch.device(a.device)
    reader = ShardReader(a.model)
    layer = _build_layer(tcfg, a.layer, reader, torch.float32)
    layer = layer.to(dev).eval()
    attn = layer.self_attn

    H, D = attn.num_heads, attn.head_dim
    QKV = H * D
    torch.manual_seed(a.seed)

    # ---- prefill, so the state we decode from is a real one, not zeros ----
    # Real embedding rows, not noise. A gate driven by N(0, 0.02) tensors runs the kernel in a
    # regime the model never sees: |core_out| comes out ~1e-3, where relative error is dominated by
    # cancellation and a real bug can hide inside the noise floor.
    emb = reader.get('model.language_model.embed_tokens.weight')
    g_ = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(0, tcfg.vocab_size, (a.prefill + 1,), generator=g_)
    rows = emb[ids].to(torch.float32).to(dev)
    xs = rows[:a.prefill].unsqueeze(0)
    x1 = rows[a.prefill:].unsqueeze(0)

    with torch.no_grad():
        # Drive the reference recurrence directly rather than through the Cache plumbing, so the
        # oracle depends only on math we are reimplementing, not on transformers' cache classes.
        from transformers.models.glm5_next.modeling_glm5_next import l2norm

        def qkv_of(x, conv_state):
            """conv + silu for a sequence; returns (q,k,v) [B,S,H,D] and the new conv window."""
            mixed = torch.cat([attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)], dim=-1)  # [B,S,3Q]
            mixed = mixed.transpose(1, 2)                                                # [B,3Q,S]
            padded = torch.cat([conv_state, mixed], dim=-1)                              # [B,3Q,k-1+S]
            wgt = attn.conv1d.weight.squeeze(1)                                          # [3Q,k]
            S = mixed.shape[-1]
            out = torch.zeros_like(mixed)
            for i in range(S):
                win = padded[:, :, i:i + attn.conv_kernel_size]                          # [B,3Q,k]
                out[:, :, i] = (win * wgt.unsqueeze(0)).sum(-1)
            out = torch.nn.functional.silu(out)
            new_state = padded[:, :, -(attn.conv_kernel_size - 1):]
            return out.transpose(1, 2), new_state                                        # [B,S,3Q]

        def gates_of(x):
            fg = attn.forget_gate(x)                        # [B,S,H,D], already -5*sigmoid(...)
            beta = torch.sigmoid(attn.b_proj(x))            # [B,S,H]
            gate = attn.g_b_proj(attn.g_a_proj(x)).view(*x.shape[:2], H, D)
            return fg, beta, gate

        def recur(q, k, v, g, beta, S0):
            """The upstream fp32 recurrence, one step at a time. S: [B,H,Dk,Dv]."""
            q = l2norm(q.float(), dim=-1, eps=1e-6)
            k = l2norm(k.float(), dim=-1, eps=1e-6)
            q = q * (D ** -0.5)
            Sst = S0
            outs = []
            for i in range(q.shape[1]):
                g_i = g[:, i][..., None].exp()               # [B,H,Dk,1]
                b_i = beta[:, i][..., None]                  # [B,H,1]
                Sst = Sst * g_i
                kv_mem = (Sst * k[:, i][..., None]).sum(dim=-2)      # [B,H,Dv]
                delta = (v[:, i] - kv_mem) * b_i
                Sst = Sst + k[:, i].unsqueeze(-1) * delta.unsqueeze(-2)
                outs.append((Sst * q[:, i].unsqueeze(-1)).sum(dim=-2))
            return torch.stack(outs, dim=1), Sst, q, k

        conv0 = torch.zeros(1, 3 * QKV, attn.conv_kernel_size - 1, device=dev)
        S0 = torch.zeros(1, H, D, D, device=dev)

        qkv_p, conv_p = qkv_of(xs, conv0)
        qp, kp, vp = qkv_p.split([QKV] * 3, dim=-1)
        shape = (1, a.prefill, H, D)
        fgp, betap, _ = gates_of(xs)
        _, S_pre, _, _ = recur(qp.view(shape), kp.view(shape), vp.view(shape), fgp, betap, S0)

        # ---- the decode step we actually gate ----
        qkv_c, conv_out = qkv_of(x1, conv_p)
        q1, k1, v1 = qkv_c.split([QKV] * 3, dim=-1)
        s1 = (1, 1, H, D)
        fg1, beta1, gate1 = gates_of(x1)
        core, S_out, q_n, k_n = recur(q1.view(s1), k1.view(s1), v1.view(s1), fg1, beta1, S_pre)
        normed = attn.o_norm(core.view(1, 1, H, D), gate1)
        y = attn.o_proj(normed.reshape(1, 1, -1))

    o = a.out
    w(os.path.join(o, 'x.bin'), x1[0, 0])
    w(os.path.join(o, 'conv_state_in.bin'), conv_p[0])          # [3Q, k-1]
    w(os.path.join(o, 'conv_state_out.bin'), conv_out[0])
    w(os.path.join(o, 'S_in.bin'), S_pre[0])                    # [H, Dk, Dv]
    w(os.path.join(o, 'qkv_conv.bin'), qkv_c[0, 0])
    w(os.path.join(o, 'g.bin'), fg1[0, 0])
    w(os.path.join(o, 'beta.bin'), beta1[0, 0])
    w(os.path.join(o, 'q_n.bin'), q_n[0, 0])
    w(os.path.join(o, 'k_n.bin'), k_n[0, 0])
    w(os.path.join(o, 'core_out.bin'), core[0, 0])
    w(os.path.join(o, 'S_out.bin'), S_out[0])
    w(os.path.join(o, 'gate.bin'), gate1[0, 0])
    w(os.path.join(o, 'normed.bin'), normed[0, 0])
    w(os.path.join(o, 'y.bin'), y[0, 0])

    # weights, fp32, row-major exactly as stored in the checkpoint
    for name, t in [
        ('q_proj', attn.q_proj.weight), ('k_proj', attn.k_proj.weight), ('v_proj', attn.v_proj.weight),
        ('o_proj', attn.o_proj.weight), ('conv1d', attn.conv1d.weight.squeeze(1)),
        ('f_a', attn.forget_gate.f_a_proj.weight), ('f_b', attn.forget_gate.f_b_proj.weight),
        ('dt_bias', attn.forget_gate.dt_bias), ('A_log', attn.forget_gate.A_log),
        ('b_proj', attn.b_proj.weight),
        ('g_a', attn.g_a_proj.weight), ('g_b', attn.g_b_proj.weight),
        ('o_norm', attn.o_norm.weight),
    ]:
        w(os.path.join(o, 'w_%s.bin' % name), t)

    with open(os.path.join(o, 'meta.txt'), 'w') as f:
        f.write('layer=%d\nhidden=%d\nheads=%d\nhead_dim=%d\nqkv_dim=%d\nconv_k=%d\n'
                'gate_rank=%d\nlower_bound=%s\nrms_eps=%s\nprefill=%d\nseed=%d\n'
                % (a.layer, tcfg.hidden_size, H, D, QKV, attn.conv_kernel_size,
                   attn.forget_gate.f_a_proj.out_features,
                   attn.forget_gate.safe_gate_lower_bound, attn.layer_norm_epsilon,
                   a.prefill, a.seed))

    print('oracle written to %s' % o)
    print('  layer %d  H=%d D=%d QKV=%d conv_k=%d  prefill=%d'
          % (a.layer, H, D, QKV, attn.conv_kernel_size, a.prefill))
    print('  |x|=%.4f  |core_out|=%.4f  |y|=%.4f  |S_out|=%.4f'
          % (x1.norm(), core.norm(), y.norm(), S_out.norm()))
    print('  g range [%.4f, %.4f]  beta range [%.4f, %.4f]'
          % (fg1.min(), fg1.max(), beta1.min(), beta1.max()))


if __name__ == '__main__':
    main()
