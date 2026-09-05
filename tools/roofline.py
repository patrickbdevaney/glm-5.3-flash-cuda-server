#!/usr/bin/env python3
"""roofline.py - per-token weight bytes (B_tok) for GLM-5.3-Flash-REAP50, read from the
checkpoint itself rather than from the config. Nothing here is estimated: every byte count
comes from a safetensors header.

Decode on this box is bandwidth-bound, so B_tok x achievable-bandwidth is the AR wall, and
every optimisation is either "read fewer bytes" or "read them closer to peak".
"""
import json, struct, glob, os, sys, collections, argparse

def read_headers(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, 'model-*.safetensors'))):
        with open(f, 'rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == '__metadata__':
                continue
            beg, end = v['data_offsets']
            out[k] = (v['dtype'], tuple(v['shape']), end - beg)
    return out

def classify(name):
    """Which per-token bucket does this tensor fall in at bs=1 decode?"""
    if name.startswith('model.visual.'):
        return 'vision tower (not on text AR path)'
    if name == 'model.language_model.norm.weight':
        return 'norms'
    if 'embed_tokens' in name:
        return 'embed (gathered, 1 row)'
    if name.startswith('lm_head'):
        return 'lm_head'
    if '.layers.' not in name:
        return 'other'
    L = int(name.split('.layers.')[1].split('.')[0])
    body = name.split('.layers.%d.' % L)[1]
    if body.startswith('mlp.experts.'):
        return 'moe routed experts'
    if body.startswith('mlp.shared_experts.'):
        return 'moe shared expert'
    # The dot is load-bearing. `mlp.gate` also prefixes `mlp.gate_proj.weight`, which is the
    # SwiGLU gate of the three DENSE MLPs -- a [12288, 4096] bf16, 0.302 G across the three, 6x
    # the real router. Without the dot it lands in this bucket and inflates 'moe router' from
    # 0.051 G to 0.352 G while under-reporting 'dense mlp' by the same amount. Caught by dprof:
    # the router mark measured 313 GB/s against a machine that tops out at 247, and a phase
    # cannot beat the memory system (OPTIMIZATION_LOG #9).
    if body.startswith('mlp.gate.'):
        return 'moe router'
    if body.startswith('mlp.'):
        return 'dense mlp (first_k_dense_replace)'
    if body.startswith('hc_'):
        return 'hyper-connections'
    if body.startswith('self_attn.indexer.'):
        return 'dsa indexer'
    if body.startswith('self_attn.'):
        # KDA layers carry q_conv1d/A_log/dt_bias; MLA layers carry kv_a_proj etc.
        kda = ('conv1d' in body or 'A_log' in body or 'dt_bias' in body or '_a_proj' in body
               and body.startswith('self_attn.f') or body.startswith('self_attn.g_')
               or body.startswith('self_attn.b_proj') or 'o_norm' in body)
        return 'kda' if _is_kda(L) else 'mla (dsa full-attn)'
    if body in ('input_layernorm.weight', 'post_attention_layernorm.weight'):
        return 'norms'
    if body.startswith(('eh_proj', 'enorm', 'hnorm', 'shared_head')):
        return 'mtp glue'
    return 'other'

KDA_LAYERS = set()

def _is_kda(L):
    return L in KDA_LAYERS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--bw', type=float, default=240.0,
                    help='achievable GB/s (Thor MEASURED 240 streaming / 212 contended / 273 spec)')
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.model, 'config.json')))
    t = cfg.get('text_config', cfg)
    global KDA_LAYERS
    KDA_LAYERS = set(t['linear_attn_config']['kda_layers'])
    n_layer = t['num_hidden_layers']
    mtp_layer = n_layer  # index n_layer is the MTP block
    n_exp = t['n_routed_experts']
    k_exp = t['num_experts_per_tok']

    H = read_headers(a.model)
    tot = sum(v[2] for v in H.values())

    # ---- resident bytes, by bucket ----
    res = collections.Counter()
    for k, v in H.items():
        res[classify(k)] += v[2]

    # ---- per-token bytes ----
    # everything is read once per token EXCEPT routed experts (k of n) and embed (one row).
    per_tok = collections.Counter()
    mtp_bytes = 0
    for k, v in H.items():
        c = classify(k)
        nb = v[2]
        if '.layers.' in k and int(k.split('.layers.')[1].split('.')[0]) == mtp_layer:
            mtp_bytes += nb if c != 'moe routed experts' else nb * k_exp / n_exp
            continue                      # MTP is not on the AR path
        if c == 'vision tower (not on text AR path)':
            continue
        if c == 'moe routed experts':
            nb = nb * k_exp / n_exp
        elif c == 'embed (gathered, 1 row)':
            nb = v[1][1] * 2              # one row of hidden_size, bf16
        per_tok[c] += nb

    GB = 1e9
    print('=' * 74)
    print('GLM-5.3-Flash-REAP50-NVFP4  roofline')
    print('=' * 74)
    print('%-34s %s' % ('checkpoint', a.model))
    print('%-34s %d tensors, %.2f GiB on disk' % ('size', len(H), tot / 2**30))
    print('%-34s %d (%d KDA linear + %d MLA/DSA full) + 1 MTP'
          % ('layers', n_layer, len(KDA_LAYERS), n_layer - len(KDA_LAYERS)))
    print('%-34s %d of %d + %d shared' % ('experts/token', k_exp, n_exp, t['n_shared_experts']))
    print()
    print('%-34s %10s %8s   %10s %8s' % ('bucket', 'RESIDENT', '%', 'PER-TOKEN', '%'))
    print('-' * 74)
    b_tok = sum(per_tok.values())
    for c, _ in res.most_common():
        print('%-34s %9.2f G %7.1f%%   %9.3f G %7.1f%%'
              % (c, res[c] / GB, 100 * res[c] / tot, per_tok[c] / GB,
                 100 * per_tok[c] / b_tok if b_tok else 0))
    print('-' * 74)
    print('%-34s %9.2f G            %9.3f G' % ('TOTAL', tot / GB, b_tok / GB))
    print('%-34s %9.2f G  (not on the AR path)' % ('  of which MTP block', mtp_bytes / GB))
    print()
    print('AR wall (bandwidth only, no kernel inefficiency):')
    for bw, tag in ((212, 'contended'), (240, 'achievable  <- operative'), (273, 'spec peak')):
        print('  @ %3d GB/s (%-20s): %6.2f tok/s  (%5.1f ms/tok)'
              % (bw, tag, bw * GB / b_tok, 1000 * b_tok / (bw * GB)))
    print()

    # ---- what quantising the bf16 remainder would buy ----
    bf16_dense = sum(v[2] for k, v in H.items()
                     if v[0] == 'BF16' and classify(k) in
                     ('kda', 'mla (dsa full-attn)', 'dsa indexer', 'lm_head',
                      'dense mlp (first_k_dense_replace)', 'hyper-connections'))
    # NVFP4 stores 4 bits/weight + fp8 scale per 16 -> 0.5 + 0.0625 bytes vs 2.0
    ratio = (0.5 + 1 / 16) / 2.0
    saved = 0.0
    for c in ('kda', 'mla (dsa full-attn)', 'dsa indexer', 'lm_head',
              'dense mlp (first_k_dense_replace)', 'hyper-connections'):
        saved += per_tok[c] * (1 - ratio)
    print('IF the bf16 dense weights were NVFP4 too (%.2f GiB resident today):' % (bf16_dense / 2**30))
    print('  B_tok  %.3f G -> %.3f G   (-%.1f%%)'
          % (b_tok / GB, (b_tok - saved) / GB, 100 * saved / b_tok))
    print('  AR wall @240 GB/s  %.2f -> %.2f tok/s' % (240 * GB / b_tok, 240 * GB / (b_tok - saved)))
    print('  resident  %.2f -> %.2f GiB  (Thor envelope ~117 GiB)'
          % (tot / 2**30, (tot - bf16_dense * (1 - ratio)) / 2**30))
    print()

    # ---- KDA recurrent state ----
    lc = t['linear_attn_config']
    st = len(KDA_LAYERS) * lc['num_heads'] * lc['head_dim'] * lc['head_dim'] * 4
    conv = len(KDA_LAYERS) * 3 * lc['num_heads'] * lc['head_dim'] * (lc['short_conv_kernel_size'] - 1) * 4
    print('KDA recurrent state (context-INDEPENDENT):')
    print('  %d layers x %d heads x %d x %d fp32 = %.2f MiB' %
          (len(KDA_LAYERS), lc['num_heads'], lc['head_dim'], lc['head_dim'], st / 2**20))
    print('  conv windows (k-1=%d)                = %.2f MiB' % (lc['short_conv_kernel_size'] - 1, conv / 2**20))
    print('  read+write per token                 = %.3f G (%.1f%% of B_tok)'
          % (2 * (st + conv) / GB, 100 * 2 * (st + conv) / b_tok))
    print()

    # ---- MLA KV cache, which IS context-dependent ----
    kv = t['kv_lora_rank']
    n_full = n_layer - len(KDA_LAYERS)
    for ctx in (8192, 32768, 131072):
        b = n_full * ctx * kv * 2          # bf16 latent, MQA: one kv_lora vector per token
        print('  MLA latent KV @ %6d ctx: %6.2f MiB  (%d full layers x %d)' % (ctx, b / 2**20, n_full, kv))

if __name__ == '__main__':
    main()
