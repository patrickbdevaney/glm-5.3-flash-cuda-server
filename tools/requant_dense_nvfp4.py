#!/usr/bin/env python3
"""requant_dense_nvfp4.py — the B_tok lever (ROOFLINE.md §3).

13.3 GiB of bf16 dense weights sit on the AR path untouched and are 76% of B_tok. REAP+NVFP4
quantised the *experts* — 83% of the disk, 24% of the bytes per token. This quantises what is
actually read every token.

It writes an OVERLAY, not a new checkpoint: only the converted tensors, ~3.7 GiB, as
`<name>.weight_packed` / `.weight_scale` / `.weight_global_scale` in exactly the layout the
experts already use. The engine loads the base checkpoint and this directory on top, and prefers
the NVFP4 form wherever it exists. Consequences of that choice, both deliberate:

  * nothing is destroyed and nothing is rewritten — the base checkpoint is opened read-only;
  * a family is disabled by not emitting it, so `--families` IS the gate, at zero runtime cost.

The dequant convention is fixed by kernels/moe.cu and reproduced here exactly:

    w = kE2M1[nib & 7] * (-1)^(nib >> 3) * fp8_e4m3(scale_byte) * (1 / global_scale)

so any drift between this file and that kernel is a silent accuracy bug. gate_nvfp4 checks it on
the real checkpoint rather than trusting the comment.
"""
import argparse, json, os, re, struct, sys, time
import numpy as np

# ---- the two number formats, built the same way the kernel builds them --------------------------

E2M1 = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=np.float32)
E2M1_MID = ((E2M1[1:] + E2M1[:-1]) / 2).astype(np.float32)      # 7 split points -> code 0..7


def _e4m3_table():
    """Decode every positive e4m3 code. Same arithmetic as fp8e4m3() in kernels/moe.cu."""
    c = np.arange(127, dtype=np.int32)                           # 0x7F is NaN, excluded
    e, m = (c >> 3) & 0xF, c & 0x7
    sub = m.astype(np.float32) * (1. / 8.) * (1. / 64.)
    nrm = (1. + m.astype(np.float32) / 8.) * np.exp2((e - 7).astype(np.float32))
    return np.where(e == 0, sub, nrm).astype(np.float32)


E4M3 = _e4m3_table()                                             # monotonically increasing
E4M3_MID = ((E4M3[1:] + E4M3[:-1]) / 2).astype(np.float32)
FP8_MAX = float(E4M3[-1])                                        # 448.0

GROUP = 16


def bf16_to_f32(raw):
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (u << 16).view(np.float32)


def quant_block(w, gs):
    """w [rows, in] fp32, gs the tensor's global scale. -> packed u8 [rows, in/2], scale u8 [rows, in/16]."""
    rows, K = w.shape
    g = w.reshape(rows, K // GROUP, GROUP)
    amax = np.abs(g).max(-1)                                     # [rows, K/16]
    # scale is stored as fp8-e4m3; encode by nearest table entry, which is what makes the
    # round-trip here identical to the one the kernel performs on the way back.
    s = np.clip(amax / 6.0 * gs, 0.0, FP8_MAX)
    code = np.searchsorted(E4M3_MID, s).astype(np.uint8)
    eff = (E4M3[code] / gs).astype(np.float32)                   # the scale the kernel will apply
    q = g / np.maximum(eff, 1e-30)[:, :, None]
    idx = np.searchsorted(E2M1_MID, np.abs(q)).astype(np.uint8)  # 0..7 magnitude
    idx |= (np.signbit(q) & (idx != 0)).astype(np.uint8) << 3    # -0 is not a distinct code
    idx = idx.reshape(rows, K)
    packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).astype(np.uint8)   # LOW nibble first
    return packed, code


# ---- which tensors, and under which gate --------------------------------------------------------
#
# Only tensors consumed through gemv()/gemm() are here. kv_b_proj is bf16 on purpose: MLA reads it
# strided inside k_absorb_q/k_expand_v, not through gemv, so converting it would need a second
# kernel for 1.7% of B_tok. mlp.gate.weight (the router) is bf16 on purpose too — 0.05 GiB, and it
# decides which experts run.
FAMILIES = {
    'kda_qkv':   r'layers\.\d+\.self_attn\.[qkv]_proj\.weight$',
    'o_proj':    r'layers\.\d+\.self_attn\.o_proj\.weight$',
    'kda_gates': r'layers\.\d+\.self_attn\.(f_a|f_b|g_a|g_b|b)_proj\.weight$',
    'mla':       r'layers\.\d+\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa)\.weight$',
    'indexer':   r'layers\.\d+\.self_attn\.indexer\.(wq_b|wk|weights_proj)\.weight$|index_kpool_compress_gate$',
    'dense_mlp': r'layers\.\d+\.mlp\.(gate|up|down)_proj\.weight$',
    'lm_head':   r'^lm_head\.weight$',
}
DEFAULT = 'kda_qkv,o_proj,lm_head,dense_mlp,mla,indexer,kda_gates'


def family_of(name, sel):
    if '.visual.' in name or 'embed_tokens' in name:
        return None
    for f in sel:
        if re.search(FAMILIES[f], name):
            return f
    return None


# ---- safetensors ---------------------------------------------------------------------------------

def read_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


class ShardWriter:
    """Writes one output shard, padding every tensor to 16 bytes.

    The base checkpoint aligns to 4, which is why 777 weight_packed tensors per shard land at
    offset 4 mod 8 and a uint2 load on them faults (OPTIMIZATION_LOG #2). Nothing produced here
    inherits that: everything is 16-byte aligned, absolutely, header padding included.
    """
    def __init__(self, path):
        self.path, self.buf, self.off, self.meta = path, [], 0, {}

    def add(self, name, arr, dtype):
        b = arr.tobytes()
        pad = (-self.off) % 16
        if pad:
            self.buf.append(b'\0' * pad); self.off += pad
        self.meta[name] = {'dtype': dtype, 'shape': list(arr.shape),
                           'data_offsets': [self.off, self.off + len(b)]}
        self.buf.append(b); self.off += len(b)

    def flush(self):
        hdr = json.dumps(self.meta, separators=(',', ':')).encode()
        hdr += b' ' * ((-(8 + len(hdr))) % 16)                   # data starts 16-byte aligned
        with open(self.path, 'wb') as f:
            f.write(struct.pack('<Q', len(hdr))); f.write(hdr)
            for b in self.buf:
                f.write(b)
        return len(hdr) + 8 + self.off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.expanduser(
        '~/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2'))
    ap.add_argument('--out', default=os.path.expanduser(
        '~/glm-5.3-reap/output/glm-5.3-flash-dense-nvfp4-overlay'))
    ap.add_argument('--families', default=DEFAULT)
    ap.add_argument('--rows', type=int, default=8192, help='row block; bounds peak RSS')
    ap.add_argument('--shard-bytes', type=int, default=2 << 30)
    a = ap.parse_args()

    sel = [f.strip() for f in a.families.split(',') if f.strip()]
    for f in sel:
        if f not in FAMILIES:
            sys.exit('unknown family %s; known: %s' % (f, ','.join(FAMILIES)))
    os.makedirs(a.out, exist_ok=True)

    shards = sorted(f for f in os.listdir(a.model) if f.endswith('.safetensors'))
    todo = []
    for sh in shards:
        h, base = read_header(os.path.join(a.model, sh))
        for name, t in h.items():
            if name == '__metadata__' or t['dtype'] != 'BF16':
                continue
            fam = family_of(name, sel)
            if fam and len(t['shape']) == 2 and t['shape'][1] % GROUP == 0:
                todo.append((sh, base, name, t, fam))
    tot_in = sum(np.prod(t['shape']) * 2 for _, _, _, t, _ in todo)
    print('%d tensors, %.3f GiB bf16 in, families %s' % (len(todo), tot_in / 2**30, ','.join(sel)),
          flush=True)

    idxmap, written, si = {}, 0, 0
    W = ShardWriter(os.path.join(a.out, 'overlay-%05d.safetensors' % si))
    t0 = time.time()
    cur_sh, fh = None, None
    for n, (sh, base, name, t, fam) in enumerate(todo):
        if sh != cur_sh:
            if fh: fh.close()
            fh = open(os.path.join(a.model, sh), 'rb'); cur_sh = sh
        out_f, in_f = t['shape']
        o0, o1 = t['data_offsets']
        # pass 1: the tensor-global amax, which sets the global scale.
        amax = 0.0
        for r in range(0, out_f, a.rows):
            nr = min(a.rows, out_f - r)
            fh.seek(base + o0 + r * in_f * 2)
            amax = max(amax, float(np.abs(bf16_to_f32(fh.read(nr * in_f * 2))).max()))
        gs = np.float32((448.0 * 6.0) / max(amax, 1e-12))
        # pass 2: quantise, block of rows at a time.
        packed = np.empty((out_f, in_f // 2), np.uint8)
        scale = np.empty((out_f, in_f // GROUP), np.uint8)
        for r in range(0, out_f, a.rows):
            nr = min(a.rows, out_f - r)
            fh.seek(base + o0 + r * in_f * 2)
            w = bf16_to_f32(fh.read(nr * in_f * 2)).reshape(nr, in_f)
            packed[r:r + nr], scale[r:r + nr] = quant_block(w, gs)
        stem = name[:-len('.weight')] if name.endswith('.weight') else name
        W.add(stem + '.weight_packed', packed, 'U8')
        W.add(stem + '.weight_scale', scale, 'F8_E4M3')
        W.add(stem + '.weight_global_scale', np.array([gs], np.float32), 'F32')
        for suf in ('.weight_packed', '.weight_scale', '.weight_global_scale'):
            idxmap[stem + suf] = os.path.basename(W.path)
        written += packed.nbytes + scale.nbytes + 4
        if W.off >= a.shard_bytes:
            print('  shard %d: %.2f GiB' % (si, W.flush() / 2**30), flush=True)
            si += 1
            W = ShardWriter(os.path.join(a.out, 'overlay-%05d.safetensors' % si))
        if (n + 1) % 25 == 0 or n + 1 == len(todo):
            print('  [%4d/%d] %-62s %6.1f GiB out, %5.0fs'
                  % (n + 1, len(todo), name[-60:], written / 2**30, time.time() - t0), flush=True)
    if fh: fh.close()
    if W.meta:
        print('  shard %d: %.2f GiB' % (si, W.flush() / 2**30), flush=True)
    json.dump({'metadata': {'total_size': written}, 'weight_map': idxmap},
              open(os.path.join(a.out, 'model.safetensors.index.json'), 'w'))
    print('DONE  %.3f GiB in -> %.3f GiB out  (%.1f%%)  in %.0fs'
          % (tot_in / 2**30, written / 2**30, 100.0 * written / tot_in, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
