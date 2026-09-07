#!/usr/bin/env python3
"""Compare two perplexity dumps token-for-token.

Perplexity alone can hide a behaviour change: a model can hold its mean NLL and still pick a
different token. So this also reports TOP-1 AGREEMENT, which is what actually shows up in
generated text, and the distribution of per-token NLL differences rather than only the mean.
"""
import sys, struct, numpy as np

def load(p):
    with open(p, 'rb') as f:
        n = struct.unpack('q', f.read(8))[0]
        nll = np.frombuffer(f.read(4 * n), dtype=np.float32)
        am = np.frombuffer(f.read(4 * n), dtype=np.int32)
    return nll.astype(np.float64), am

a_p, b_p = sys.argv[1], sys.argv[2]
a_name = sys.argv[3] if len(sys.argv) > 3 else a_p
b_name = sys.argv[4] if len(sys.argv) > 4 else b_p
a, aa = load(a_p); b, ba = load(b_p)
n = min(len(a), len(b))
if len(a) != len(b):
    print(f"WARNING: token counts differ ({len(a)} vs {len(b)}); comparing first {n}")
a, aa, b, ba = a[:n], aa[:n], b[:n], ba[:n]

pa, pb = np.exp(a.mean()), np.exp(b.mean())
print(f"tokens            {n}")
print(f"{b_name:<18}PPL {pb:.6f}   mean NLL {b.mean():.8f}   (reference)")
print(f"{a_name:<18}PPL {pa:.6f}   mean NLL {a.mean():.8f}")
print(f"delta             PPL {pa-pb:+.6f}  ({100*(pa-pb)/pb:+.4f}%)   NLL {a.mean()-b.mean():+.8f}")
d = a - b
print(f"\nper-token NLL delta ({a_name} - {b_name}):")
for q in (1, 5, 25, 50, 75, 95, 99):
    print(f"   p{q:<3} {np.percentile(d, q):+.6f}")
print(f"   mean {d.mean():+.6f}   std {d.std():.6f}   |d|>0.1: {100*(np.abs(d)>0.1).mean():.2f}%")
agree = (aa == ba).mean()
print(f"\ntop-1 agreement   {100*agree:.3f}%   ({int((aa!=ba).sum())} of {n} tokens differ)")
# Where they disagree, is the reference confident? A disagreement on a low-NLL (confident) token
# matters more than one where the model was already unsure.
dis = aa != ba
if dis.sum():
    print(f"   on disagreements: reference NLL median {np.median(b[dis]):.4f}"
          f"  vs agreements {np.median(b[~dis]):.4f}")
    print(f"   confident disagreements (ref NLL < 0.5): {int((b[dis] < 0.5).sum())}"
          f" = {100*(b[dis]<0.5).sum()/n:.3f}% of all tokens")
