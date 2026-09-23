"""Does NVFP4 damage MiMo's o_proj more than it damaged GLM's?

WHY THIS EXISTS. Xiaomi fused q/k/v and quantized it to F8_E4M3, and left o_proj at BF16 on
every layer (`ignored_layers`, 49 entries). A vendor that quantized 3 of 4 attention projections
and skipped the 4th is telling us something. Before spending the overlay on it we measure the
damage against a control whose sign we already know: GLM-5.3-Flash's o_proj, which we DID take
to NVFP4 and shipped (+1.56% ppl for the whole dense overlay).

Uses the PRODUCTION quantizer imported from tools/requant_dense_nvfp4.py -- not a
reimplementation -- so what is measured here is what the engine would actually load.
Also measures the FP8-e4m3 fallback, i.e. exactly what the vendor did to qkv, so the choice is
between two measured options rather than one measured and one assumed.
"""
import importlib.util, json, os, struct, sys
import numpy as np

spec = importlib.util.spec_from_file_location("rq", "tools/requant_dense_nvfp4.py")
rq = importlib.util.module_from_spec(spec)
sys.argv = ["rq"]                      # its main() is guarded by __name__, but be safe
spec.loader.exec_module(rq)

GROUP = rq.GROUP


def fp4_roundtrip(w):
    """Quantize then dequantize exactly as kernels/moe.cu will read it back."""
    amax = np.abs(w).max()
    gs = np.float32(448.0 * 6.0 / max(amax, 1e-30))          # global scale, production convention
    packed, code = rq.quant_block(w, gs)
    rows, K = w.shape
    eff = (rq.E4M3[code] / gs).astype(np.float32)            # [rows, K/16]
    lo = (packed & 0x0F).astype(np.uint8)
    hi = (packed >> 4).astype(np.uint8)
    idx = np.empty((rows, K), dtype=np.uint8)
    idx[:, 0::2], idx[:, 1::2] = lo, hi
    mag = rq.E2M1[idx & 7]
    sign = np.where((idx >> 3) & 1, -1.0, 1.0).astype(np.float32)
    return (mag * sign).reshape(rows, K // GROUP, GROUP) * eff[:, :, None]


def fp8_roundtrip(w):
    """Per-128x128-block e4m3, which is what the vendor applied to qkv_proj."""
    rows, K = w.shape
    out = np.empty_like(w)
    for i in range(0, rows, 128):
        for j in range(0, K, 128):
            b = w[i:i+128, j:j+128]
            s = np.abs(b).max() / rq.FP8_MAX
            s = max(s, 1e-30)
            q = np.clip(b / s, -rq.FP8_MAX, rq.FP8_MAX)
            code = np.searchsorted(rq.E4M3_MID, np.abs(q)).astype(np.uint8)
            out[i:i+128, j:j+128] = rq.E4M3[code] * np.sign(q) * s
    return out


def report(tag, w):
    w = w.astype(np.float32)
    for name, f in (("NVFP4", fp4_roundtrip), ("FP8-e4m3", fp8_roundtrip)):
        d = f(w).reshape(w.shape)
        err = np.linalg.norm(d - w) / np.linalg.norm(w)
        # Per-output-row cosine: a row IS one output channel of the projection.
        num = (d * w).sum(1)
        cos = num / (np.linalg.norm(d, axis=1) * np.linalg.norm(w, axis=1) + 1e-30)
        print(f"  {tag:28} {name:9}  rel_fro {err:.5f}   cos mean {cos.mean():.6f}  "
              f"min {cos.min():.6f}")


print("=== MiMo-V2.6-Flash o_proj (vendor left these BF16) ===")
for L in (0, 21, 47):
    raw = open(f"/tmp/claude-2002/-home-patrickd/3952824c-51a0-4b40-aa8e-acf3b3b02ab9/scratchpad/oproj_{L}.bin", "rb").read()
    w = rq.bf16_to_f32(raw).reshape(4096, 8192)
    report(f"MiMo layer {L}", w)

print("\n=== GLM-5.3-Flash o_proj (CONTROL: we shipped these as NVFP4) ===")
M = os.path.expanduser('~/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2')
found = json.load(open('/tmp/glm_oproj.json'))
for sh, base, name, t in [found[0], found[len(found)//2], found[-1]]:
    a, b = t['data_offsets']
    with open(os.path.join(M, sh), 'rb') as f:
        f.seek(base + a); raw = f.read(b - a)
    w = rq.bf16_to_f32(raw).reshape(*t['shape'])
    report(name.split('layers.')[1].split('.')[0].join(["GLM layer ", ""]), w)
