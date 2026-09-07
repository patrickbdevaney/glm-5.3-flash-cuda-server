"""Oracle for the vision tower: transformers' own Glm5NextVisionModel on the real weights.

WHY THIS ONE IS LOADABLE. The full model is not instantiable on this box (documented in
nvfp4_trace_capture.py: the NVFP4 experts have no model-level loader here), which is why every
other oracle in ref/ streams a layer at a time. The vision tower is the exception -- ~0.5B params
in plain bf16, about 1.2 GiB -- so it loads whole and the gate compares against transformers
proper rather than against a reimplementation of it (CLAUDE.md §2).

Dumps staged tensors so a failure localises: patch embed, block 0, all blocks + post_layernorm,
downsample, and the merger output that is what actually reaches the language model.
"""
import json, os, sys, glob
import numpy as np, torch
from safetensors import safe_open
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

CKPT = os.environ.get("GLM5_CKPT", os.path.expanduser(
    "~/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "vision_fp32" if os.environ.get("GLM5_VIS_DTYPE") == "fp32" else "vision")
os.makedirs(OUT, exist_ok=True)

def dump(name, t):
    a = t.detach().to(torch.float32).cpu().numpy()
    a.tofile(os.path.join(OUT, name + ".bin"))
    print(f"  {name:28s} {str(tuple(a.shape)):22s} {a.dtype}")
    return a

cfg_all = json.load(open(os.path.join(CKPT, "config.json")))
vcfg = Glm5NextVisionConfig(**cfg_all["vision_config"])
print("vision config: depth", vcfg.depth, "hidden", vcfg.hidden_size,
      "heads", vcfg.num_heads, "patch", vcfg.patch_size, "merge", vcfg.spatial_merge_size)

torch.manual_seed(0)
# GLM5_VIS_DTYPE=fp32 upcasts the tower. The engine keeps fp32 activations with bf16 weights, so
# a bf16 oracle is the LESS precise of the two and its drift compounds over 24 blocks; comparing
# against both says whether a mismatch is the kernel or the reference.
_DT = torch.float32 if os.environ.get("GLM5_VIS_DTYPE") == "fp32" else torch.bfloat16
model = Glm5NextVisionModel._from_config(vcfg).to(_DT).eval()

# gather model.visual.* out of the shards
idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
want = {k: v for k, v in idx.items() if k.startswith("model.visual.")}
byfile = {}
for k, f in want.items(): byfile.setdefault(f, []).append(k)
sd = {}
for f, keys in byfile.items():
    with safe_open(os.path.join(CKPT, f), framework="pt") as fh:
        for k in keys: sd[k[len("model.visual."):]] = fh.get_tensor(k).to(_DT)
missing, unexpected = model.load_state_dict(sd, strict=False)
missing = [m for m in missing if "rotary" not in m and "inv_freq" not in m]
print(f"loaded {len(sd)} vision tensors; missing {missing}; unexpected {unexpected[:4]}")
if missing or unexpected:
    print("REFUSING: the oracle must run on the real weights, complete", file=sys.stderr); sys.exit(2)

# One 448x448 image: 32x32 patches of 14, temporal_patch_size 2 -> input row is 3*2*14*14 = 1176.
GH = GW = vcfg.image_size // vcfg.patch_size
grid_thw = torch.tensor([[1, GH, GW]], dtype=torch.long)
n_patch = GH * GW
in_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size * vcfg.patch_size
g = torch.Generator().manual_seed(1234)
x = (torch.rand(n_patch, in_dim, generator=g) * 2 - 1).to(_DT)
print(f"input: {n_patch} patches x {in_dim}  (grid {GH}x{GW})")

np.array([1, GH, GW], dtype=np.int32).tofile(os.path.join(OUT, "grid_thw.bin"))
dump("input", x)

# Dump cos/sin so the gate can test the TOWER independently of position-id generation. Those are
# two separate things to get wrong and a single number cannot say which one did.
from transformers.vision_utils import get_vision_position_ids
pos = get_vision_position_ids(grid_thw, vcfg.spatial_merge_size)
rot = model.rotary_pos_emb(pos)
emb = torch.cat((rot, rot), dim=-1)
dump("pos_ids", pos.to(torch.float32))
dump("cos", emb.cos()); dump("sin", emb.sin())

acts = {}
h = model.patch_embed(x); acts["patch_embed"] = h
hooks = []
def mk(i):
    def fn(_m, _i, o): acts[f"block{i}"] = o[0] if isinstance(o, tuple) else o
    return fn
for i in (0, 1, vcfg.depth - 1): hooks.append(model.blocks[i].register_forward_hook(mk(i)))
pl = {}
hooks.append(model.post_layernorm.register_forward_hook(
    lambda _m, _i, o: pl.__setitem__("post_layernorm", o)))

with torch.no_grad():
    out = model(hidden_states=x, grid_thw=grid_thw)
for h_ in hooks: h_.remove()

dump("patch_embed", acts["patch_embed"])
dump("block0", acts["block0"]); dump("block1", acts["block1"])
dump(f"block{vcfg.depth-1}", acts[f"block{vcfg.depth-1}"])
dump("post_layernorm", pl["post_layernorm"])
dump("downsampled", out.last_hidden_state)     # [n_merged, out_hidden_size]
dump("merged", out.pooler_output)              # what reaches the language model
print("wrote", OUT)
