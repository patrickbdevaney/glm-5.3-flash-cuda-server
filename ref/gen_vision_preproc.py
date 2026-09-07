"""Oracle for image preprocessing: transformers' own Glm5NextImageProcessor.

Writes a deterministic test PNG and the processor's output for it, so the C++ path can be gated on
the same bytes. Dumps BOTH the processor's resized/normalized pixel tensor and its final
flatten_patches: patchify is exact arithmetic and must match to the bit, whereas an antialiased
bicubic resize is implementation-defined and only has to match closely. Separating them means a
failure says which one moved.
"""
import os, json, numpy as np, torch
from PIL import Image
from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision")
os.makedirs(OUT, exist_ok=True)

# A deterministic image with real structure (gradients + edges), not noise: a resize bug on flat
# noise looks like a resize bug on anything, but on edges it is visible in the numbers.
import sys
CASE = sys.argv[1] if len(sys.argv) > 1 else "small"
# "small" takes the PAD path (content fits, zero-padded to the aligned canvas) -- exact arithmetic,
# gateable to the bit. "big" exceeds the token budget and forces the ANTIALIASED BICUBIC downscale,
# which is implementation-defined and can only be gated closely.
H, W = (233, 311) if CASE == "small" else (1701, 2203)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
img = np.zeros((H, W, 3), np.float32)
img[..., 0] = (xx / W) * 255
img[..., 1] = (yy / H) * 255
img[..., 2] = ((((xx // 17) + (yy // 13)) % 2) * 200 + 20)
SUF = "" if CASE == "small" else "_big"
u8 = np.clip(img, 0, 255).astype(np.uint8)
Image.fromarray(u8).save(os.path.join(OUT, "test_image%s.png" % SUF))
np.array([H, W], np.int32).tofile(os.path.join(OUT, "test_image_hw%s.bin" % SUF))
u8.tofile(os.path.join(OUT, "test_image_rgb%s.bin" % SUF))          # HWC uint8, what stb_image returns

proc = Glm5NextImageProcessor()
out = proc(images=[Image.fromarray(u8)], return_tensors="pt")
pv, grid = out["pixel_values"], out["image_grid_thw"]
print("pixel_values", tuple(pv.shape), "grid_thw", grid.tolist())
pv.to(torch.float32).numpy().tofile(os.path.join(OUT, "preproc_patches%s.bin" % SUF))
grid.to(torch.int32).numpy().tofile(os.path.join(OUT, "preproc_grid%s.bin" % SUF))

# The resized+normalized image, before patchify, so patchify can be gated on its own.
import transformers.models.glm5_next.image_processing_glm5_next as M
t = torch.from_numpy(u8).permute(2, 0, 1).unsqueeze(0).float()
r = proc.resize(t, resample=M.PILImageResampling.BICUBIC, factor=proc.patch_size * proc.merge_size,
                temporal_factor=proc.temporal_patch_size,
                min_image_tokens=proc.min_image_tokens if hasattr(proc, "min_image_tokens") else 16,
                max_image_tokens=proc.max_image_tokens if hasattr(proc, "max_image_tokens") else 8000)
r = r * proc.rescale_factor
mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1); std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
r = (r - mean) / std
print("resized+normalized", tuple(r.shape))
np.array(list(r.shape[-2:]), np.int32).tofile(os.path.join(OUT, "preproc_chw_hw%s.bin" % SUF))
r.to(torch.float32).numpy().tofile(os.path.join(OUT, "preproc_chw%s.bin" % SUF))
print("mean/std", proc.image_mean, proc.image_std)
print("wrote", OUT)
