"""The OpenVINO BiRefNet must produce the same mask as the PyTorch one.

Runs inside the image build, right after the IR is created, because that is
the only place both builds exist side by side and it costs no extra
conversion.

The criterion is deliberately the STORED mask, not the raw float. Masks are
persisted as 8-bit PNG (see masking.encode), so the finest difference that
can survive is 1/255 = 0.00392. Comparing raw floats against a tighter
threshold measures nothing useful: a 3.9e-03 float difference and a 0-level
difference in the file are the same thing. What matters downstream is the
byte that gets written.
"""
import io
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_LEVELS = 1          # one 8-bit step at the worst pixel
MAX_FRACTION = 0.001    # and at most 0.1% of pixels may differ even by that


def main() -> int:
    import masking

    if masking._device() != "cpu":
        print(f"  torch is on {masking._device()}; the OpenVINO path is CPU-only, skipping")
        return 0

    comp = masking._birefnet_ov()
    if comp is None:
        print(f"  FAIL: no OpenVINO build loaded from {masking.BIREFNET_IR}")
        return 1

    # A deterministic, structured image -- flat noise gives the segmenter
    # nothing to find, so the masks would agree trivially.
    rng = np.random.default_rng(7)
    h, w = 1024, 683
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 0.25 + 0.5 * (xx / w)
    img[..., 1] = 0.30 + 0.4 * (yy / h)
    img[..., 2] = 0.55
    blob = ((yy - h * 0.55) ** 2 / (h * 0.28) ** 2 + (xx - w * 0.5) ** 2 / (w * 0.3) ** 2) < 1.0
    img[blob] = (0.88, 0.80, 0.72)
    img = np.clip(img + rng.normal(0, 0.01, img.shape), 0, 1)
    pil = Image.fromarray((img * 255).astype(np.uint8), "RGB")

    ov_mask = masking.subject_mask(pil)
    masking._cache["birefnet_ov"] = None          # force the torch fallback
    try:
        pt_mask = masking.subject_mask(pil)
    finally:
        masking._cache.pop("birefnet_ov", None)

    if ov_mask.shape != pt_mask.shape:
        print(f"  FAIL: shapes differ, {ov_mask.shape} vs {pt_mask.shape}")
        return 1

    a = np.asarray(Image.open(io.BytesIO(masking.encode(ov_mask))).convert("L"), dtype=np.int16)
    b = np.asarray(Image.open(io.BytesIO(masking.encode(pt_mask))).convert("L"), dtype=np.int16)
    d = np.abs(a - b)
    worst = int(d.max())
    frac = float((d > 0).mean())
    inter = float(((ov_mask > 0.5) & (pt_mask > 0.5)).sum())
    union = float(((ov_mask > 0.5) | (pt_mask > 0.5)).sum())
    iou = inter / union if union else 1.0

    print(f"  stored mask: worst pixel differs by {worst} level(s) of 255, "
          f"{frac*100:.4f}% of pixels differ at all, IoU {iou:.7f}")
    if worst > MAX_LEVELS or frac > MAX_FRACTION:
        print(f"  FAIL: allowed {MAX_LEVELS} level on at most {MAX_FRACTION*100:.1f}% of pixels")
        return 1
    print("  OK: OpenVINO and PyTorch masks are equivalent at storage precision")
    return 0


if __name__ == "__main__":
    sys.exit(main())
