"""The OpenVINO BiRefNet must produce the same mask as the PyTorch one.

Runs inside the image build, right after the IR is created, because that is
the only place both builds exist side by side and it costs no extra
conversion.

What "the same" has to mean here
--------------------------------
Not bitwise. The two backends reduce floating point in different orders, and
that order depends on the CPU the build happens to run on -- thread count,
AVX2 vs AVX-512, kernel selection. Measured across build machines, the worst
single pixel moved between 1 and 7 levels of 255 for identical inputs and
identical weights. A bound tight enough to reject 7 levels rejects healthy
builds at random, which is worse than no test.

So the criterion is structural. A genuinely broken conversion -- wrong output
port, a transposed tensor, an untrained head -- does not miss by 3% on a
fringe of edge pixels; it produces a different picture, and IoU collapses.
The bounds below are set to pass CPU noise and fail that, and the script
proves it by also scoring a deliberately corrupted mask and requiring the
criteria to reject it. If that control ever passes, the test has gone slack
and says so.

Masks are persisted as 8-bit PNG (see masking.encode), so everything is
measured on the stored bytes, which is what downstream code actually reads.
"""
import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MIN_IOU = 0.999          # thresholded masks must cover the same region
MAX_MEAN_LEVELS = 1.0    # average difference under one 8-bit step
MAX_WORST_LEVELS = 16    # ~6% alpha at the single worst edge pixel
MAX_FRAC_OVER_2 = 0.01   # at most 1% of pixels off by more than 2 levels


def _as_levels(mask, encode):
    return np.asarray(Image.open(io.BytesIO(encode(mask))).convert("L"), dtype=np.int16)


def score(a_mask, b_mask, encode):
    """Comparison stats between two masks, on the bytes that get stored."""
    a, b = _as_levels(a_mask, encode), _as_levels(b_mask, encode)
    d = np.abs(a - b)
    inter = float(((a_mask > 0.5) & (b_mask > 0.5)).sum())
    union = float(((a_mask > 0.5) | (b_mask > 0.5)).sum())
    return {
        "iou": inter / union if union else 1.0,
        "mean": float(d.mean()),
        "worst": int(d.max()),
        "frac_over_2": float((d > 2).mean()),
    }


def passes(s):
    return (s["iou"] >= MIN_IOU and s["mean"] <= MAX_MEAN_LEVELS
            and s["worst"] <= MAX_WORST_LEVELS and s["frac_over_2"] <= MAX_FRAC_OVER_2)


def describe(label, s):
    print(f"  {label:<22} IoU {s['iou']:.6f}  mean {s['mean']:5.3f} lv  "
          f"worst {s['worst']:3d} lv  >2lv on {s['frac_over_2']*100:6.3f}% of pixels")


def main() -> int:
    import masking

    if masking._device() != "cpu":
        print(f"  torch is on {masking._device()}; the OpenVINO path is CPU-only, skipping")
        return 0

    comp = masking._birefnet_ov()
    if comp is None:
        print(f"  FAIL: no OpenVINO build loaded from {masking.BIREFNET_IR}")
        return 1

    # Deterministic and structured -- flat noise gives the segmenter nothing
    # to find, so the two builds would agree trivially.
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

    real = score(ov_mask, pt_mask, masking.encode)
    describe("OpenVINO vs torch", real)

    # Control: the same mask nudged 8px sideways. Visually almost the same
    # picture, and the criteria must still reject it -- otherwise they are
    # loose enough to wave through a real regression.
    shifted = np.roll(pt_mask, 8, axis=1)
    control = score(ov_mask, shifted, masking.encode)
    describe("control (8px shift)", control)

    ok = passes(real)
    if passes(control):
        print("  FAIL: the control passed too -- these bounds no longer test anything")
        return 1
    if not ok:
        print(f"  FAIL: need IoU >= {MIN_IOU}, mean <= {MAX_MEAN_LEVELS} lv, "
              f"worst <= {MAX_WORST_LEVELS} lv, >2lv on <= {MAX_FRAC_OVER_2*100:.1f}% of pixels")
        return 1
    print("  OK: OpenVINO and PyTorch masks agree, and the control was rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
