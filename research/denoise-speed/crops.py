"""Side-by-side 100% crops, because PSNR cannot answer "does it still look good".

Picks the noisiest patches in the frame automatically -- flat, dark areas are
where denoising is visible and where a cheap method gives itself away, and
picking them by hand invites picking the ones that flatter the result.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "..", "service"))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench  # noqa: E402

CROP = 420          # pixels of original, shown 1:1
PATCHES = 3


def noisiest_patches(a, n=PATCHES, size=CROP):
    """Flat regions with the most high-frequency energy: that is noise."""
    g = a.mean(2)
    t = torch.from_numpy(g)[None, None]
    lo = F.avg_pool2d(t, 8, 8)
    hi = t - F.interpolate(lo, size=g.shape, mode="bilinear", align_corners=False)
    energy = F.avg_pool2d(hi.abs(), 64, 64)[0, 0].numpy()
    # Ignore blown highlights and pure black, where nothing is visible anyway.
    lum = F.avg_pool2d(t, 64, 64)[0, 0].numpy()
    energy[(lum > 0.92) | (lum < 0.03)] = 0

    picks, taken = [], np.zeros_like(energy, bool)
    for _ in range(n):
        idx = np.argmax(np.where(taken, -1, energy))
        y, x = np.unravel_index(idx, energy.shape)
        taken[max(0, y - 3):y + 4, max(0, x - 3):x + 4] = True
        py = min(max(0, y * 64 - size // 2), a.shape[0] - size)
        px = min(max(0, x * 64 - size // 2), a.shape[1] - size)
        picks.append((py, px))
    return picks


def main():
    raw = bench.RAW
    print("decoding...", flush=True)
    full = bench.decode(raw)
    half = bench.decode(raw, half=True)

    cache = os.path.join(HERE, "ref.npy")
    if not os.path.exists(cache):
        sys.exit("run bench.py first so the reference is cached")
    ref = np.load(cache)

    def up(a):
        """Bring a half-size result back to full size, for a fair 1:1 crop."""
        if a.shape[:2] == full.shape[:2]:
            return a
        t = bench.to_t(a)
        return bench.to_a(F.interpolate(t, size=full.shape[:2], mode="bilinear",
                                        align_corners=False))

    print("building variants...", flush=True)
    variants = [("original (no denoise)", full), ("SCUNet full res  700s", ref)]
    for label, fn in [
        ("half decode + SCUNet  15s", lambda: up(bench.denoise_full(half))),
        ("residual 1/2  16s", lambda: bench.denoise_residual(full, 2)),
        ("guided chroma  1.2s", lambda: bench.guided_chroma(full)),
    ]:
        t0 = time.time()
        variants.append((label, fn()))
        print(f"  {label} {time.time()-t0:.2f}s", flush=True)

    picks = noisiest_patches(full)
    pad, header = 8, 26
    cols, rows = len(variants), len(picks)
    W = cols * CROP + (cols + 1) * pad
    H = rows * (CROP + header) + (rows + 1) * pad
    sheet = Image.new("RGB", (W, H), (16, 18, 21))
    d = ImageDraw.Draw(sheet)

    for r, (py, px) in enumerate(picks):
        for c, (label, img) in enumerate(variants):
            patch = np.clip(img[py:py + CROP, px:px + CROP], 0, 1)
            x = pad + c * (CROP + pad)
            y = pad + r * (CROP + header + pad)
            d.text((x, y + 6), f"{label}", fill=(200, 205, 212))
            sheet.paste(Image.fromarray((patch * 255).astype(np.uint8)),
                        (x, y + header))

    out = os.path.join(HERE, "compare_100pct.png")
    sheet.save(out)
    print("wrote", out, sheet.size)

    # Second sheet: everything downscaled to the delivered size first, then
    # cropped. This is what the viewer actually receives.
    scale = bench.OUTPUT_EDGE / max(full.shape[:2])
    small_crop = int(CROP * scale)
    W2 = cols * small_crop + (cols + 1) * pad
    H2 = rows * (small_crop + header) + (rows + 1) * pad
    sheet2 = Image.new("RGB", (W2, H2), (16, 18, 21))
    d2 = ImageDraw.Draw(sheet2)
    shrunk = [(lab, bench.resize(img, bench.OUTPUT_EDGE)) for lab, img in variants]
    for r, (py, px) in enumerate(picks):
        sy, sx = int(py * scale), int(px * scale)
        for c, (label, img) in enumerate(shrunk):
            patch = np.clip(img[sy:sy + small_crop, sx:sx + small_crop], 0, 1)
            x = pad + c * (small_crop + pad)
            y = pad + r * (small_crop + header + pad)
            d2.text((x, y + 6), label, fill=(200, 205, 212))
            sheet2.paste(Image.fromarray((patch * 255).astype(np.uint8)), (x, y + header))
    out2 = os.path.join(HERE, "compare_delivered.png")
    sheet2.save(out2)
    print("wrote", out2, sheet2.size, f"(at {bench.OUTPUT_EDGE}px delivery)")


if __name__ == "__main__":
    main()
