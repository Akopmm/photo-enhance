"""Measure the file the viewer downloads. Nothing else counts.

Every denoise finding in this folder that later turned out to be wrong was
wrong the same way: it measured a component instead of the product. The
engines were compared on the colour-corrected baseline and scored 0.93 against
0.78 -- near identical, and reported as such. Through the actual download path
the same two engines score 3.83 against 1.43. Both numbers were real. Only one
of them was about the thing anyone would ever look at.

So this harness renders through `render_full_style`, the function the download
button calls, with the parameters the editor sends, and measures the JPEG that
comes out. If a change does not move these numbers, it did not happen.

Noise is reported per brightness band because that is where it is visible and
where a single figure hides it: a dark indoor frame can measure 0.93 overall
and still carry 11.33 in its shadows, which is exactly the photo that prompted
this file.

    PHOTO=/path/to/shot.cr3 ./service/.venv/bin/python \\
        research/denoise-speed/validate.py

Run it inside the service container, or anywhere the service package imports.
"""
import asyncio
import hashlib
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "service"))

import denoise as dn  # noqa: E402
import pipeline  # noqa: E402
import settings  # noqa: E402
import storage  # noqa: E402

BANDS = [(0.00, 0.15, "deep shadow"), (0.15, 0.35, "shadow"),
         (0.35, 0.60, "midtone"), (0.60, 1.01, "highlight")]


def banded_sigma(arr: np.ndarray) -> dict:
    """Noise per brightness band, worst channel, flattest blocks in each.

    Per band because an overall figure averages the shadows -- where noise
    lives and where the eye finds it -- into the smooth bright areas that
    dominate the block count.
    """
    bs = 32
    h, w = arr.shape[:2]
    out = {}
    for lo, hi, label in BANDS:
        worst = float("nan")
        for c in range(3):
            g = (np.clip(arr[..., c], 0, 1) * 255).astype(np.float32)
            blocks = (g[:h // bs * bs, :w // bs * bs]
                      .reshape(h // bs, bs, w // bs, bs).swapaxes(1, 2).reshape(-1, bs * bs))
            lum = blocks.mean(1) / 255.0
            sel = blocks[(lum >= lo) & (lum < hi)]
            if len(sel) < 10:
                continue
            flat = sel[sel.var(1) <= np.percentile(sel.var(1), 30)]
            s = float(np.median(np.abs(flat - flat.mean(1, keepdims=True))) * 1.4826)
            worst = s if np.isnan(worst) else max(worst, s)
        out[label] = worst
    return out


def detail(arr: np.ndarray) -> float:
    """High-frequency energy in the BUSIEST blocks, as a proxy for detail kept.

    The other half of the question, and the half that keeps getting skipped.
    Noise and detail are both high-frequency, so any measure of "less noise"
    is equally a measure of "less detail" unless the two are looked at
    separately: noise in the flattest blocks, detail in the busiest. A change
    that improves one and quietly destroys the other reads as a win on a
    single number.
    """
    bs = 32
    h, w = arr.shape[:2]
    g = (np.clip(arr, 0, 1).mean(2) * 255).astype(np.float32)
    blocks = (g[:h // bs * bs, :w // bs * bs]
              .reshape(h // bs, bs, w // bs, bs).swapaxes(1, 2))
    var = blocks.reshape(-1, bs * bs).var(1)
    busy = blocks.reshape(-1, bs, bs)[var >= np.percentile(var, 90)]
    if not len(busy):
        return 0.0
    # Mean absolute Laplacian: edge energy, which smoothing removes.
    lap = (busy[:, 1:-1, 1:-1] * 4 - busy[:, :-2, 1:-1] - busy[:, 2:, 1:-1]
           - busy[:, 1:-1, :-2] - busy[:, 1:-1, 2:])
    return float(np.abs(lap).mean())


def load(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def render(import_id, style, method, amount=1.0, size="medium"):
    """One download, exactly as the button makes it."""
    settings.update({"denoise_method": method})
    dn._cache.pop("ffdnet_ov", None)
    t0 = time.time()
    path = asyncio.run(pipeline.render_full_style(
        import_id, style, strength=1.0, denoise_amount=amount, size=size))
    return path, time.time() - t0


def main():
    photo = os.environ.get("PHOTO")
    if not photo:
        sys.exit("set PHOTO to a RAW or JPEG file")

    data = open(photo, "rb").read()
    settings.update({"denoise_method": "balanced"})
    import_id = asyncio.run(pipeline.process_preview(
        data, os.path.basename(photo), "upload", "akop"))
    meta = storage.get_import(import_id)
    style = meta["styles"][0]["key"]
    print(f"imported {os.path.basename(photo)}  denoise={meta.get('denoise')}")
    print(f"style={style}\n")

    rows = []
    for method, amount in [("quality", 1.0), ("balanced", 1.0), ("fast", 1.0),
                           ("balanced", 0.0)]:
        path, secs = render(import_id, style, method, amount)
        arr = load(path)
        rows.append((f"{method} @{amount:.0%}", secs, os.path.getsize(path),
                     hashlib.sha256(open(path, "rb").read()).hexdigest()[:8],
                     banded_sigma(arr), detail(arr)))
        print(f"  rendered {method} @{amount:.0%}  {secs:6.1f}s", flush=True)

    print(f"\n{'variant':18s} {'time':>7s} {'KB':>6s} {'sha':>9s}", end="")
    for _, _, label in BANDS:
        print(f" {label:>12s}", end="")
    print(f" {'detail':>8s}")
    print("-" * 106)
    for label, secs, size, sha, bands, det in rows:
        print(f"{label:18s} {secs:6.1f}s {size // 1024:6d} {sha:>9s}", end="")
        for _, _, b in BANDS:
            print(f" {bands[b]:12.2f}", end="")
        print(f" {det:8.2f}")
    print("\n  detail: higher is more edge energy kept. A denoise that beats the "
          "reference\n  on noise while scoring well below it here is smearing, "
          "not denoising.")

    shas = [r[3] for r in rows]
    if len(set(shas)) != len(shas):
        print("\n  !! two renders produced the same file -- the cache key is not "
              "distinguishing them, and any comparison above is worthless")


if __name__ == "__main__":
    main()
