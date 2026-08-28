"""Can the render reach seconds without giving up the denoise quality?

Denoise is 99.7% of a full-resolution render (measured: 897s of 900s on the
optiplex iGPU). Everything here attacks that one number.

The comparison that matters is at the size the photo is actually delivered at,
not at sensor resolution: downscaling is itself a denoiser, so work spent
removing noise from pixels that are about to be averaged away is work wasted.
That is why "no denoise at all" is one of the variants -- if it scores well at
the output size, the whole stage is worth less than it looks.

Reference = what the service does today: decode full, denoise full, resize.
"""
import os
import sys
import time

import numpy as np
import rawpy
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "..", "service"))
import denoise as dn  # noqa: E402

# Checked in main(), not here: exiting at import time would make the script
# unimportable, and CI imports these to catch drift from the service API.
RAW = os.environ.get("PHOTO", "")


def _require_photo():
    if not RAW:
        sys.exit("set PHOTO to a RAW file, e.g.\n"
                 "  PHOTO=/path/to/shot.cr3 ./service/.venv/bin/python "
                 f"research/denoise-speed/{os.path.basename(__file__)}")
# 2400, not 3200: a half-size decode tops out at 3000px, and every variant
# has to be judged at a size all of them can actually deliver.
OUTPUT_EDGE = 2400


# ----------------------------------------------------------------- helpers

def decode(path, half=False):
    with rawpy.imread(path) as r:
        rgb = r.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8,
                            output_color=rawpy.ColorSpace.sRGB, half_size=half)
    return rgb.astype(np.float32) / 255.0


def to_t(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).unsqueeze(0)


def to_a(t):
    return t.squeeze(0).permute(1, 2, 0).numpy()


def resize(a, edge):
    t = to_t(a)
    h, w = a.shape[:2]
    s = edge / max(h, w)
    if s >= 1:
        return a          # never upscale; a variant that cannot reach the
                          # target size is reported at its own size instead
    out = F.interpolate(t, size=(max(1, round(h * s)), max(1, round(w * s))),
                        mode="area")
    return to_a(out)


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else 10 * np.log10(1.0 / mse)


def box(a, r):
    """Mean over a (2r+1) window, via summed-area table. O(1) per pixel."""
    pad = np.pad(a, ((r + 1, r), (r + 1, r)) + ((0, 0),) * (a.ndim - 2), mode="edge")
    s = pad.cumsum(0).cumsum(1)
    k = 2 * r + 1
    out = (s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k])
    return out / (k * k)


# ----------------------------------------------------------------- variants

def denoise_full(a):
    return dn.denoise(a)


def denoise_residual(a, factor):
    """Denoise a shrunken copy, then apply only the CORRECTION at full size.

    Noise removal is a small, mostly low-frequency correction on smooth areas.
    Carrying the correction back up costs one resize instead of factor^2 tiles,
    and never touches the detail the full-resolution image already has right.
    """
    h, w = a.shape[:2]
    small = to_a(F.interpolate(to_t(a), scale_factor=1 / factor, mode="area"))
    cleaned = dn.denoise(small)
    residual = cleaned - small
    up = F.interpolate(to_t(residual), size=(h, w), mode="bilinear", align_corners=False)
    return np.clip(a + to_a(up), 0, 1)


def guided_chroma(a, radius=4, eps=1e-3, factor=4):
    """Denoise chroma only, with a guided filter, at reduced resolution.

    The visible ugliness in a high-ISO frame is chroma blotching; luma noise
    reads as grain and is often wanted. Chroma is also where the eye has least
    acuity, so it survives being filtered at quarter scale. No network, no
    weights, O(N) in the pixel count.
    """
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb, cr = b - y, r - y

    h, w = a.shape[:2]
    small = (max(1, h // factor), max(1, w // factor))

    def shrink(x):
        return to_a(F.interpolate(to_t(x[..., None]), size=small, mode="area"))[..., 0]

    def grow(x):
        return to_a(F.interpolate(to_t(x[..., None]), size=(h, w),
                                  mode="bilinear", align_corners=False))[..., 0]

    guide = shrink(y)
    mean_g = box(guide, radius)
    var_g = box(guide * guide, radius) - mean_g * mean_g

    def filt(ch):
        c = shrink(ch)
        mean_c = box(c, radius)
        cov = box(guide * c, radius) - mean_g * mean_c
        A = cov / (var_g + eps)
        B = mean_c - A * mean_g
        return grow(box(A, radius)) * y + grow(box(B, radius))

    cb2, cr2 = filt(cb), filt(cr)
    out = np.empty_like(a)
    out[..., 0] = y + cr2
    out[..., 2] = y + cb2
    out[..., 1] = (y - 0.299 * out[..., 0] - 0.114 * out[..., 2]) / 0.587
    return np.clip(out, 0, 1)


# --------------------------------------------------------------------- run

def main():
    _require_photo()
    print(f"device: {dn._device()}   output edge: {OUTPUT_EDGE}px\n")

    t0 = time.time()
    full = decode(RAW)
    t_dec_full = time.time() - t0
    t0 = time.time()
    half = decode(RAW, half=True)
    t_dec_half = time.time() - t0
    print(f"decode full  {full.shape[1]}x{full.shape[0]}  {t_dec_full:5.2f}s")
    print(f"decode half  {half.shape[1]}x{half.shape[0]}  {t_dec_half:5.2f}s\n")

    import os
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref.npy")
    meta = cache + ".time"
    if os.path.exists(cache):
        ref_full = np.load(cache)
        t_ref = float(open(meta).read())
        print(f"reference loaded from cache ({t_ref:.1f}s when measured)", flush=True)
    else:
        print("building reference (full-res denoise, what ships today)...", flush=True)
        t0 = time.time()
        ref_full = denoise_full(full)
        t_ref = time.time() - t0
        np.save(cache, ref_full)
        open(meta, "w").write(str(t_ref))
    ref = resize(ref_full, OUTPUT_EDGE)
    print(f"  reference {t_ref:6.1f}s\n")

    rows = [("reference: decode full + denoise full", t_dec_full + t_ref, ref)]

    variants = [
        ("no denoise at all", lambda: (full, t_dec_full)),
        ("residual 1/2", lambda: (denoise_residual(full, 2), t_dec_full)),
        ("residual 1/4", lambda: (denoise_residual(full, 4), t_dec_full)),
        ("half decode + denoise full", lambda: (denoise_full(half), t_dec_half)),
        ("half decode + residual 1/2", lambda: (denoise_residual(half, 2), t_dec_half)),
        ("guided chroma only (no NN)", lambda: (guided_chroma(full), t_dec_full)),
        ("half decode + guided chroma", lambda: (guided_chroma(half), t_dec_half)),
    ]

    for label, fn in variants:
        t0 = time.time()
        img, t_decode = fn()
        t_work = time.time() - t0
        rows.append((label, t_decode + t_work, resize(img, OUTPUT_EDGE)))
        print(f"  {label:34s} {t_decode + t_work:7.2f}s", flush=True)

    # "vs untouched" says how much denoising a variant actually did. Without
    # it, doing nothing scores well on "vs reference" purely because the
    # reference does not change the image much at this size either.
    untouched = resize(full, OUTPUT_EDGE)
    print()
    print(f"{'variant':34s} {'total':>8s} {'speedup':>8s} {'vs ref':>11s} {'vs untouched':>13s}")
    print("-" * 80)
    base = rows[0][1]
    for label, secs, img in rows:
        vs_ref = "reference" if label.startswith("reference") else f"{psnr(img, ref):6.2f} dB"
        print(f"{label:34s} {secs:7.2f}s {base / secs:7.1f}x {vs_ref:>11s} "
              f"{psnr(img, untouched):9.2f} dB")
    print()
    print("vs untouched: LOWER = changed the photo more, i.e. denoised harder.")
    print(f"the reference itself is {psnr(ref, untouched):.2f} dB from untouched --")
    print(f"that gap is the entire value of this stage at {OUTPUT_EDGE}px.")


if __name__ == "__main__":
    main()
