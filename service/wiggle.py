"""Depth-parallax animation ("3D GIF") from a stored import.

Costs almost nothing, because it needs no model: the depth map is already
persisted at import time for the Depth Pop / Aerial Depth looks, and the
baseline is already stored for the editor hero. This just re-projects one
against the other.

Each pixel is lifted into 3D using depth as Z, the virtual camera is moved,
and the scene is re-projected. Two modes:

  wiggle -- the camera slides sideways. Simple, robust, reads as parallax.
  turn   -- the camera yaws about a vertical axis through the SUBJECT, so
            the near cheek sweeps further than the far one. Reads as the
            subject turning rather than the picture sliding, but it opens
            wider gaps, so the angle has to stay small.

Forward-warped with painter's ordering (far written first, near overwrites)
rather than inverse-sampled: inverse sampling smears the foreground across
depth discontinuities, which on a face is glaring. Whatever is revealed from
behind an object has no source pixel at all, so those gaps are filled from
the nearest valid pixel along the row -- honest, and invisible at small
amplitudes, but it is why the amplitude is deliberately modest.
"""
import io
import logging

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger("photo-enhance.wiggle")

MAX_EDGE = 720          # GIFs get large fast; this is plenty for a share
FRAMES = 28
FRAME_MS = 55
COLORS = 160

# Thin objects very close to the lens are the worst case: rotating opens a gap
# behind them with nothing to fill from. These amplitudes were tuned by eye
# until that stopped showing.
WIGGLE_AMPLITUDE = 0.020    # fraction of width
TURN_DEGREES = 3.2


def _prepare(base: np.ndarray, depth: np.ndarray):
    h, w = base.shape[:2]
    d = np.asarray(Image.fromarray((np.clip(depth, 0, 1) * 255).astype(np.uint8))
                   .resize((w, h), Image.BILINEAR)
                   .filter(ImageFilter.GaussianBlur(2.0))).astype(np.float32) / 255.0
    lo, hi = float(d.min()), float(d.max())
    d = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
    return d


def _fill_rows(out: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Fill gaps from the nearest written pixel on the same row."""
    h, w = seen.shape
    cols = np.arange(w)[None, :]
    fwd = np.maximum.accumulate(np.where(seen, cols, -1), axis=1)
    rev = np.where(seen, cols, w)
    bwd = np.flip(np.minimum.accumulate(np.flip(rev, axis=1), axis=1), axis=1)
    pick = np.clip(np.where(fwd >= 0, fwd, bwd), 0, w - 1)
    return np.where(seen[..., None], out, out[np.arange(h)[:, None], pick])


def _scatter(cols, py, px, shape):
    h, w = shape
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    out = np.zeros((h, w, 3), np.float32)
    seen = np.zeros((h, w), bool)
    out[py[ok], px[ok]] = cols[ok]
    seen[py[ok], px[ok]] = True
    return _fill_rows(out, seen)


def render(base: np.ndarray, depth: np.ndarray, mode: str = "wiggle",
           subject: np.ndarray | None = None, progress=None) -> bytes:
    """base/(0..1) float32 (H,W,3), depth (H,W) -> animated GIF bytes."""
    img = Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8))
    img.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
    rgb = np.asarray(img).astype(np.float32)
    h, w = rgb.shape[:2]
    d = _prepare(rgb / 255.0, depth)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    order = np.argsort(d.ravel())            # far first, near overwrites
    oy, ox, od = yy.ravel()[order], xx.ravel()[order], d.ravel()[order]
    cols = rgb.reshape(-1, 3)[order]

    # Rotate about the subject when we know where it is, so a "turn" looks
    # like the subject turning rather than the room swinging past.
    cx, cy, pivot_d = w / 2.0, h / 2.0, float(np.percentile(d, 65))
    if subject is not None and (subject > 0.5).any():
        sm = np.asarray(Image.fromarray((subject * 255).astype(np.uint8))
                        .resize((w, h), Image.BILINEAR)) > 127
        sy, sx = np.where(sm)
        cx = float(sx.mean())
        upper = sy < np.percentile(sy, 35)
        cy = float(sy[upper].mean()) if upper.any() else float(sy.mean())
        pivot_d = float(np.percentile(d[sm], 60))

    f = w * 1.6
    zn, zf = 1.0, 2.2
    Z = zn + (1.0 - od) * (zf - zn)
    Zp = zn + (1.0 - pivot_d) * (zf - zn)
    X = (ox - cx) * Z / f
    Y = (oy - cy) * Z / f

    frames = []
    for i in range(FRAMES):
        t = 2 * np.pi * i / FRAMES
        if mode == "turn":
            th = np.deg2rad(TURN_DEGREES) * np.sin(t)
            c, s = np.cos(th), np.sin(th)
            zc = Z - Zp
            zr = np.maximum(-X * s + zc * c + Zp, 0.05)
            px = np.rint((X * c + zc * s) * f / zr + cx).astype(np.int32)
            py = np.rint(Y * f / zr + cy).astype(np.int32)
        else:
            shift = w * WIGGLE_AMPLITUDE * np.sin(t)
            px = np.rint(ox - shift * (od - pivot_d)).astype(np.int32)
            py = oy.astype(np.int32)
        frame = _scatter(cols, py, px, (h, w))
        trim = int(w * 0.045)
        frame = np.clip(frame, 0, 255).astype(np.uint8)[trim:h - trim, trim:w - trim]
        frames.append(Image.fromarray(frame).convert(
            "P", palette=Image.ADAPTIVE, colors=COLORS))
        if progress:
            progress(i + 1, FRAMES)

    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=FRAME_MS, loop=0, optimize=True, disposal=2)
    return buf.getvalue()
