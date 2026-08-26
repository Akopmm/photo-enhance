"""Denoising with SCUNet (Apache-2.0), on the model's corrected baseline.

Why here, and why this model
----------------------------
High-ISO frames come out of the colour model with the noise amplified --
brightening a dark exposure multiplies the sensor noise along with the
signal, and `dehaze`/`contrast` in the looks then sharpen it. Measured on a
real indoor CR3: sigma 4.0 across the frame and 8.1 at 1:1 over skin.

SCUNet's `scunet_color_real_psnr` is trained on *realistic* degradations
rather than synthetic Gaussian noise, which is what makes it hold up on an
actual camera file. Measured on that same photo: sigma 4.0 -> 0.8 full
frame, 8.1 -> 0.3 at 1:1, with eyelashes, eyebrow hairs and catchlights
intact -- it is not simply blurring.

It is also the permissive choice, which was the other requirement. NAFNet is
comparable and MIT, but its weights are only distributed via Google Drive,
which cannot be fetched reproducibly at image build time.

Cost
----
This is by far the most expensive model in the service: unlike the colour
model (which downsamples to 256x256 internally) and the segmenters (1024px),
it works at whatever resolution it is handed. ~22s for 1200x1800 on an M4
Pro. So it runs ONCE per import, over the stored baseline, and every look
and preview inherits the result for free. Full-resolution renders tile it.
"""
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("photo-enhance.denoise")

_cache = {}

# SCUNet downsamples three times, so both edges must be a multiple of 8.
_ALIGN = 8

# Full-resolution work is tiled to bound peak memory -- the same reason the
# rest of the pipeline was reworked. Tiles overlap and are feathered together
# so the seams don't show as banding in flat areas, which is exactly where
# denoising is most visible.
TILE = 512
OVERLAP = 64


def _device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _model():
    if "scunet" not in _cache:
        from vendor.network_scunet import SCUNet
        path = os.environ.get("DENOISE_WEIGHTS_PATH",
                              os.path.join(os.path.dirname(__file__), "weights", "scunet_color_real_psnr.pth"))
        m = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval().to(_device())
        _cache["scunet"] = m
        logger.info("denoise model loaded from %s", path)
    return _cache["scunet"]


def available() -> bool:
    path = os.environ.get("DENOISE_WEIGHTS_PATH",
                          os.path.join(os.path.dirname(__file__), "weights", "scunet_color_real_psnr.pth"))
    return os.path.exists(path)


def loaded() -> bool:
    return bool(_cache)


def unload():
    _cache.clear()
    import gc
    gc.collect()


def estimate_sigma(arr: np.ndarray) -> float:
    """Noise level in 8-bit units, from the smoothest 15% of 32px blocks.

    Deliberately measured on flat regions only: a global standard deviation
    would mostly report how much detail the photo has, not how noisy it is.
    """
    g = (np.clip(arr, 0, 1).mean(axis=2) * 255).astype(np.float32)
    bs = 32
    h, w = g.shape
    if h < bs * 2 or w < bs * 2:
        return 0.0
    blocks = (g[:h // bs * bs, :w // bs * bs]
              .reshape(h // bs, bs, w // bs, bs).swapaxes(1, 2).reshape(-1, bs * bs))
    var = blocks.var(axis=1)
    flat = blocks[var <= np.percentile(var, 15)]
    if not len(flat):
        return 0.0
    return float(np.median(np.abs(flat - flat.mean(axis=1, keepdims=True))) * 1.4826)


@torch.no_grad()
def _run(t: torch.Tensor) -> torch.Tensor:
    _, _, h, w = t.shape
    ph, pw = (-h) % _ALIGN, (-w) % _ALIGN
    if ph or pw:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")
    out = _model()(t.to(_device())).cpu()
    return out[:, :, :h, :w].clamp(0, 1)


def _blend_window(n: int, lead: int, trail: int) -> np.ndarray:
    """Ramp up over `lead` samples and down over `trail`, flat between."""
    w = np.ones(n, dtype=np.float32)
    if lead:
        w[:lead] = np.linspace(0, 1, lead, endpoint=False)
    if trail:
        w[n - trail:] = np.linspace(1, 0, trail, endpoint=False)
    return w


def denoise(arr: np.ndarray, progress=None) -> np.ndarray:
    """(H, W, 3) float32 in [0,1] -> denoised, same shape.

    Tiled with feathered overlap when the image is large. A hard tile grid
    leaves visible seams in smooth areas; weighting each tile by a ramp and
    normalising by the accumulated weight removes them.
    """
    h, w = arr.shape[:2]
    if max(h, w) <= TILE:
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0)
        return _run(t).squeeze(0).permute(1, 2, 0).numpy()

    step = TILE - OVERLAP
    ys = list(range(0, max(h - OVERLAP, 1), step))
    xs = list(range(0, max(w - OVERLAP, 1), step))
    acc = np.zeros_like(arr, dtype=np.float32)
    wsum = np.zeros((h, w, 1), dtype=np.float32)
    total = len(ys) * len(xs)
    done = 0
    for y in ys:
        for x in xs:
            y1, x1 = min(y + TILE, h), min(x + TILE, w)
            y0, x0 = max(0, y1 - TILE), max(0, x1 - TILE)
            tile = arr[y0:y1, x0:x1]
            t = torch.from_numpy(np.ascontiguousarray(tile)).permute(2, 0, 1).unsqueeze(0)
            out = _run(t).squeeze(0).permute(1, 2, 0).numpy()
            wy = _blend_window(y1 - y0, OVERLAP if y0 > 0 else 0, OVERLAP if y1 < h else 0)
            wx = _blend_window(x1 - x0, OVERLAP if x0 > 0 else 0, OVERLAP if x1 < w else 0)
            wt = (wy[:, None] * wx[None, :])[..., None]
            acc[y0:y1, x0:x1] += out * wt
            wsum[y0:y1, x0:x1] += wt
            done += 1
            if progress:
                progress(done, total)
    return np.clip(acc / np.maximum(wsum, 1e-6), 0, 1)


def blend(original: np.ndarray, denoised: np.ndarray, amount: float) -> np.ndarray:
    """Dial the effect back toward the original. Full strength can read as
    waxy on skin, so this is exposed rather than fixed at 1.0."""
    a = float(np.clip(amount, 0.0, 1.0))
    if a >= 1.0:
        return denoised
    if a <= 0.0:
        return original
    return np.clip(original * (1 - a) + denoised * a, 0, 1)
