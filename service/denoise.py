"""Denoising with SCUNet, on the model's corrected baseline.

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

NAFNet is comparable, but its weights are only distributed via Google
Drive, which cannot be fetched reproducibly at image build time.

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
import threading

import numpy as np

import ov_infer
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("photo-enhance.denoise")

_cache = {}
# Guards the one-off OpenVINO conversion; see _openvino_tile_model.
# Reentrant: _ffdnet_ov holds this while calling _ffdnet_model, which takes it
# again to load the weights. With a plain Lock that is a deadlock, and it is
# not a subtle one -- the import thread stops forever and the photo sits at
# "processing" until the container is restarted. It shipped, because the only
# machine with an OpenVINO GPU is the server: everywhere else _ffdnet_ov
# returns before reaching that call, so nothing local could reach the bug.
_build_lock = threading.RLock()

# SCUNet downsamples three times, so both edges must be a multiple of 8.
_ALIGN = 8

# Full-resolution work is tiled to bound peak memory -- the same reason the
# rest of the pipeline was reworked. Tiles overlap and are feathered together
# so the seams don't show as banding in flat areas, which is exactly where
# denoising is most visible.
TILE = 512
OVERLAP = 64


# auto | cpu | gpu. "auto" uses the Intel iGPU through OpenVINO when a driver
# is actually present, and falls back silently otherwise -- the container runs
# unchanged on a box with no GPU.
def _configured_device() -> str:
    """Env wins (it is how the container is pinned), otherwise the setting."""
    env = os.environ.get("DENOISE_DEVICE")
    if env:
        return env
    try:
        import settings
        return (settings.get("denoise_device") or "auto").lower()
    except Exception:  # noqa: BLE001
        return "auto"


def _device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _openvino_tile_model():
    """SCUNet compiled for the iGPU at the fixed tile shape, or None.

    Every tile is exactly TILE x TILE (edge tiles clamp their origin rather
    than shrinking), so a static shape is safe -- and static is what makes
    the GPU plugin worth using.

    Measured on an idle optiplex, per tile: torch CPU 5.67s, OpenVINO CPU
    4.27s, OpenVINO iGPU 3.39s. (The 9.75s -> 3.86s / 2.52x once recorded
    here was taken on a busy box and overstates the win.)

    Two things deliberately NOT done, both measured:
      * INT8 via NNCF, calibrated on real tiles -- 4.11s, i.e. 20% SLOWER.
        UHD 630 is Gen9.5 and has no usable INT8 dot-product path; FP16 is
        already the iGPU default and is the right choice here.
      * Baking the IR at build time, as masking.py does for BiRefNet --
        `ov.convert_model` on SCUNet needs MORE THAN 12GB, which OOM-kills
        both a capped container and a 16GB CI runner. That is also why this
        conversion is a real memory event at runtime, once per process.
    """
    if _configured_device() == "cpu":
        logger.info("denoise: pinned to CPU by configuration")
        return None
    if "ov" in _cache:
        return _cache["ov"]
    # Serialised: two imports can start together, and without this the second
    # thread saw the None written below while the first was still converting,
    # silently dropping that whole image onto the slow path. The conversion is
    # also a multi-gigabyte event that must not happen twice at once.
    with _build_lock:
        if "ov" in _cache:
            return _cache["ov"]
        _cache["ov"] = None
        try:
            import openvino as ov
            core = ov.Core()
            if "GPU" not in core.available_devices:
                logger.info("denoise: no OpenVINO GPU device, staying on CPU")
                return None
            example = torch.zeros(1, 3, TILE, TILE)
            m = ov.convert_model(_model(), example_input=example)
            _cache["ov"] = core.compile_model(m, "GPU")
            logger.info("denoise: compiled for iGPU (%s)",
                        core.get_property("GPU", "FULL_DEVICE_NAME"))
        except Exception as e:  # noqa: BLE001
            logger.warning("denoise: iGPU unavailable (%s), staying on CPU", e)
            _cache["ov"] = None
        return _cache["ov"]


def device_in_use() -> str:
    """What the denoiser will actually run on, for /health.

    Reported "cpu" on a Mac while SCUNet was demonstrably resident on mps:0 --
    it only ever looked at the OpenVINO cache, which is Intel-iGPU-specific
    and is always None on Apple silicon. The torch device is the fallback
    path, so it has to be part of the answer.
    """
    if _cache.get("ov") is not None:
        return "gpu"          # Intel iGPU via OpenVINO (optiplex)
    return _device()          # "mps" (Apple GPU) or "cpu"


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
    # NOT bool(_cache): a failed iGPU probe stores _cache["ov"] = None, which
    # would otherwise report the model as resident forever and stop the
    # idle-unload watchdog from ever firing.
    return any(v is not None for v in _cache.values())


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
    if t.shape[2] == TILE and t.shape[3] == TILE:
        comp = _openvino_tile_model()
        if comp is not None:
            # Not comp(...): that shares one infer request across every
            # thread, and a second concurrent render then fails outright
            # with "Infer Request is busy".
            arr = ov_infer.infer(comp, "denoise_tile", t.numpy())
            if arr.shape == tuple(t.shape):
                return torch.from_numpy(arr)[:, :, :h, :w].clamp(0, 1)
            # Anything else is a bug, and passing it on makes the tiler fail
            # far away with an unreadable broadcast error. Say what happened
            # and finish the tile on the CPU rather than losing the import.
            logger.error("denoise: iGPU returned %s for a %s tile, using CPU "
                         "for this one", arr.shape, tuple(t.shape))
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



# FFDNet takes the noise level as an input, and the estimator here reports the
# standard deviation of the flattest blocks. Those are not the same number: a
# real sensor's noise is signal-dependent and not the white Gaussian FFDNet
# was trained on, so feeding the raw estimate leaves the result visibly
# under-denoised. Twice the estimate matched SCUNet closely on the frame this
# was calibrated against -- ONE frame, which is the weakest part of this and
# is written down as such in research/denoise-speed/FINDINGS.md.
FFDNET_SIGMA_SCALE = 2.0

# Below this the model is being asked to remove almost nothing, and the
# estimator is noisier than the noise at that point.
FFDNET_MIN_SIGMA = 1.0


def _ffdnet_model():
    if "ffdnet" in _cache:
        return _cache["ffdnet"]
    with _build_lock:
        if "ffdnet" in _cache:
            return _cache["ffdnet"]
        from vendor.network_ffdnet import FFDNet
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "weights", "ffdnet_color.pth")
        m = FFDNet(in_nc=3, out_nc=3, nc=96, nb=12, act_mode="R")
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval().to(_device())
        _cache["ffdnet"] = m
        logger.info("ffdnet loaded from %s", path)
        return m


def _ffdnet_ov():
    """FFDNet compiled for the iGPU at the fixed tile shape, or None.

    Worth doing here in a way it is not for SCUNet. Converting SCUNet peaks
    above 12GB and has OOM-killed this service; converting FFDNet takes 2.4s
    and peaks at 0.70GB, because it is 0.85M parameters rather than 17.9M.

    Measured on the optiplex UHD 630, per 512px tile: torch CPU 0.300s,
    OpenVINO CPU 0.213s, OpenVINO iGPU 0.190s -- so 26.6s for a 26MP frame
    against 41.6s on torch, and against 476s for SCUNet on the same iGPU.
    """
    if _configured_device() == "cpu":
        return None
    if "ffdnet_ov" in _cache:
        return _cache["ffdnet_ov"]
    with _build_lock:
        if "ffdnet_ov" in _cache:
            return _cache["ffdnet_ov"]
        _cache["ffdnet_ov"] = None
        try:
            import openvino as ov
            core = ov.Core()
            if "GPU" not in core.available_devices:
                logger.info("ffdnet: no OpenVINO GPU device, staying on CPU")
                return None
            example = (torch.zeros(1, 3, TILE, TILE), torch.zeros(1, 1, 1, 1))
            m = ov.convert_model(_ffdnet_model(), example_input=example)
            _cache["ffdnet_ov"] = core.compile_model(m, "GPU")
            logger.info("ffdnet: compiled for iGPU (%s)",
                        core.get_property("GPU", "FULL_DEVICE_NAME"))
        except Exception as e:  # noqa: BLE001
            logger.warning("ffdnet: iGPU unavailable (%s), staying on CPU", e)
            _cache["ffdnet_ov"] = None
        return _cache["ffdnet_ov"]


@torch.no_grad()
def _run_ffdnet(t: torch.Tensor, sigma8: float) -> torch.Tensor:
    """One tile through FFDNet. `sigma8` is the noise level in 8-bit units."""
    _, _, h, w = t.shape
    # The pixel-shuffle needs even dimensions.
    ph, pw = (-h) % 2, (-w) % 2
    if ph or pw:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")

    # The compiled model is built at one fixed tile shape, which is what makes
    # the GPU plugin worth using; anything else goes to torch.
    if t.shape[2] == TILE and t.shape[3] == TILE:
        comp = _ffdnet_ov()
        if comp is not None:
            level = np.full((1, 1, 1, 1), sigma8 / 255.0, dtype=np.float32)
            arr = ov_infer.infer(comp, "ffdnet_tile", {0: t.numpy(), 1: level})
            if arr.shape == tuple(t.shape):
                return torch.from_numpy(arr)[:, :, :h, :w].clamp(0, 1)
            logger.error("ffdnet: iGPU returned %s for a %s tile, using CPU "
                         "for this one", arr.shape, tuple(t.shape))

    m = _ffdnet_model()
    dev = _device()
    level = torch.full((1, 1, 1, 1), sigma8 / 255.0, device=dev)
    out = m(t.to(dev), level).cpu()
    return out[:, :, :h, :w].clamp(0, 1)


def _configured_method() -> str:
    try:
        import settings
        return (settings.get("denoise_method") or "quality").lower()
    except Exception:  # noqa: BLE001
        return "quality"


def _box(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1) window, via a summed-area table: O(1) per pixel."""
    pad = np.pad(a, ((r + 1, r), (r + 1, r)), mode="edge")
    s = pad.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return (s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]) / (k * k)


def guided_chroma(arr: np.ndarray, radius: int = 4, eps: float = 1e-3,
                  factor: int = 4) -> np.ndarray:
    """Denoise chroma only, with a guided filter, at reduced resolution.

    The fast alternative to the network. Measured against SCUNet on a 3200px
    working frame: 0.09s against 16s, and it moves the image about as far from
    the original as SCUNet does. It is NOT the same result -- it takes out the
    colour blotching and leaves the luma grain, which is the part the eye
    objects to. See research/denoise-speed/FINDINGS.md, where PSNR could not
    tell the two apart and only the crops could.

    Chroma tolerates this because the eye has far less acuity for it than for
    luminance, so filtering it at quarter scale is invisible while costing a
    sixteenth of the work. Luminance is used as the guide, which is what keeps
    colour from bleeding across edges.
    """
    a = arr.astype(np.float32, copy=False)
    h, w = a.shape[:2]
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    cb, cr = a[..., 2] - y, a[..., 0] - y

    # Shrinking needs room for the filter window on both sides of the result.
    factor = max(1, min(factor, h // (4 * radius + 4) or 1, w // (4 * radius + 4) or 1))
    sh, sw = max(1, h // factor), max(1, w // factor)

    def shrink(x):
        if factor == 1:
            return x
        t = torch.from_numpy(np.ascontiguousarray(x))[None, None]
        return torch.nn.functional.interpolate(t, size=(sh, sw), mode="area")[0, 0].numpy()

    def grow(x):
        if factor == 1:
            return x
        t = torch.from_numpy(np.ascontiguousarray(x))[None, None]
        return torch.nn.functional.interpolate(
            t, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()

    guide = shrink(y)
    mean_g = _box(guide, radius)
    var_g = _box(guide * guide, radius) - mean_g * mean_g

    def filt(chan):
        c = shrink(chan)
        mean_c = _box(c, radius)
        cov = _box(guide * c, radius) - mean_g * mean_c
        A = cov / (var_g + eps)
        B = mean_c - A * mean_g
        # A and B are smoothed before being grown, so the filter varies
        # gently rather than showing the block structure of the small grid.
        return grow(_box(A, radius)) * y + grow(_box(B, radius))

    cb2, cr2 = filt(cb), filt(cr)
    out = np.empty_like(a)
    out[..., 0] = y + cr2
    out[..., 2] = y + cb2
    # Green is recovered from the luminance identity, so Y is preserved
    # exactly and no luminance detail is touched.
    out[..., 1] = (y - 0.299 * out[..., 0] - 0.114 * out[..., 2]) / 0.587
    return np.clip(out, 0, 1)


def denoise(arr: np.ndarray, progress=None, method: str | None = None) -> np.ndarray:
    """(H, W, 3) float32 in [0,1] -> denoised, same shape.

    Tiled with feathered overlap when the image is large. A hard tile grid
    leaves visible seams in smooth areas; weighting each tile by a ramp and
    normalising by the accumulated weight removes them.

    `method` picks the engine, unset taking it from settings:

      quality  -- SCUNet. 3.40s a tile on the optiplex iGPU, 476s for 26MP.
      balanced -- FFDNet. 0.19s a tile on the iGPU, 26.6s for 26MP, and 43.25 dB
                  from SCUNet's output on the calibration frame.
      fast     -- the guided chroma filter. No network, ~0.1s, and it leaves
                  the luma grain rather than removing it.
    """
    method = method or _configured_method()
    if method == "fast":
        out = guided_chroma(arr)
        if progress:
            progress(1, 1)
        return out

    # FFDNet is non-blind, so the level is measured once for the whole frame
    # rather than per tile -- a tile of sky and a tile of foliage would
    # otherwise be told they carry different noise and be smoothed unequally.
    tile_sigma = None
    if method == "balanced":
        tile_sigma = max(estimate_sigma(arr) * FFDNET_SIGMA_SCALE, FFDNET_MIN_SIGMA)

    h, w = arr.shape[:2]
    def run_tile(t):
        return _run_ffdnet(t, tile_sigma) if tile_sigma is not None else _run(t)

    if max(h, w) <= TILE:
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0)
        return run_tile(t).squeeze(0).permute(1, 2, 0).numpy()

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
            out = run_tile(t).squeeze(0).permute(1, 2, 0).numpy()
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
