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
import threading

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

# Full-resolution renders tile a 6000x4000 frame into 126 tiles and, until
# now, ran them one at a time on one device while the other sat idle. The
# tiles are independent, so both can work. Measured on optiplex:
#
#     sequential iGPU     3.47 s/tile   7.3 min for 6000x4000
#     iGPU + CPU          2.79 s/tile   5.9 min          1.25x
#
# Short of the 1.79x the per-device rates imply, because OpenVINO's CPU
# inference competes for the very cores the GPU driver needs on the host
# side -- but a real 19% off the longest wait in the product.
#
# Only for big jobs: a second compiled copy of SCUNet costs resident memory,
# and an import (6 tiles) does not run long enough to earn that back.
HYBRID_MIN_TILES = 16


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


SCUNET_IR = os.environ.get(
    "SCUNET_IR_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "scunet_tile.xml"))


def _converted():
    """SCUNet as an OpenVINO model at the fixed tile shape.

    Read from the IR baked into the image when it is there, and only fall
    back to converting at runtime when it is not.

    That fallback is genuinely expensive and it is worth saying why:
    `ov.convert_model` on SCUNet needs **more than 12 GB** -- measured, by
    OOM-killing a 12 GB-capped container. On a 15 GB box also running Immich
    and Jellyfin that is a spike big enough to get something else killed by
    the kernel. Doing it once at build time removes the spike entirely, and
    removes the ~35 s stall the first denoise used to pay.

    The CPU and GPU builds compile from this same graph -- converting per
    device would double the worst moment.
    """
    if "ov_model" not in _cache:
        import openvino as ov
        if os.path.exists(SCUNET_IR):
            _cache["ov_model"] = ov.Core().read_model(SCUNET_IR)
            logger.info("denoise: loaded prebuilt IR %s", SCUNET_IR)
        else:
            logger.warning("denoise: no prebuilt IR at %s, converting at runtime "
                           "(needs >12GB, one time)", SCUNET_IR)
            _cache["ov_model"] = ov.convert_model(
                _model(), example_input=torch.zeros(1, 3, TILE, TILE))
    return _cache["ov_model"]


def _ov_tile_model(device: str = "GPU"):
    """SCUNet compiled by OpenVINO for `device` at the fixed tile shape.

    Every tile is exactly TILE x TILE (edge tiles clamp their origin rather
    than shrinking), so a static shape is safe -- and static is what makes
    the GPU plugin worth using at all. Measured on optiplex, per tile:
    torch CPU 5.67s, OpenVINO CPU 4.27s, OpenVINO iGPU 3.39s.

    NB: FP16 is already the iGPU default and is the right choice. INT8 was
    built and measured -- 4.11s, i.e. 20% SLOWER -- because UHD 630 is
    Gen9.5 and has no usable INT8 dot-product path. Don't re-try it.
    """
    if device == "GPU" and _configured_device() == "cpu":
        logger.info("denoise: pinned to CPU by configuration")
        return None
    key = f"ov_{device}"
    if key in _cache:
        return _cache[key]
    _cache[key] = None
    try:
        import openvino as ov
        core = ov.Core()
        if device not in core.available_devices:
            logger.info("denoise: no OpenVINO %s device", device)
            return None
        _cache[key] = core.compile_model(_converted(), device)
        name = core.get_property(device, "FULL_DEVICE_NAME") if device == "GPU" else device
        logger.info("denoise: compiled for %s (%s)", device, name)
    except Exception as e:  # noqa: BLE001
        logger.warning("denoise: OpenVINO %s unavailable (%s)", device, e)
        _cache[key] = None
    return _cache[key]


def _openvino_tile_model():
    """The iGPU build. Kept as the name the rest of the module uses."""
    return _ov_tile_model("GPU")


def device_in_use() -> str:
    """What the denoiser will actually run on, for /health.

    Reported "cpu" on a Mac while SCUNet was demonstrably resident on mps:0 --
    it only ever looked at the OpenVINO cache, which is Intel-iGPU-specific and
    is always None on Apple silicon. The torch device is the fallback path, so
    it has to be part of the answer.
    """
    if _cache.get("ov_GPU") is not None:
        return "gpu+cpu" if _cache.get("ov_CPU") is not None else "gpu"
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
    # NOT bool(_cache): a failed device probe stores None, which would
    # otherwise report the model as resident forever and stop the
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
            arr = comp(t.numpy())[comp.output(0)]
            return torch.from_numpy(arr)[:, :, :h, :w].clamp(0, 1)
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
    coords = []
    for y in ys:
        for x in xs:
            y1, x1 = min(y + TILE, h), min(x + TILE, w)
            coords.append((max(0, y1 - TILE), y1, max(0, x1 - TILE), x1))
    acc = np.zeros_like(arr, dtype=np.float32)
    wsum = np.zeros((h, w, 1), dtype=np.float32)
    total = len(coords)

    def compose(y0, y1, x0, x1, out):
        wy = _blend_window(y1 - y0, OVERLAP if y0 > 0 else 0, OVERLAP if y1 < h else 0)
        wx = _blend_window(x1 - x0, OVERLAP if x0 > 0 else 0, OVERLAP if x1 < w else 0)
        wt = (wy[:, None] * wx[None, :])[..., None]
        acc[y0:y1, x0:x1] += out * wt
        wsum[y0:y1, x0:x1] += wt

    runners = _hybrid_runners(h, w, total)
    if runners is None:
        done = 0
        for (y0, y1, x0, x1) in coords:
            t = torch.from_numpy(np.ascontiguousarray(arr[y0:y1, x0:x1])) \
                     .permute(2, 0, 1).unsqueeze(0)
            compose(y0, y1, x0, x1, _run(t).squeeze(0).permute(1, 2, 0).numpy())
            done += 1
            if progress:
                progress(done, total)
    else:
        _run_hybrid(arr, coords, runners, compose, progress, total)

    return np.clip(acc / np.maximum(wsum, 1e-6), 0, 1)


def _hybrid_runners(h: int, w: int, total: int):
    """Two callables (one per device) when splitting the work is worth it.

    Requires every tile to be exactly TILE x TILE, which is what the static
    OpenVINO shape assumes: a frame shorter than TILE on one side produces
    short tiles and must stay on the sequential path.
    """
    if total < HYBRID_MIN_TILES or h < TILE or w < TILE:
        return None
    gpu = _ov_tile_model("GPU")
    if gpu is None:
        return None
    cpu = _ov_tile_model("CPU")
    if cpu is None:
        return None

    def make(compiled):
        port = compiled.output(0)
        # One compiled model per thread: a single compiled model shares one
        # default infer request, so calling it from both threads at once
        # would interleave inputs.
        return lambda nchw: compiled(nchw)[port]
    return [make(gpu), make(cpu)]


def _run_hybrid(arr, coords, runners, compose, progress, total):
    """Feed one shared queue of tiles to every device at once.

    Faster devices simply take more tiles; no static split to tune, and a
    device that stalls cannot hold up the others.
    """
    idx = [0]
    done = [0]
    err = []
    pick = threading.Lock()      # hands out tile indices
    write = threading.Lock()     # guards acc/wsum, whose tiles overlap

    def worker(run):
        while True:
            with pick:
                if err or idx[0] >= len(coords):
                    return
                i = idx[0]; idx[0] += 1
            y0, y1, x0, x1 = coords[i]
            try:
                nchw = np.ascontiguousarray(
                    arr[y0:y1, x0:x1].transpose(2, 0, 1)[None])
                out = np.asarray(run(nchw))[0].transpose(1, 2, 0)
                with write:
                    compose(y0, y1, x0, x1, np.clip(out, 0, 1))
                    done[0] += 1
                    n = done[0]
                # Outside the write lock: `progress` raises RenderCancelled to
                # abort a render, and it must not do that holding a lock.
                if progress:
                    progress(n, total)
            except BaseException as e:  # noqa: BLE001
                with pick:
                    err.append(e)
                return

    threads = [threading.Thread(target=worker, args=(r,), daemon=True) for r in runners]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if err:
        # Propagate the first failure -- notably RenderCancelled, which the
        # caller relies on to stop a deleted photo's render mid-flight.
        raise err[0]


def blend(original: np.ndarray, denoised: np.ndarray, amount: float) -> np.ndarray:
    """Dial the effect back toward the original. Full strength can read as
    waxy on skin, so this is exposed rather than fixed at 1.0."""
    a = float(np.clip(amount, 0.0, 1.0))
    if a >= 1.0:
        return denoised
    if a <= 0.0:
        return original
    return np.clip(original * (1 - a) + denoised * a, 0, 1)
