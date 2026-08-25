"""Lazy-load / idle-unload wrapper around the trained model.

Per the project's RAM requirement: the model is not loaded at process
startup. The first inference request loads it (fast -- it's a few MB); a
background task then drops it and runs gc.collect() after IDLE_UNLOAD_MINUTES
with no requests. A single asyncio.Lock serializes inference so peak
transient memory (the RAW-decode buffer, not the model) stays bounded to one
request at a time regardless of concurrent uploads.
"""
import asyncio
import gc
import logging
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.model import AdaptiveLUTModel  # noqa: E402
from presets import apply_look  # noqa: E402

logger = logging.getLogger("photo-enhance.model_runtime")

WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS_PATH", os.path.join(os.path.dirname(__file__), "weights", "model.pt"))
IDLE_UNLOAD_MINUTES = float(os.environ.get("IDLE_UNLOAD_MINUTES", "15"))

# "cpu" (default, plain eager PyTorch -- safest, works everywhere) |
# "openvino_cpu" (exercises the OpenVINO torch.compile path on any machine,
# useful for testing the integration itself) | "openvino_gpu" (optiplex's
# integrated GPU via /dev/dri passthrough -- only valid on that box; there's
# no Intel iGPU on this Mac to actually verify GPU execution against, only
# that the openvino_cpu path compiles and runs correctly).
INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "cpu")


def _return_arenas_to_os():
    """gc.collect() frees the objects, but glibc keeps the arenas, so RSS
    barely moves and creeps upward across load/unload cycles (measured
    1643 -> 1870 -> 1991 MB over three enhanced imports). malloc_trim hands
    the free arenas back, which is what actually matters inside a container
    where RSS is the number the host sees. Linux/glibc only -- absent on
    macOS and musl, so failure here is normal and not worth logging loudly.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def compile_for_device(fn):
    """Wrap a plain function or bound method with torch.compile targeting
    the configured OpenVINO device, or return it unchanged for "cpu"."""
    if INFERENCE_DEVICE == "cpu":
        return fn
    ov_device = "GPU" if INFERENCE_DEVICE == "openvino_gpu" else "CPU"
    return torch.compile(fn, backend="openvino", options={"device": ov_device}, dynamic=True)


class ModelRuntime:
    """Bundles the model AND the compiled preset-rendering function under one
    load/unload lifecycle -- both are OpenVINO-compiled together on first use
    and both get dropped together on idle-unload, so there's no separate
    always-resident compiled artifact outside the model's own lifecycle."""

    def __init__(self):
        self._model = None
        self._render_style = None
        self._lock = asyncio.Lock()
        self._last_used = 0.0
        self._watchdog_task = None

    def weights_available(self) -> bool:
        return os.path.exists(WEIGHTS_PATH)

    def _load(self):
        model = AdaptiveLUTModel()
        ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        model.infer = compile_for_device(model.infer)
        self._render_style = compile_for_device(apply_look)
        logger.info("model loaded from %s (device=%s)", WEIGHTS_PATH, INFERENCE_DEVICE)
        return model

    def _unload(self):
        released = False
        if self._model is not None:
            self._model = None
            self._render_style = None
            released = True

        # The segmentation models are ~1.2GB -- vastly more than the 2.4MB
        # colour model -- and they cache themselves on first use in
        # masking._cache. Releasing only the colour model (as this once did)
        # meant enhanced mode permanently held that 1.2GB after a single
        # photo, which defeats the whole point of idle-unloading. Import
        # lazily so classic mode never pulls transformers in at all.
        try:
            import masking
            if masking.loaded():
                masking.unload()
                released = True
        except Exception as e:  # noqa: BLE001
            logger.warning("could not release segmentation models: %s", e)

        if released:
            gc.collect()
            _return_arenas_to_os()
            logger.info("models idle-unloaded after %.1f min", IDLE_UNLOAD_MINUTES)

    async def _watchdog(self):
        while True:
            await asyncio.sleep(60)
            if (time.time() - self._last_used) <= IDLE_UNLOAD_MINUTES * 60:
                continue
            try:
                import masking
                seg_loaded = masking.loaded()
            except Exception:  # noqa: BLE001
                seg_loaded = False
            # Segmentation can outlive the colour model, so gate on either
            # being resident rather than on self._model alone.
            if self._model is not None or seg_loaded:
                async with self._lock:
                    self._unload()

    def start_watchdog(self):
        if self._watchdog_task is None:
            self._watchdog_task = asyncio.create_task(self._watchdog())

    def _infer_sync(self, img_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out, _ = self._model.infer(img_tensor)
        return out

    async def infer(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """img_tensor: (1, 3, H, W) float32 in [0, 1] on CPU. Returns the
        model's corrected-baseline output, same shape. Runs on a worker
        thread -- this is several seconds of synchronous CPU work at full
        resolution, and doing it directly on the event loop thread would
        freeze every other request (gallery, thumbnails, health checks) for
        the whole duration, not just the ones waiting on this import."""
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load)
            self._last_used = time.time()
            out = await asyncio.to_thread(self._infer_sync, img_tensor)
            self._last_used = time.time()
            return out

    async def render_style(self, baseline: torch.Tensor, params: dict) -> torch.Tensor:
        """baseline: (3, H, W) or (B, 3, H, W) float32 in [0, 1]. Applies one
        style preset via the same compiled backend as the model, on a worker
        thread for the same reason as infer()."""
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load)
            self._last_used = time.time()
            with torch.no_grad():
                out = await asyncio.to_thread(self._render_style, baseline, params)
            self._last_used = time.time()
            return out

    def status(self) -> dict:
        try:
            import masking
            seg = masking.loaded()
        except Exception:  # noqa: BLE001
            seg = False
        return {
            "weights_available": self.weights_available(),
            "loaded": self._model is not None,
            "segmentation_loaded": seg,
            "idle_unload_minutes": IDLE_UNLOAD_MINUTES,
            "inference_device": INFERENCE_DEVICE,
        }


runtime = ModelRuntime()
