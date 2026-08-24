"""End-to-end processing, preview-first.

Decode + model inference happen once, at full resolution -- both are cheap
regardless of resolution (the model's weight-predictor CNN always resizes
its input to 256x256 internally, and its LUT color correction is a
pointwise per-pixel operation, so applying it at full res costs about the
same as applying it small). The genuinely expensive step is the 8+
deterministic style-preset effects (dehaze/glow box blurs, O(pixels) each).

So: on import, run decode + model once, resize the corrected result down,
and render fast *preview* JPEGs for every style from that small image
(process_preview). A full-resolution render for one specific style is only
generated the first time someone actually requests to download it
(render_full_style), and cached after that -- most photos in a browsing
session never get all styles downloaded, so this avoids doing that
expensive work for styles nobody asked for.
"""
import asyncio
import io
import logging
import os
import tempfile

import numpy as np
import rawpy
import torch
from PIL import Image

import immich_client
import storage
from model_runtime import runtime
from presets import PRESETS

logger = logging.getLogger("photo-enhance.pipeline")

RAW_EXTENSIONS = {".cr3", ".cr2", ".arw", ".dng", ".nef", ".raf", ".orf", ".rw2"}
JPEG_QUALITY = 95  # full-res downloads -- quality over file size, 10MB+ is fine
THUMB_LONG_EDGE = 480
THUMB_QUALITY = 82


def _resize_long_edge(img: Image.Image, long_edge: int) -> Image.Image:
    w, h = img.size
    scale = long_edge / max(w, h)
    if scale >= 1:
        return img
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def _decode_full(raw_bytes: bytes, filename: str) -> np.ndarray:
    """Returns an (H, W, 3) float32 array in [0, 1] at native resolution."""
    ext = os.path.splitext(filename.lower())[1]
    if ext in RAW_EXTENSIONS:
        with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
            tmp.write(raw_bytes)
            tmp.flush()
            with rawpy.imread(tmp.name) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=8,
                    output_color=rawpy.ColorSpace.sRGB,
                )
        img = Image.fromarray(rgb)
    else:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def _tensor_to_jpeg_bytes(t: torch.Tensor, quality: int) -> bytes:
    arr = (t.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _resize_tensor(t: torch.Tensor, long_edge: int) -> torch.Tensor:
    """t: (1,3,H,W) in [0,1]."""
    h, w = t.shape[2], t.shape[3]
    scale = long_edge / max(h, w)
    if scale >= 1:
        return t
    new_h, new_w = round(h * scale), round(w * scale)
    return torch.nn.functional.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)


async def process_preview(raw_bytes: bytes, filename: str, source_type: str, immich_asset_id: str | None = None) -> str:
    """Decode + model once at full res, render small previews for every
    style. Fast (~1-3s) -- no full-resolution style rendering happens here."""
    arr = await asyncio.to_thread(_decode_full, raw_bytes, filename)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()

    baseline = await runtime.infer(tensor)
    small_baseline = await asyncio.to_thread(_resize_tensor, baseline, THUMB_LONG_EDGE)

    import_id = storage.create_import(filename, source_type, immich_asset_id)

    if source_type == "upload":
        await asyncio.to_thread(storage.save_source_upload, import_id, raw_bytes, filename)

    small_input = await asyncio.to_thread(_resize_tensor, tensor, THUMB_LONG_EDGE)
    orig_thumb_bytes = await asyncio.to_thread(_tensor_to_jpeg_bytes, small_input, THUMB_QUALITY)
    await asyncio.to_thread(storage.save_original_thumb, import_id, orig_thumb_bytes)

    for key, label, params in PRESETS:
        styled = await runtime.render_style(small_baseline, params)
        thumb_bytes = await asyncio.to_thread(_tensor_to_jpeg_bytes, styled, THUMB_QUALITY)
        await asyncio.to_thread(storage.save_preview, import_id, key, label, thumb_bytes)

    logger.info("preview: %s -> import %s (%d styles)", filename, import_id, len(PRESETS))
    return import_id


async def _get_original_bytes(import_id: str) -> tuple[bytes, str]:
    meta = storage.get_import(import_id)
    if meta["source_type"] == "upload":
        data, name = await asyncio.to_thread(storage.get_source_upload_bytes, import_id)
        return data, name
    else:
        data, name = await immich_client.get_original_bytes(meta["immich_asset_id"])
        return data, meta["source_name"] or name


async def render_full_style(import_id: str, style_key: str) -> str:
    """Full-resolution render for one style, generated on first request and
    cached on disk afterwards. Redecodes + reruns the model (cheap) then
    runs just this one preset at full res (the actually expensive part)."""
    if storage.full_render_exists(import_id, style_key):
        return storage.full_render_path(import_id, style_key)

    params = next(p for k, l, p in PRESETS if k == style_key)
    raw_bytes, filename = await _get_original_bytes(import_id)

    arr = await asyncio.to_thread(_decode_full, raw_bytes, filename)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    baseline = await runtime.infer(tensor)
    styled = await runtime.render_style(baseline, params)
    jpeg_bytes = await asyncio.to_thread(_tensor_to_jpeg_bytes, styled, JPEG_QUALITY)
    await asyncio.to_thread(storage.save_full_render, import_id, style_key, jpeg_bytes)

    logger.info("full render: import %s style %s", import_id, style_key)
    return storage.full_render_path(import_id, style_key)
