"""End-to-end processing: raw bytes in -> style-variant JPEGs saved to the
gallery. Shared by both ingest paths (Immich import and plain upload).
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

import storage
from model_runtime import runtime
from presets import PRESETS

logger = logging.getLogger("photo-enhance.pipeline")

RAW_EXTENSIONS = {".cr3", ".cr2", ".arw", ".dng", ".nef", ".raf", ".orf", ".rw2"}
JPEG_QUALITY = 95  # full native resolution output -- quality over speed, 10MB+ renders are expected
THUMB_LONG_EDGE = 480  # gallery-grid preview size; the full-res file is still the download/original
THUMB_QUALITY = 82


def _make_thumb_bytes(img: Image.Image) -> bytes:
    w, h = img.size
    scale = THUMB_LONG_EDGE / max(w, h)
    thumb = img if scale >= 1 else img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=THUMB_QUALITY)
    return buf.getvalue()


def _decode_to_array(raw_bytes: bytes, filename: str) -> np.ndarray:
    """Returns an (H, W, 3) float32 array in [0, 1] at the source's native
    resolution. No downsizing: output quality is prioritized over per-photo
    processing time -- expect several seconds per style at full sensor
    resolution (24-45MP), not the sub-second numbers a downsized render gets."""
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

    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def _encode_and_save(import_id: str, key: str, label: str, styled: torch.Tensor):
    """Numpy conversion + JPEG encode (full-res and thumb) + disk write --
    all synchronous CPU/IO work, run off the event loop thread (see process())."""
    # apply_look already clamps at every stage internally, but a value
    # slightly outside [0,1] reaching astype(uint8) wraps around (numpy
    # overflow) rather than clipping -- producing exactly the kind of
    # chaotic pixel corruption a missing clamp caused elsewhere in this
    # project. Defensive clamp here regardless of upstream guarantees.
    styled_np = (styled.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    styled_img = Image.fromarray(styled_np, "RGB")
    buf = io.BytesIO()
    styled_img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    thumb_bytes = _make_thumb_bytes(styled_img)
    storage.save_render(import_id, key, label, buf.getvalue(), thumb_bytes)


async def process(raw_bytes: bytes, filename: str) -> str:
    """Decodes, runs the model, renders every style preset, saves to the
    gallery. Returns the new import_id.

    Every synchronous, CPU-bound step (RAW decode, model inference, preset
    rendering, JPEG encoding) runs via asyncio.to_thread rather than directly
    on the event loop -- otherwise the whole service (gallery browsing,
    thumbnails, health checks, Immich search) would freeze for the several
    seconds a full-resolution photo takes to process, not just the request
    that triggered it.
    """
    arr = await asyncio.to_thread(_decode_to_array, raw_bytes, filename)

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    baseline = await runtime.infer(tensor)

    import_id = storage.create_import(filename)
    for key, label, params in PRESETS:
        styled = await runtime.render_style(baseline, params)
        await asyncio.to_thread(_encode_and_save, import_id, key, label, styled)

    logger.info("processed %s -> import %s (%d styles)", filename, import_id, len(PRESETS))
    return import_id
