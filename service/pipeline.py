"""End-to-end processing, preview-first, with two modes.

classic   -- the global 3D-LUT correction plus full-frame style presets.
             Cheap, no segmentation models, this is what the stable service
             does today.
enhanced  -- additionally runs segmentation so different parts of the photo
             can be graded differently (subject vs background, sky vs
             ground). This is the piece the LUT model structurally cannot do
             on its own: it learns ONE colour mapping applied to every pixel
             identically, so it has no notion of *where* anything is.

Preview-first in both modes: decode + model inference run once at full
resolution (both are cheap at any size -- the weight-predictor CNN
downsamples to 256x256 internally and the LUT is a pointwise op), then the
expensive per-style effects are rendered small for the gallery. A
full-resolution render of one style happens only when it's actually
downloaded, and is cached after that.
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

import cropping
import immich_client
import region_grade as rg
import settings
import storage
from cinematic import CINEMATIC
from model_runtime import runtime
from presets import PRESETS

logger = logging.getLogger("photo-enhance.pipeline")

# Bounds the ENTIRE pipeline (decode included) across the whole service.
# model_runtime's own lock only ever covered the model call -- RAW decode
# runs via asyncio.to_thread *before* that, on Python's default thread pool,
# completely unbounded. 30 concurrent requests meant 30 simultaneous
# full-resolution decodes (each ~250-400MB in memory for a 24MP+ RAW) on
# every CPU core at once -- exactly what pegged the host. This semaphore is
# the actual fix; the model-level lock alone was never enough.
_pipeline_gate = asyncio.Semaphore(settings.get("max_concurrent_jobs") or 3)

# Segmentation is far heavier than the LUT model (~1.2GB resident, ~7s CPU
# per image on a fast laptop and slower on the deploy box), so it gets its
# own stricter gate on top of the pipeline gate. Without this, three
# concurrent enhanced jobs would hold ~3.5GB of segmentation state at once.
_mask_gate = asyncio.Semaphore(1)

RAW_EXTENSIONS = {".cr3", ".cr2", ".arw", ".dng", ".nef", ".raf", ".orf", ".rw2"}
THUMB_QUALITY = 82


def _jpeg_quality() -> int:
    return int(settings.get("jpeg_quality") or 95)


def _thumb_edge() -> int:
    return int(settings.get("thumb_long_edge") or 480)


def _decode_full(raw_bytes: bytes, filename: str, no_auto_bright: bool = True) -> np.ndarray:
    """(H, W, 3) float32 in [0,1] at native resolution.

    no_auto_bright=True (the model's input) is a deliberately flat rendering
    so the model has something real to correct. The gallery's "Original"
    preview uses False instead -- the flat version looks misleadingly dark
    as a human before/after reference.
    """
    ext = os.path.splitext(filename.lower())[1]
    if ext in RAW_EXTENSIONS:
        with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
            tmp.write(raw_bytes)
            tmp.flush()
            with rawpy.imread(tmp.name) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=no_auto_bright,
                    output_bps=8,
                    output_color=rawpy.ColorSpace.sRGB,
                )
        img = Image.fromarray(rgb)
    else:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def _tensor_to_jpeg_bytes(t: torch.Tensor, quality: int) -> bytes:
    arr = (t.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return _array_to_jpeg_bytes(arr, quality)


def _array_to_jpeg_bytes(arr: np.ndarray, quality: int) -> bytes:
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _resize_tensor(t: torch.Tensor, long_edge: int) -> torch.Tensor:
    h, w = t.shape[2], t.shape[3]
    scale = long_edge / max(h, w)
    if scale >= 1:
        return t
    return torch.nn.functional.interpolate(
        t, size=(round(h * scale), round(w * scale)), mode="bilinear", align_corners=False)


def _style_list() -> list:
    styles = list(PRESETS)
    if settings.get("cinematic_presets"):
        styles += list(CINEMATIC)
    return styles


# ------------------------------------------------------------ enhanced mode

def _region_recipes(masks: dict) -> list:
    """Which region-aware looks are available given the masks we actually
    got. Each entry is (key, label, [(mask, params), ...])."""
    out = []
    subject = masks.get("subject")
    sky = masks.get("sky")

    if subject is not None:
        r = rg.SELECTIVE_COLOR
        out.append(("selective_color", "Selective Colour",
                    [(None, r["background"]), (subject, r["subject"])]))
        r = rg.SUBJECT_POP
        out.append(("subject_pop", "Subject Pop",
                    [(None, r["background"]), (subject, r["subject"])]))
    if sky is not None:
        r = rg.SKY_DRAMA
        out.append(("sky_drama", "Sky Drama",
                    [(None, r["ground"]), (sky, r["sky"])]))
    return out


def _compute_masks_sync(small_img: Image.Image) -> dict:
    """Runs on a worker thread. Masks are computed on the small preview --
    these models downsample internally anyway, so full resolution buys
    nothing but time."""
    import masking
    out = {}
    try:
        if settings.get("subject_masking"):
            m = masking.subject_mask(small_img)
            # Only offer subject recipes when something was actually found.
            # A near-empty or near-full mask means the model didn't isolate
            # anything, and grading through it would look like a bug.
            cov = float((m > 0.5).mean())
            if 0.01 < cov < 0.95:
                out["subject"] = masking.refine(m, threshold=0.5, feather_px=2.5)
        if settings.get("sky_masking"):
            m = masking.class_mask(small_img, "sky")
            cov = float((m > 0.5).mean())
            if 0.02 < cov < 0.98:
                out["sky"] = masking.refine(m, threshold=0.5, feather_px=8)
    except Exception as e:  # noqa: BLE001
        # Segmentation is an enhancement, not a requirement -- if it fails,
        # the import should still produce its normal global-LUT styles.
        logger.warning("masking failed, continuing without regions: %s", e)
    return out


# ------------------------------------------------------------ entry points

async def process_preview(raw_bytes: bytes, filename: str, source_type: str,
                          username: str, immich_asset_id: str | None = None) -> str:
    async with _pipeline_gate:
        arr = await asyncio.to_thread(_decode_full, raw_bytes, filename)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()

        baseline = await runtime.infer(tensor)
        edge = _thumb_edge()
        small_baseline = await asyncio.to_thread(_resize_tensor, baseline, edge)

        import_id = storage.create_import(filename, source_type, immich_asset_id, owner=username)

        if source_type == "upload":
            await asyncio.to_thread(storage.save_source_upload, import_id, raw_bytes, filename)

        normal_arr = await asyncio.to_thread(_decode_full, raw_bytes, filename, False)
        normal_t = torch.from_numpy(normal_arr).permute(2, 0, 1).unsqueeze(0).contiguous()
        small_normal = await asyncio.to_thread(_resize_tensor, normal_t, edge)
        orig_bytes = await asyncio.to_thread(_tensor_to_jpeg_bytes, small_normal, THUMB_QUALITY)
        await asyncio.to_thread(storage.save_original_thumb, import_id, orig_bytes)

        # Global styles (both modes)
        for key, label, params in _style_list():
            styled = await runtime.render_style(small_baseline, params)
            b = await asyncio.to_thread(_tensor_to_jpeg_bytes, styled, THUMB_QUALITY)
            await asyncio.to_thread(storage.save_preview, import_id, key, label, b)

        # Region-aware styles (enhanced mode only)
        if settings.get("mode") == "enhanced":
            small_pil = Image.fromarray(
                (small_baseline.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            async with _mask_gate:
                masks = await asyncio.to_thread(_compute_masks_sync, small_pil)
            base_small = small_baseline.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
            for key, label, regions in _region_recipes(masks):
                out = await asyncio.to_thread(rg.region_grade, base_small, regions)
                b = await asyncio.to_thread(_array_to_jpeg_bytes, out, THUMB_QUALITY)
                await asyncio.to_thread(storage.save_preview, import_id, key, label, b)

            # Composition suggestions ride along on the subject mask we just
            # computed, so they cost essentially nothing extra.
            h, w = base_small.shape[:2]
            crops = await asyncio.to_thread(
                cropping.suggest_crops, w, h, masks.get("subject"))
            await asyncio.to_thread(storage.save_crops, import_id, crops, w, h)

            # Render a small preview per crop so the picker shows the actual
            # framing. Without this the user selects a crop and nothing on
            # screen changes, so there's no way to judge it before
            # downloading. Cheap: these are crops of an already-small image.
            for c in crops:
                thumb = cropping.apply_crop(base_small, c)
                b = await asyncio.to_thread(_array_to_jpeg_bytes, thumb, THUMB_QUALITY)
                await asyncio.to_thread(storage.save_crop_thumb, import_id, c["key"], b)

    logger.info("preview: %s -> %s (user=%s, mode=%s)", filename, import_id,
                username, settings.get("mode"))
    return import_id


async def _get_original_bytes(import_id: str) -> tuple[bytes, str]:
    meta = storage.get_import(import_id)
    if meta["source_type"] == "upload":
        return await asyncio.to_thread(storage.get_source_upload_bytes, import_id)
    data, name = await immich_client.get_original_bytes(
        meta.get("owner", ""), meta["immich_asset_id"])
    return data, meta["source_name"] or name


async def render_full_style(import_id: str, style_key: str, crop_key: str | None = None) -> str:
    """Full-resolution render of one style, on first request, then cached.
    `crop_key` optionally applies one of the stored composition suggestions."""
    cache_key = style_key if not crop_key else f"{style_key}__{crop_key}"
    if storage.full_render_exists(import_id, cache_key):
        return storage.full_render_path(import_id, cache_key)

    global_styles = {k: p for k, _l, p in _style_list()}

    async with _pipeline_gate:
        if storage.full_render_exists(import_id, cache_key):
            return storage.full_render_path(import_id, cache_key)

        raw_bytes, filename = await _get_original_bytes(import_id)
        arr = await asyncio.to_thread(_decode_full, raw_bytes, filename)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
        baseline = await runtime.infer(tensor)

        if style_key in global_styles:
            styled = await runtime.render_style(baseline, global_styles[style_key])
            jpeg = await asyncio.to_thread(_tensor_to_jpeg_bytes, styled, _jpeg_quality())
        else:
            # Region recipe: masks must be recomputed at this resolution.
            base_arr = baseline.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
            pil = Image.fromarray((base_arr * 255).astype(np.uint8))
            async with _mask_gate:
                masks = await asyncio.to_thread(_compute_masks_sync, pil)
            recipe = next((r for k, _l, r in _region_recipes(masks) if k == style_key), None)
            if recipe is None:
                raise KeyError(f"unknown or unavailable style {style_key!r}")
            out = await asyncio.to_thread(rg.region_grade, base_arr, recipe)
            jpeg = await asyncio.to_thread(_array_to_jpeg_bytes, out, _jpeg_quality())

        if crop_key:
            meta = storage.get_import(import_id) or {}
            crop = next((c for c in meta.get("crops", []) if c["key"] == crop_key), None)
            if crop is None:
                raise KeyError(f"unknown crop {crop_key!r}")
            ref = meta.get("crop_ref", {})
            crop = dict(crop, ref_w=ref.get("w"), ref_h=ref.get("h"))
            full = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
            jpeg = await asyncio.to_thread(
                _array_to_jpeg_bytes, cropping.apply_crop(full, crop), _jpeg_quality())

        await asyncio.to_thread(storage.save_full_render, import_id, cache_key, jpeg)

    logger.info("full render: %s / %s", import_id, cache_key)
    return storage.full_render_path(import_id, cache_key)
