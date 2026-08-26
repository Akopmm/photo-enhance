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
import contextlib
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
from model_runtime import runtime, return_arenas_to_os
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

# Full-resolution renders are the heavy path: measured ~3.5GB peak RSS each
# on optiplex, against ~2.5GB for an import. Three at once is ~10.5GB, which
# on a 15GB box already running Immich/Jellyfin/Postgres leaves no headroom
# -- that is how a burst of downloads took the host to its ceiling. This gate
# is acquired BEFORE _pipeline_gate so a waiting render doesn't sit on a
# pipeline slot while it queues.
_render_gate = asyncio.Semaphore(settings.get("max_concurrent_renders") or 2)

# Admission control, held by the HTTP handler across the WHOLE request --
# fetch included. _pipeline_gate starts too late: the Immich original (~32MB
# for a CR3) is downloaded *before* process_preview is called, so 30 clicks
# parked ~980MB of raw bytes in queued requests plus 30 simultaneous
# downloads, none of it bounded. Measured: 30 concurrent imports peaked at
# 11.7GB, of which ~1GB was raw bytes that had not yet reached the gate.
_admission_gate = asyncio.Semaphore(settings.get("max_concurrent_jobs") or 3)


@contextlib.asynccontextmanager
async def admission():
    """Wrap fetch + process so a queued request holds no payload."""
    async with _admission_gate:
        yield

RAW_EXTENSIONS = {".cr3", ".cr2", ".arw", ".dng", ".nef", ".raf", ".orf", ".rw2"}
THUMB_QUALITY = 82


def _jpeg_quality() -> int:
    return int(settings.get("jpeg_quality") or 95)


def _thumb_edge() -> int:
    return int(settings.get("thumb_long_edge") or 480)


# The editor shows one big preview, and 480px upscaled to a full-width hero
# looks soft. The stored baseline is kept larger than the filmstrip
# thumbnails so the hero can be re-rendered from it at any look/strength and
# still be sharp. ~250KB per import, and rendering a look over it is well
# under the time it takes to drag a slider.
HERO_EDGE = 1280

# Output size presets. The long edge is what actually matters -- denoise cost
# scales with pixel count, so halving the edge quarters the work, and the
# resize happens right after the colour model so denoise, styling and
# cropping ALL run smaller, not just the final encode.
#
# Measured on a 26MP CR3 with GPU denoise: full frame 140 tiles / 9.1 min,
# against 48 tiles / 3.1 min at "large". "Original" stays the default -- the
# whole point of the service was keeping full-resolution quality.
SIZE_PRESETS = {
    "original": {"edge": None, "target_mb": None, "label": "Original"},
    "large":    {"edge": 4200, "target_mb": 5.0,  "label": "Large (~5 MB)"},
    "medium":   {"edge": 3200, "target_mb": 3.0,  "label": "Medium (~3 MB)"},
    "small":    {"edge": 1800, "target_mb": 1.0,  "label": "Small (~1 MB)"},
}


class RenderCancelled(Exception):
    """Raised when the import a render belongs to is deleted underneath it.

    Cooperative rather than task.cancel(): the render can then stop at a
    clean boundary instead of part-way through writing a file into a
    directory that no longer exists."""


def _resolve_crop(import_id: str, crop_key: str | None, rect: dict | None) -> dict | None:
    """Any crop -- dragged rectangle or stored preset -- as fractions of the
    frame, so it can be applied before anything expensive runs."""
    if rect:
        return rect
    if not crop_key:
        return None
    meta = storage.get_import(import_id) or {}
    c = next((c for c in meta.get("crops", []) if c["key"] == crop_key), None)
    if not c:
        raise KeyError(f"unknown crop {crop_key!r}")
    ref = meta.get("crop_ref") or {}
    rw, rh = ref.get("w"), ref.get("h")
    if not rw or not rh:
        return None
    return {"x": c["x"] / rw, "y": c["y"] / rh, "w": c["w"] / rw, "h": c["h"] / rh}


def _crop_tensor(t: torch.Tensor, r: dict) -> torch.Tensor:
    h, w = t.shape[2], t.shape[3]
    x0, y0 = int(round(r["x"] * w)), int(round(r["y"] * h))
    x1, y1 = min(w, x0 + int(round(r["w"] * w))), min(h, y0 + int(round(r["h"] * h)))
    return t[:, :, y0:y1, x0:x1]


def _crop_array(a: np.ndarray, r: dict) -> np.ndarray:
    h, w = a.shape[:2]
    x0, y0 = int(round(r["x"] * w)), int(round(r["y"] * h))
    x1, y1 = min(w, x0 + int(round(r["w"] * w))), min(h, y0 + int(round(r["h"] * h)))
    return a[y0:y1, x0:x1]


def _encode_to_target(arr: np.ndarray, target_mb: float | None) -> bytes:
    """Encode, and if a size was asked for, step quality down until it fits.

    Sizes are labelled in MB, so they should mean something: a denoised frame
    compresses far better than a noisy one, and content varies wildly, so a
    fixed quality would make the label a guess. Three encodes at most.
    """
    q = _jpeg_quality()
    data = _array_to_jpeg_bytes(arr, q)
    if not target_mb:
        return data
    limit = target_mb * 1e6
    for q in (88, 82, 76):
        if len(data) <= limit:
            break
        data = _array_to_jpeg_bytes(arr, q)
    return data


def _decode_full(raw_bytes: bytes, filename: str, no_auto_bright: bool = True,
                 half_size: bool = False) -> np.ndarray:
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
                    half_size=half_size,
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

# Long edge at which masks are computed and stored. Mirrors
# masking.MASK_RESOLUTION, duplicated only so classic mode never has to
# import masking (and therefore transformers) just to read a constant.
MASK_SRC_EDGE = 1024

MASK_NAMES = ("subject", "sky", "depth", "foliage")

# Feather radius per mask, as a FRACTION OF THE LONG EDGE. These were pixel
# constants shared between the 480px preview and the 6000px render, which
# made the downloaded edges ~12.5x harder than the ones actually approved in
# the gallery -- the exact artefact feather() exists to prevent.
_FEATHER = {"subject": 0.002, "sky": 0.012, "foliage": 0.008}

# How far to pull each matte inward before feathering. Segmentation mattes
# run wide, and the overhang keeps the region's grade on background pixels --
# the glow around the subject in Selective Colour. Subject mattes are the
# crispest and take the most choke; sky is graded softly so it needs least.
_CHOKE = {"subject": 0.45, "sky": 0.20, "foliage": 0.30}

# Depth grading does nothing on a flat scene, so the depth looks are offered
# only when the depth map actually spans a range.
MIN_DEPTH_SPREAD = 0.25

# Depth bands, as percentages of the frame: the nearest 30% and the farthest
# 40% of pixels, leaving an ungraded middle so the transition reads as depth
# rather than as two flat layers.
NEAR_BAND_PCT = 30
FAR_BAND_PCT = 40

# Aerial perspective is an outdoor phenomenon -- hazing the "far" field of an
# indoor scene just looks like a lifted black point. Offered only when the
# frame actually contains open-air content.
_OUTDOOR_CLASSES = ("sky", "mountain", "tree", "grass", "water", "sea", "building")


def _should_denoise(arr: np.ndarray) -> tuple[bool, float]:
    """(do it?, measured sigma). Cheap enough to always measure, so `auto`
    can decide per photo rather than making the user judge noise by eye."""
    mode = (settings.get("denoise_mode") or "auto").lower()
    if mode == "off":
        return False, 0.0
    import denoise as dn
    if not dn.available():
        return False, 0.0
    sigma = dn.estimate_sigma(arr)
    if mode == "always":
        return True, sigma
    # Offer it whenever there is anything to work with -- the slider and the
    # preview blend are free. It is the DEFAULT AMOUNT that gets scaled by
    # how noisy the photo really is (see _default_denoise_amount), because
    # that is what costs minutes on a full-resolution download.
    return sigma >= float(settings.get("denoise_threshold") or 3.0), sigma


def _default_denoise_amount(sigma: float) -> float:
    """0 for a mildly noisy frame, ramping to the configured maximum by the
    time it is genuinely bad.

    A hard threshold defaulted a bright daylight portrait at sigma 4.4 to 90%
    denoise, which bought little and cost ~126 tiles (5-8 minutes) on
    download. The slider still goes to 100% if you want it; it just is not
    the default for a photo that does not need it.
    """
    lo = float(settings.get("denoise_threshold") or 3.0) + 2.0
    hi = lo + 3.0
    top = float(settings.get("denoise_amount") or 0.9)
    if sigma <= lo:
        return 0.0
    return round(min(top, top * (sigma - lo) / (hi - lo)), 2)


def _denoise_full(arr: np.ndarray, progress=None) -> np.ndarray:
    """Fully denoised (amount = 1). Callers blend toward the original
    themselves, so the amount is never baked in."""
    import denoise as dn
    return dn.denoise(arr, progress=progress)


def _blend_denoise(original: np.ndarray, denoised: np.ndarray, amount: float) -> np.ndarray:
    import denoise as dn
    return dn.blend(original, denoised, amount)


def _tensor_to_pil(t: torch.Tensor, long_edge: int | None = None) -> Image.Image:
    if long_edge:
        t = _resize_tensor(t, long_edge)
    arr = (t.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _save_masks(import_id: str, masks: dict):
    import masking
    for name, m in masks.items():
        storage.save_mask(import_id, name, masking.encode(m))


def _load_masks(import_id: str) -> dict:
    import masking
    out = {}
    for name in MASK_NAMES:
        data = storage.load_mask(import_id, name)
        if data is not None:
            out[name] = masking.decode(data)
    return out


# Which prepared masks each recipe actually consumes. render_full_style
# renders exactly one style, so preparing all five at 6000x4000 (~96MB each,
# plus depth-band transients) burned over a gigabyte on masks the chosen
# recipe never touched. That doubled peak RSS per job, and at
# max_concurrent_jobs=3 it took the box to its 15GB ceiling.
_RECIPE_MASKS = {
    "selective_color": ("subject",),
    "subject_pop": ("subject",),
    "sky_drama": ("sky",),
    "depth_pop": ("near", "far"),
    "depth_haze": ("near", "far"),
    "foliage_lift": ("foliage",),
}

# Prepared-mask name -> the stored mask it derives from.
_MASK_SOURCE = {"near": "depth", "far": "depth"}


def _prepare_masks(stored: dict, size: tuple[int, int], only: tuple | None = None,
                   crop: dict | None = None) -> dict:
    """Resample stored masks to (w, h) and feather relative to THAT size.

    Storing raw alpha and feathering per-target is what keeps the preview and
    the full render visually identical -- the mask itself is resolution
    independent, only its softening is not.

    `only` restricts the work to the prepared masks named, which matters at
    full resolution where each one is ~96MB. The preview path leaves it None
    because it needs every recipe and its masks are 480px anyway.
    """
    import masking
    if only is not None:
        wanted = {_MASK_SOURCE.get(n, n) for n in only}
        stored = {k: v for k, v in stored.items() if k in wanted}
    out = {}
    for key, mask in stored.items():
        if crop:
            # Crop the stored (full-frame) mask by the same fractions as the
            # image, otherwise a cropped render would be graded through a
            # mask belonging to the whole frame.
            mask = _crop_array(mask, crop)
        m = masking.resample(mask, size)
        if key == "depth":
            # Bands are PERCENTILES of this photo's own depth distribution,
            # not fixed thresholds. Fixed cuts collapse on scenes that aren't
            # evenly distributed in depth -- on a bird against distant
            # foliage they put 95% of the frame in "far", which turns a
            # depth grade into a global one. Percentiles keep both bands
            # meaningful whatever the scene.
            # Percentiles come from the STORED map, not the resampled one,
            # so every output size cuts the bands at exactly the same depth.
            near_edge = float(np.percentile(mask, 100 - NEAR_BAND_PCT))
            far_edge = float(np.percentile(mask, FAR_BAND_PCT))
            if only is None or "near" in only:
                out["near"] = masking.depth_band(m, near_edge, 1.0)
            if only is None or "far" in only:
                out["far"] = masking.depth_band(m, 0.0, far_edge)
            del m
        else:
            out[key] = masking.refine(m, feather_frac=_FEATHER.get(key, 0.006),
                                      choke_amount=_CHOKE.get(key, 0.0))
    return out


def _region_recipes(masks: dict, scene: dict | None = None) -> list:
    """Which region-aware looks are available given the masks we actually
    got. Each entry is (key, label, [(mask, params), ...])."""
    out = []
    subject = masks.get("subject")
    sky = masks.get("sky")
    near, far = masks.get("near"), masks.get("far")
    foliage = masks.get("foliage")

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
    if near is not None and far is not None:
        r = rg.DEPTH_POP
        out.append(("depth_pop", "Depth Pop",
                    [(far, r["far"]), (near, r["near"])]))
        outdoor = sum((scene or {}).get(k, 0) for k in _OUTDOOR_CLASSES)
        if outdoor >= 0.15:
            r = rg.DEPTH_HAZE
            out.append(("depth_haze", "Aerial Depth",
                        [(near, r["near"]), (far, r["far"])]))
    if foliage is not None:
        r = rg.FOLIAGE_LIFT
        out.append(("foliage_lift", "Foliage",
                    [(None, r["base"]), (foliage, r["foliage"])]))
    return out


def _compute_masks_sync(img: Image.Image) -> tuple[dict, dict]:
    """Runs on a worker thread. Returns (masks, scene).

    Masks come back as raw soft alpha at masking.MASK_RESOLUTION and
    UNFEATHERED -- they are persisted in that form and softened per-target by
    _prepare_masks, so one computation serves every output size.
    """
    import numpy as _np

    import masking
    masks, scene = {}, {}
    try:
        if settings.get("subject_masking"):
            m = masking.subject_mask(img)
            # is_usable() weighs confidence as well as area. The previous
            # check was area-only with a 1% floor, which rejected exactly the
            # photos these looks are for -- a bird on a wire covers 0.9% of
            # the frame and was silently getting no subject recipes at all.
            if masking.is_usable(m):
                masks["subject"] = m
        if settings.get("sky_masking"):
            # scene_classes() was implemented and never called, so recipe
            # selection was blind to what was actually in the frame. The
            # SegFormer pass happens anyway for the sky mask, so this is free.
            scene = masking.scene_classes(img)
            if scene.get("sky", 0) >= 0.02:
                m = masking.class_mask(img, "sky")
                if masking.is_usable(m, min_coverage=0.02, min_confidence=0.5):
                    masks["sky"] = m
            greenery = sum(scene.get(k, 0) for k in ("tree", "grass", "plant"))
            if greenery >= 0.12:
                m = masking.class_mask_any(img, ("tree", "grass", "plant"))
                if masking.is_usable(m, min_coverage=0.05, min_confidence=0.5):
                    masks["foliage"] = m
        if settings.get("depth_masking"):
            d = masking.depth_map(img)
            spread = float(_np.percentile(d, 90) - _np.percentile(d, 10))
            if spread >= MIN_DEPTH_SPREAD:
                masks["depth"] = d
    except Exception as e:  # noqa: BLE001
        # Segmentation is an enhancement, not a requirement -- if it fails,
        # the import should still produce its normal global-LUT styles.
        logger.warning("masking failed, continuing without regions: %s", e)
    return masks, scene


async def _masks_for(import_id: str, baseline: torch.Tensor) -> tuple[dict, dict]:
    """Stored masks for an import, computing and persisting them if absent.

    The absent case covers imports made before masks were persisted; without
    it those would silently lose their region styles.
    """
    stored = await asyncio.to_thread(_load_masks, import_id)
    if stored:
        meta = storage.get_import(import_id) or {}
        return stored, meta.get("scene", {})
    mask_pil = await asyncio.to_thread(_tensor_to_pil, baseline, MASK_SRC_EDGE)
    async with _mask_gate:
        stored, scene = await asyncio.to_thread(_compute_masks_sync, mask_pil)
    await asyncio.to_thread(_save_masks, import_id, stored)
    return stored, scene


# ------------------------------------------------------------ entry points

async def process_preview(raw_bytes: bytes, filename: str, source_type: str,
                          username: str, immich_asset_id: str | None = None) -> str:
    async with _pipeline_gate:
        arr = await asyncio.to_thread(_decode_full, raw_bytes, filename)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()

        baseline = await runtime.infer(tensor)
        arr = tensor = None  # ~576MB of decode buffers, dead once inferred
        edge = _thumb_edge()

        # Denoise ONCE here, on the baseline, so every look, thumbnail and
        # hero inherits it without paying again. Done at HERO_EDGE rather
        # than native resolution: this model has no internal downsampling, so
        # full-res would be minutes. The full-resolution render redoes it
        # tiled, because a denoised preview and a noisy download would be the
        # same divergence bug we just removed from masking.
        hero_for_noise = await asyncio.to_thread(_resize_tensor, baseline, HERO_EDGE)
        probe = hero_for_noise.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
        do_dn, sigma = await asyncio.to_thread(_should_denoise, probe)
        default_amount = _default_denoise_amount(sigma)
        denoise_meta = {"available": bool(do_dn), "sigma": round(sigma, 2),
                        "amount": default_amount}
        clean = None
        if do_dn:
            logger.info("denoising %s (sigma %.2f)", filename, sigma)
            clean = await asyncio.to_thread(_denoise_full, probe)
        probe = None
        small_baseline = await asyncio.to_thread(_resize_tensor, baseline, edge)

        import_id = storage.create_import(filename, source_type, immich_asset_id, owner=username)
        await asyncio.to_thread(storage.save_denoise_info, import_id, denoise_meta)

        if source_type == "upload":
            await asyncio.to_thread(storage.save_source_upload, import_id, raw_bytes, filename)

        # half_size: this decode exists only to make a ~480px "Original"
        # thumbnail, so a full 6000px decode was ~430MB of peak RSS and ~0.4s
        # spent to throw 97% of the pixels away immediately.
        normal_arr = await asyncio.to_thread(_decode_full, raw_bytes, filename, False, True)
        normal_t = torch.from_numpy(normal_arr).permute(2, 0, 1).unsqueeze(0).contiguous()
        normal_arr = None
        small_normal = await asyncio.to_thread(_resize_tensor, normal_t, edge)
        normal_t = None
        orig_bytes = await asyncio.to_thread(_tensor_to_jpeg_bytes, small_normal, THUMB_QUALITY)
        await asyncio.to_thread(storage.save_original_thumb, import_id, orig_bytes)

        # Store BOTH baselines: the untouched one, and (when this photo was
        # noisy enough to bother) the fully denoised one. Any denoise amount
        # is then a blend between the two, so the slider costs nothing.
        baseline_bytes = await asyncio.to_thread(
            _tensor_to_jpeg_bytes, hero_for_noise, THUMB_QUALITY)
        await asyncio.to_thread(storage.save_baseline_thumb, import_id, baseline_bytes)
        if clean is not None:
            dn_bytes = await asyncio.to_thread(_array_to_jpeg_bytes, clean, THUMB_QUALITY)
            await asyncio.to_thread(storage.save_denoised_thumb, import_id, dn_bytes)
            # The filmstrip thumbnails are rendered from the default amount;
            # they exist to pick a LOOK, and noise is invisible at 104px.
            base_np = hero_for_noise.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
            merged = await asyncio.to_thread(_blend_denoise, base_np, clean, default_amount)
            baseline = torch.from_numpy(merged).permute(2, 0, 1).unsqueeze(0).contiguous()
            small_baseline = await asyncio.to_thread(_resize_tensor, baseline, edge)
            base_np = merged = None
        clean = hero_for_noise = None

        # Global styles (both modes)
        for key, label, params in _style_list():
            styled = await runtime.render_style(small_baseline, params)
            b = await asyncio.to_thread(_tensor_to_jpeg_bytes, styled, THUMB_QUALITY)
            await asyncio.to_thread(storage.save_preview, import_id, key, label, b)

        # Region-aware styles (enhanced mode only)
        if settings.get("mode") == "enhanced":
            # Masks are computed once, at MASK_SRC_EDGE, and persisted. The
            # full-resolution render reuses these exact masks rather than
            # segmenting again, so the download grades through what the
            # gallery was judged on.
            mask_pil = await asyncio.to_thread(_tensor_to_pil, baseline, MASK_SRC_EDGE)
            async with _mask_gate:
                stored_masks, scene = await asyncio.to_thread(_compute_masks_sync, mask_pil)
            await asyncio.to_thread(_save_masks, import_id, stored_masks)

            base_small = small_baseline.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
            h, w = base_small.shape[:2]
            masks = _prepare_masks(stored_masks, (w, h))
            recipes = _region_recipes(masks, scene)
            await asyncio.to_thread(storage.save_scene, import_id, scene,
                                    [k for k, _l, _r in recipes])
            for key, label, regions in recipes:
                out = await asyncio.to_thread(rg.region_grade, base_small, regions)
                b = await asyncio.to_thread(_array_to_jpeg_bytes, out, THUMB_QUALITY)
                await asyncio.to_thread(storage.save_preview, import_id, key, label, b)

            # Composition suggestions ride along on the subject mask we just
            # computed, so they cost essentially nothing extra.
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

    # Hand the freed arenas back before the next job starts, so a burst of
    # imports doesn't ratchet RSS upward for the whole batch.
    await asyncio.to_thread(return_arenas_to_os)
    logger.info("preview: %s -> %s (user=%s, mode=%s)", filename, import_id,
                username, settings.get("mode"))
    return import_id


async def render_preview_at_strength(import_id: str, style_key: str, strength: float,
                                     denoise_amount: float | None = None) -> bytes:
    """Re-render ONE region preview at a given strength, from the stored
    baseline thumbnail and masks. No RAW decode, no model -- it is a 480px
    composite, so the slider can drive it interactively.

    Returns JPEG bytes, or None if this import predates the stored baseline
    (older imports keep their fixed 100% previews).
    """
    if not storage.has_baseline_thumb(import_id):
        return None
    meta = storage.get_import(import_id) or {}
    global_styles = {k: p for k, _l, p in _style_list()}
    stored = None
    if style_key not in global_styles:
        stored = await asyncio.to_thread(_load_masks, import_id)
        if not stored:
            return None

    def _render():
        img = Image.open(storage.baseline_thumb_path(import_id)).convert("RGB")
        base = np.asarray(img).astype(np.float32) / 255.0
        # The denoise slider is a blend between the two stored baselines --
        # no model runs, so it is as responsive as the strength slider.
        dn_meta = meta.get("denoise") or {}
        amt = dn_meta.get("amount", 0.9) if denoise_amount is None else denoise_amount
        if storage.has_denoised_thumb(import_id):
            clean = np.asarray(Image.open(storage.denoised_thumb_path(import_id))
                               .convert("RGB")).astype(np.float32) / 255.0
            if clean.shape == base.shape:
                base = _blend_denoise(base, clean, float(amt))
        if style_key in global_styles:
            # A global style is just a recipe with one whole-frame layer, so
            # region_grade's strength blending covers it too -- which is why
            # the slider can apply to every look instead of appearing only
            # for the region ones.
            recipe = [(None, global_styles[style_key])]
        else:
            h, w = base.shape[:2]
            masks = _prepare_masks(stored, (w, h), only=_RECIPE_MASKS.get(style_key))
            recipe = next((r for k, _l, r in _region_recipes(masks, meta.get("scene", {}))
                           if k == style_key), None)
            if recipe is None:
                return None
        out = rg.region_grade(base, recipe, min(max(strength, 0.0), 1.0))
        return _array_to_jpeg_bytes(out, THUMB_QUALITY)

    return await asyncio.to_thread(_render)


async def _get_original_bytes(import_id: str) -> tuple[bytes, str]:
    meta = storage.get_import(import_id)
    if meta["source_type"] == "upload":
        return await asyncio.to_thread(storage.get_source_upload_bytes, import_id)
    data, name = await immich_client.get_original_bytes(
        meta.get("owner", ""), meta["immich_asset_id"])
    return data, meta["source_name"] or name


def _parse_crop_rect(spec: str | None) -> dict | None:
    """`x,y,w,h` as fractions of the frame -> a crop dict in full-image
    pixels at render time. Lets the editor send a rectangle the user dragged
    instead of only one of the suggested presets."""
    if not spec:
        return None
    try:
        x, y, w, h = (float(v) for v in spec.split(","))
    except (ValueError, AttributeError):
        raise KeyError(f"bad crop_rect {spec!r}")
    x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
    w, h = min(max(w, 0.01), 1.0 - x), min(max(h, 0.01), 1.0 - y)
    return {"x": x, "y": y, "w": w, "h": h}


async def render_full_style(import_id: str, style_key: str, crop_key: str | None = None,
                            strength: float = 1.0, progress=None,
                            crop_rect: str | None = None,
                            denoise_amount: float | None = None,
                            size: str = "original", cancelled=None) -> str:
    """Full-resolution render of one style, on first request, then cached.

    `crop_key` optionally applies one of the stored composition suggestions.
    `strength` (region recipes only) dials the effect back toward the
    ungraded image; it is part of the cache key, so each setting is rendered
    once and then reused.
    """
    # `progress(pct, message)` reports real stages rather than a spinner --
    # a full-resolution render is ~12s and silence for that long reads as a
    # hang. Stages are coarse but each one is a genuine boundary in the work.
    def _say(pct, msg):
        # Also the cancellation checkpoint: called at every stage boundary
        # and once per denoise tile, which is the granularity that matters
        # when the expensive stage is minutes long.
        if cancelled and cancelled():
            raise RenderCancelled(import_id)
        if progress:
            progress(pct, msg)

    strength = min(max(float(strength), 0.0), 1.0)
    _size = SIZE_PRESETS.get(size or "original", SIZE_PRESETS["original"])
    rect = _parse_crop_rect(crop_rect)
    if rect:
        # A dragged rectangle gets its own cache entry, keyed by the numbers
        # so re-downloading the same framing is instant.
        tag = "r" + "_".join(f"{rect[k]:.4f}".replace("0.", "") for k in ("x", "y", "w", "h"))
        cache_key = f"{style_key}__{tag}"
        crop_key = None
    else:
        cache_key = style_key if not crop_key else f"{style_key}__{crop_key}"
    if strength < 1.0:
        cache_key = f"{cache_key}__s{round(strength * 100):03d}"
    _dn_meta = (storage.get_import(import_id) or {}).get("denoise") or {}
    _dn_amt = _dn_meta.get("amount", 0.9) if denoise_amount is None else denoise_amount
    if _dn_meta.get("available") and _dn_amt:
        cache_key = f"{cache_key}__d{round(float(_dn_amt) * 100):03d}"
    if _size["edge"]:
        cache_key = f"{cache_key}__{size}"
    if storage.full_render_exists(import_id, cache_key):
        _say(100, "Ready")
        return storage.full_render_path(import_id, cache_key)

    global_styles = {k: p for k, _l, p in _style_list()}

    # Stage weights differ hugely depending on whether denoising runs: on a
    # noisy photo the denoiser is most of the wall clock, so the bar has to
    # give it most of the range or it crawls 50->78 for eight minutes and
    # then jumps. Without denoise, the earlier stages get the room instead.
    _dn_heavy = bool(_dn_meta.get("available")) and float(_dn_amt or 0) > 0
    def _stage(pct_light, pct_heavy):
        return pct_heavy if _dn_heavy else pct_light

    _say(2, "Waiting for a slot")
    async with _render_gate, _pipeline_gate:
        if storage.full_render_exists(import_id, cache_key):
            _say(100, "Ready")
            return storage.full_render_path(import_id, cache_key)

        _say(_stage(10, 4), "Fetching the original")
        raw_bytes, filename = await _get_original_bytes(import_id)
        _say(_stage(25, 8), "Decoding the RAW")
        arr = await asyncio.to_thread(_decode_full, raw_bytes, filename)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
        _say(_stage(45, 12), "Colour model")
        baseline = await runtime.infer(tensor)
        arr = tensor = None

        # Crop FIRST, before denoise, styling and sizing. It used to happen
        # last, on the finished JPEG: that denoised and styled the whole frame
        # only to discard most of it on a 2.39:1 crop, and -- worse -- decoded
        # and re-encoded the JPEG, so every cropped download was compressed
        # twice.
        norm_crop = _resolve_crop(import_id, crop_key, rect)
        if norm_crop:
            _say(_stage(48, 13), "Cropping")
            baseline = _crop_tensor(baseline, norm_crop)

        if _size["edge"]:
            want = min(_size["edge"], max(baseline.shape[2], baseline.shape[3]))
            baseline = await asyncio.to_thread(_resize_tensor, baseline, want)
            _say(_stage(50, 14), f"Sizing to {_size['label']}")

        if _dn_meta.get("available") and float(_dn_amt) > 0:
            _say(14, "Removing noise")
            base_np = baseline.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()

            def _tile_progress(done, total):
                # Plain language, and a range that matches the real cost:
                # this stage is minutes, everything else is seconds.
                _say(14 + int(70 * done / max(total, 1)),
                     f"Removing noise ({round(100 * done / max(total, 1))}% of the frame)")

            clean = await asyncio.to_thread(_denoise_full, base_np, _tile_progress)
            merged = await asyncio.to_thread(_blend_denoise, base_np, clean, float(_dn_amt))
            baseline = torch.from_numpy(merged).permute(2, 0, 1).unsqueeze(0).contiguous()
            base_np = clean = merged = None

        _say(_stage(60, 88), "Applying the look")
        if style_key in global_styles:
            styled = await runtime.render_style(baseline, global_styles[style_key])
            if strength < 1.0:
                # Blend back toward the uncorrected baseline, matching what
                # the preview showed at this strength.
                styled = baseline + (styled - baseline) * strength
            arr_out = styled.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
            jpeg = await asyncio.to_thread(_encode_to_target, arr_out, _size["target_mb"])
            arr_out = None
        else:
            # Region recipe. The masks are NOT recomputed here: they are
            # loaded and resampled, which is both cheaper and the only way
            # the download can match the preview. Recomputing at native
            # resolution used to shift coverage (measured 0.2120 -> 0.1873
            # sky on one photo) and could even drop a style entirely, 500ing
            # on something the gallery was still offering.
            base_arr = baseline.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
            h, w = base_arr.shape[:2]
            stored_masks, scene = await _masks_for(import_id, baseline)
            masks = _prepare_masks(stored_masks, (w, h), only=_RECIPE_MASKS.get(style_key),
                                   crop=norm_crop)
            stored_masks = None  # 1024px originals are dead weight from here
            recipe = next((r for k, _l, r in _region_recipes(masks, scene) if k == style_key), None)
            if recipe is None:
                raise KeyError(f"unknown or unavailable style {style_key!r}")
            out = await asyncio.to_thread(rg.region_grade, base_arr, recipe, strength)
            jpeg = await asyncio.to_thread(_encode_to_target, out, _size["target_mb"])

        _say(_stage(94, 96), "Saving")
        await asyncio.to_thread(storage.save_full_render, import_id, cache_key, jpeg)

    await asyncio.to_thread(return_arenas_to_os)
    _say(100, "Ready")
    logger.info("full render: %s / %s", import_id, cache_key)
    return storage.full_render_path(import_id, cache_key)
