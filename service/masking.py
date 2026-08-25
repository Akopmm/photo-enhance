"""Semantic mask generation for region-aware photo grading.

Four complementary mask sources, because no single model does all of it well:

  subject  -- BiRefNet-lite (44M params, MIT). Purpose-built dichotomous
              segmentation; gives crisp foreground cutouts. This is the
              "Select Subject" equivalent and the best of the three.
  sky      -- SegFormer trained on ADE20K, which has a dedicated `sky`
              class. Also exposes other useful scene classes for free.
  depth    -- Depth Anything V2 Small (24.8M, Apache-2.0). Lightroom's
              "Depth Range Mask": grade by distance instead of by object,
              which needs no foreground/background decision at all.
  prompt   -- CLIPSeg, arbitrary text ("car", "the red jacket"). Most
              flexible, but noticeably coarser/softer than the others --
              use it for broad regions, not crisp cutouts.

MASK_RESOLUTION and why masks are computed exactly once
-------------------------------------------------------
Every mask is produced at MASK_RESOLUTION and resampled to whatever the
caller needs. That is not an approximation: BiRefNet resizes its input to
1024x1024 internally, so a mask "computed at 6000px" is a 1024x1024
prediction upscaled either way -- feeding it the full image only changes
which library does the downsampling. SegFormer is coarser still (it emits
h/4 x w/4 from a 512x512 processor input).

This matters because the preview and the final render used to segment
independently, at 480px and at native resolution, and quietly disagreed --
measured 0.2120 vs 0.1873 sky coverage on the same photo. The gallery
showed one grade and the download produced another. Computing once and
resampling makes them identical by construction.
"""
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter

_cache = {}

# Long edge at which every mask is computed. 1024 is BiRefNet's own working
# resolution, so this is its native precision rather than a compromise.
MASK_RESOLUTION = 1024


def _device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def working_size(size: tuple[int, int]) -> tuple[int, int]:
    """The (w, h) a mask should be computed at for an image of this size."""
    w, h = size
    scale = MASK_RESOLUTION / max(w, h)
    if scale >= 1:
        return w, h
    return max(1, round(w * scale)), max(1, round(h * scale))


def _at_working_size(img: Image.Image) -> Image.Image:
    target = working_size(img.size)
    return img if target == img.size else img.resize(target, Image.LANCZOS)


# ---------------- subject (BiRefNet) ----------------

def _birefnet():
    if "birefnet" not in _cache:
        from transformers import AutoModelForImageSegmentation
        m = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet_lite", trust_remote_code=True)
        m.eval().to(_device())
        _cache["birefnet"] = m
    return _cache["birefnet"]


@torch.no_grad()
def subject_mask(img: Image.Image) -> np.ndarray:
    """Soft alpha, float32 (H, W) in [0,1] at MASK_RESOLUTION -- 1 = subject.

    Deliberately NOT thresholded. The graded alpha at hair, fur and feather
    edges is the whole reason to run BiRefNet rather than something cheap,
    and a hard cut is the one operation that destroys it.
    """
    m = _birefnet()
    small = _at_working_size(img.convert("RGB"))
    tf = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(small).unsqueeze(0).to(_device())
    pred = m(x)[-1].sigmoid().cpu()[0].squeeze().numpy()
    return _resize_mask(pred, small.size)


# ---------------- semantic classes (SegFormer / ADE20K) ----------------

# ADE20K label ids we care about. The full set has 150 classes; these are the
# ones that matter for photo grading.
ADE_CLASSES = {
    "sky": 2,
    "tree": 4,
    "road": 6,
    "grass": 9,
    "person": 12,
    "earth": 13,
    "water": 21,
    "car": 20,
    "building": 1,
    "mountain": 16,
    "plant": 17,
    "sea": 26,
}


def _segformer():
    if "segformer" not in _cache:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        name = "nvidia/segformer-b0-finetuned-ade-512-512"
        proc = SegformerImageProcessor.from_pretrained(name)
        model = SegformerForSemanticSegmentation.from_pretrained(name).eval().to(_device())
        _cache["segformer"] = (proc, model)
    return _cache["segformer"]


@torch.no_grad()
def _ade_labels(img: Image.Image) -> np.ndarray:
    proc, model = _segformer()
    inputs = proc(images=img.convert("RGB"), return_tensors="pt").to(_device())
    return model(**inputs).logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)


def class_mask(img: Image.Image, class_name: str) -> np.ndarray:
    """Mask for one ADE20K class, e.g. 'sky'. float32 (H, W) in [0,1]."""
    if class_name not in ADE_CLASSES:
        raise ValueError(f"unknown class {class_name!r}; known: {sorted(ADE_CLASSES)}")
    small = _at_working_size(img)
    labels = _ade_labels(small)
    return _resize_mask((labels == ADE_CLASSES[class_name]).astype(np.float32), small.size)


def class_mask_any(img: Image.Image, class_names) -> np.ndarray:
    """Union of several ADE20K classes from a single forward pass -- calling
    class_mask() once per class would re-run SegFormer each time."""
    unknown = [c for c in class_names if c not in ADE_CLASSES]
    if unknown:
        raise ValueError(f"unknown class(es) {unknown}; known: {sorted(ADE_CLASSES)}")
    small = _at_working_size(img)
    labels = _ade_labels(small)
    ids = [ADE_CLASSES[c] for c in class_names]
    return _resize_mask(np.isin(labels, ids).astype(np.float32), small.size)


def scene_classes(img: Image.Image, min_fraction: float = 0.02) -> dict:
    """Which known classes are actually present, and how much of the frame.

    Used to decide which region recipes are worth offering -- the SegFormer
    forward pass happens anyway for the sky mask, so this is free.
    """
    labels = _ade_labels(_at_working_size(img))
    total = labels.size
    out = {}
    for name, idx in ADE_CLASSES.items():
        frac = float((labels == idx).sum()) / total
        if frac >= min_fraction:
            out[name] = round(frac, 4)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ---------------- depth (Depth Anything V2) ----------------

def _depth_model():
    if "depth" not in _cache:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        name = "depth-anything/Depth-Anything-V2-Small-hf"
        proc = AutoImageProcessor.from_pretrained(name)
        model = AutoModelForDepthEstimation.from_pretrained(name).eval().to(_device())
        _cache["depth"] = (proc, model)
    return _cache["depth"]


@torch.no_grad()
def depth_map(img: Image.Image) -> np.ndarray:
    """Relative depth, float32 (H, W) normalised to [0,1] where 1 = nearest.

    Relative, not metric -- the scale is per-image, so it is meaningful for
    "grade the near third differently from the far third" and meaningless as
    an absolute distance.
    """
    proc, model = _depth_model()
    small = _at_working_size(img.convert("RGB"))
    inputs = proc(images=small, return_tensors="pt").to(_device())
    d = model(**inputs).predicted_depth[0].cpu().numpy().astype(np.float32)
    lo, hi = float(d.min()), float(d.max())
    d = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
    return _resize_mask(d, small.size)


def depth_band(depth: np.ndarray, near: float, far: float, softness: float = 0.12) -> np.ndarray:
    """Select a slice of the depth range with soft shoulders.

    near/far are in the same [0,1] units as depth_map (1 = nearest), so
    depth_band(d, 0.6, 1.0) is "the closest 40% of the scene".
    """
    lo, hi = min(near, far), max(near, far)
    s = max(softness, 1e-3)
    rising = np.clip((depth - (lo - s)) / s, 0, 1)
    falling = np.clip(((hi + s) - depth) / s, 0, 1)
    return (rising * falling).astype(np.float32)


# ---------------- text prompt (CLIPSeg) ----------------

def _clipseg():
    if "clipseg" not in _cache:
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        name = "CIDAS/clipseg-rd64-refined"
        proc = CLIPSegProcessor.from_pretrained(name)
        model = CLIPSegForImageSegmentation.from_pretrained(name).eval().to(_device())
        _cache["clipseg"] = (proc, model)
    return _cache["clipseg"]


@torch.no_grad()
def prompt_mask(img: Image.Image, prompt: str) -> np.ndarray:
    """Mask from a free-text prompt. Coarser than the other two."""
    proc, model = _clipseg()
    small = _at_working_size(img)
    inputs = proc(text=[prompt], images=[small.convert("RGB")],
                  padding=True, return_tensors="pt").to(_device())
    logits = model(**inputs).logits
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    m = torch.sigmoid(logits)[0].cpu().numpy()
    return _resize_mask(m, small.size)


# ---------------- helpers ----------------

def _resize_mask(mask: np.ndarray, size) -> np.ndarray:
    im = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    im = im.resize(size, Image.BILINEAR)
    return np.asarray(im).astype(np.float32) / 255.0


def resample(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resample a stored mask to a target (w, h). This is what lets one
    computed mask serve both the 480px preview and the full-size render."""
    return _resize_mask(mask, size)


def encode(mask: np.ndarray) -> bytes:
    """Mask -> 8-bit grayscale PNG, for persisting alongside an import.
    Lossless, and a 1024px single-channel mask is only a few tens of KB."""
    import io
    buf = io.BytesIO()
    Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), "L").save(buf, format="PNG")
    return buf.getvalue()


def decode(data: bytes) -> np.ndarray:
    import io
    im = Image.open(io.BytesIO(data)).convert("L")
    return np.asarray(im).astype(np.float32) / 255.0


def feather(mask: np.ndarray, radius_frac: float = 0.005) -> np.ndarray:
    """Soften mask edges so the composite doesn't show a hard cutout line --
    the single biggest giveaway of a machine-made selection.

    The radius is a FRACTION OF THE LONG EDGE, not pixels. It used to be
    pixels, which meant the same constant produced a 0.52%-of-frame feather
    on the 480px preview and a 0.042% feather on the 6000px download -- edges
    12.5x harder in the file you keep than in the one you approved.
    """
    if radius_frac <= 0:
        return mask
    px = radius_frac * max(mask.shape)
    if px < 0.5:
        return mask
    im = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur(px))
    return np.asarray(im).astype(np.float32) / 255.0


def refine(mask: np.ndarray, threshold: float | None = None,
           feather_frac: float = 0.005, expand_frac: float = 0.0) -> np.ndarray:
    """threshold -> optional hard cut (leave None to keep the soft alpha);
    expand -> grow/shrink as a fraction of the long edge; then feather."""
    m = np.clip(mask, 0, 1)
    if threshold is not None:
        m = (m >= threshold).astype(np.float32)
    if expand_frac:
        px = int(abs(expand_frac) * max(m.shape))
        if px:
            im = Image.fromarray((m * 255).astype(np.uint8))
            f = ImageFilter.MaxFilter if expand_frac > 0 else ImageFilter.MinFilter
            im = im.filter(f(min(px * 2 + 1, 9)))
            m = np.asarray(im).astype(np.float32) / 255.0
    return feather(m, feather_frac)


def mask_stats(mask: np.ndarray) -> dict:
    """Coverage and confidence for a soft mask.

    `coverage` alone cannot tell "found nothing" from "found something
    small" -- a bird on a wire covers 0.9% of the frame and is exactly the
    photo selective colour is for. `confidence` (mean alpha over the pixels
    the model actually committed to) separates the two: a real small subject
    is confidently masked, noise is not.
    """
    m = np.clip(mask, 0, 1)
    committed = m > 0.5
    coverage = float(committed.mean())
    confidence = float(m[committed].mean()) if committed.any() else 0.0
    return {"coverage": round(coverage, 5), "confidence": round(confidence, 4)}


def is_usable(mask: np.ndarray, min_coverage: float = 0.001,
              max_coverage: float = 0.97, min_confidence: float = 0.80) -> bool:
    """Whether a mask isolated something worth grading through."""
    s = mask_stats(mask)
    return (min_coverage < s["coverage"] < max_coverage
            and s["confidence"] >= min_confidence)


def loaded() -> bool:
    """Whether any segmentation model is currently resident."""
    return bool(_cache)


def unload():
    """Drop all segmentation models (they're the RAM-heavy part)."""
    _cache.clear()
    import gc
    gc.collect()
