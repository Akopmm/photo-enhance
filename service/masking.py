"""Semantic mask generation for region-aware photo grading.

Three complementary mask sources, because no single model does all of it well:

  subject  -- BiRefNet-lite (44M params, MIT). Purpose-built dichotomous
              segmentation; gives crisp foreground cutouts. This is the
              "Select Subject" equivalent and the best of the three.
  sky      -- SegFormer trained on ADE20K, which has a dedicated `sky`
              class. Also exposes other useful scene classes for free.
  prompt   -- CLIPSeg, arbitrary text ("car", "the red jacket"). Most
              flexible, but noticeably coarser/softer than the other two --
              use it for broad regions, not crisp cutouts.

Masks are computed at the model's native working resolution and then
resized up to the target image, which is both faster and (for these models)
no worse than feeding full resolution -- they downsample internally anyway.
"""
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter

_cache = {}


def _device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


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
    """Returns float32 (H, W) in [0,1] -- 1 = subject."""
    m = _birefnet()
    tf = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(img.convert("RGB")).unsqueeze(0).to(_device())
    pred = m(x)[-1].sigmoid().cpu()[0].squeeze().numpy()
    return _resize_mask(pred, img.size)


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
def class_mask(img: Image.Image, class_name: str) -> np.ndarray:
    """Mask for one ADE20K class, e.g. 'sky'. float32 (H, W) in [0,1]."""
    if class_name not in ADE_CLASSES:
        raise ValueError(f"unknown class {class_name!r}; known: {sorted(ADE_CLASSES)}")
    proc, model = _segformer()
    inputs = proc(images=img.convert("RGB"), return_tensors="pt").to(_device())
    logits = model(**inputs).logits  # (1, 150, h/4, w/4)
    labels = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
    return _resize_mask((labels == ADE_CLASSES[class_name]).astype(np.float32), img.size)


@torch.no_grad()
def scene_classes(img: Image.Image, min_fraction: float = 0.02) -> dict:
    """Which known classes are actually present, and how much of the frame."""
    proc, model = _segformer()
    inputs = proc(images=img.convert("RGB"), return_tensors="pt").to(_device())
    labels = model(**inputs).logits.argmax(dim=1)[0].cpu().numpy()
    total = labels.size
    out = {}
    for name, idx in ADE_CLASSES.items():
        frac = float((labels == idx).sum()) / total
        if frac >= min_fraction:
            out[name] = round(frac, 4)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


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
    inputs = proc(text=[prompt], images=[img.convert("RGB")],
                  padding=True, return_tensors="pt").to(_device())
    logits = model(**inputs).logits
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    m = torch.sigmoid(logits)[0].cpu().numpy()
    return _resize_mask(m, img.size)


# ---------------- helpers ----------------

def _resize_mask(mask: np.ndarray, size) -> np.ndarray:
    im = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    im = im.resize(size, Image.BILINEAR)
    return np.asarray(im).astype(np.float32) / 255.0


def feather(mask: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """Soften mask edges so the composite doesn't show a hard cutout line --
    the single biggest giveaway of a machine-made selection."""
    if radius <= 0:
        return mask
    im = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(im).astype(np.float32) / 255.0


def refine(mask: np.ndarray, threshold: float | None = None,
           feather_px: float = 2.0, expand: float = 0.0) -> np.ndarray:
    """threshold -> optional hard cut; expand -> grow/shrink; then feather."""
    m = np.clip(mask, 0, 1)
    if threshold is not None:
        m = (m >= threshold).astype(np.float32)
    if expand:
        im = Image.fromarray((m * 255).astype(np.uint8))
        f = ImageFilter.MaxFilter if expand > 0 else ImageFilter.MinFilter
        k = int(abs(expand)) * 2 + 1
        im = im.filter(f(min(k, 9)))
        m = np.asarray(im).astype(np.float32) / 255.0
    return feather(m, feather_px)


def loaded() -> bool:
    """Whether any segmentation model is currently resident."""
    return bool(_cache)


def unload():
    """Drop all segmentation models (they're the RAM-heavy part)."""
    _cache.clear()
    import gc
    gc.collect()
