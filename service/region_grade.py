"""Region-aware grading: apply different looks to different parts of a photo
and composite them through a feathered mask.

This is the piece our global 3D-LUT model structurally cannot do -- that
model learns one color mapping applied to every pixel identically, so it has
no way to treat a subject differently from its background. Masking supplies
the "where"; the existing preset engine supplies the "what".
"""
import numpy as np
import torch

from presets import apply_look


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def _to_array(t: torch.Tensor) -> np.ndarray:
    return t.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()


def graded(arr: np.ndarray, params: dict) -> np.ndarray:
    """Run one preset over a whole (H,W,3) float image in [0,1]."""
    if not params:
        return arr
    return _to_array(apply_look(_to_tensor(arr), params))


def composite(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """mask=1 -> overlay, mask=0 -> base."""
    m = np.clip(mask, 0, 1)[..., None]
    return np.clip(base * (1 - m) + overlay * m, 0, 1)


def region_grade(arr: np.ndarray, regions: list) -> np.ndarray:
    """regions: list of (mask_or_None, params). A None mask means "everything"
    and is applied as a base layer; later entries composite over earlier ones.

    Example -- colour subject on a mono background:
        region_grade(img, [
            (None,         BW_PARAMS),      # whole frame goes mono
            (subject_mask, COLOR_PARAMS),   # subject painted back in colour
        ])
    """
    out = arr
    for mask, params in regions:
        layer = graded(arr, params) if params else arr
        out = layer if mask is None else composite(out, layer, mask)
    return out


# ---------------------------------------------------------------- recipes

# "Selective colour": the Lightroom effect where the subject keeps its colour
# and everything else drops to black & white.
SELECTIVE_COLOR = {
    "background": dict(bw=True, contrast=0.18, dehaze=0.20, black_crush=0.03, vignette=0.25),
    "subject": dict(saturation=0.28, dehaze=0.25, contrast=0.10, highlight_recovery=0.05),
}

# Sky-aware landscape: deepen and warm the sky, lift and clarify the ground.
SKY_DRAMA = {
    "ground": dict(dehaze=0.35, contrast=0.12, shadow_lift=0.04, saturation=0.10),
    "sky": dict(exposure=0.94, saturation=0.30, dehaze=0.45, contrast=0.18,
                highlight_tone=(0.02, 0.0, 0.05)),
}

# Subject-forward portrait/product look: background pushed back, subject popped.
SUBJECT_POP = {
    "background": dict(exposure=0.88, desaturate=0.35, contrast=-0.05, vignette=0.30),
    "subject": dict(dehaze=0.30, saturation=0.18, contrast=0.12, highlight_recovery=0.06),
}
