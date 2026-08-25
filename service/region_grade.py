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


def region_grade(arr: np.ndarray, regions: list, strength: float = 1.0) -> np.ndarray:
    """regions: list of (mask_or_None, params). A None mask means "everything"
    and is applied as a base layer; later entries composite over earlier ones.

    Example -- colour subject on a mono background:
        region_grade(img, [
            (None,         BW_PARAMS),      # whole frame goes mono
            (subject_mask, COLOR_PARAMS),   # subject painted back in colour
        ])

    `strength` in [0,1] dials the whole recipe down toward the ungraded
    image. It has to attenuate the None layers too, not just scale the
    masks -- in Selective Colour the mono conversion IS a None layer, so
    scaling only the subject mask would leave the background fully
    black & white however far the control was pulled back.
    """
    out = arr
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0:
        return arr
    for mask, params in regions:
        layer = graded(arr, params) if params else arr
        if mask is None:
            out = layer if s >= 1.0 else np.clip(out * (1 - s) + layer * s, 0, 1)
        else:
            out = composite(out, layer, mask if s >= 1.0 else mask * s)
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

# Depth-based looks. These need no foreground/background decision at all --
# they grade by distance, the way Lightroom's Depth Range Mask does, so they
# still work on the scenes where subject segmentation finds nothing to grab.
DEPTH_POP = {
    "far": dict(exposure=0.90, desaturate=0.30, contrast=-0.04),
    "near": dict(dehaze=0.28, saturation=0.16, contrast=0.10),
}

# Aerial perspective: the far field gets the lifted blacks and cool cast that
# distance actually does to a scene, the near field stays clear.
DEPTH_HAZE = {
    "near": dict(dehaze=0.22, contrast=0.08, saturation=0.08),
    "far": dict(shadow_lift=0.10, desaturate=0.22, contrast=-0.06,
                highlight_tone=(0.015, 0.025, 0.05)),
}

# Foliage-forward: offered only when SegFormer actually reports greenery.
FOLIAGE_LIFT = {
    "base": dict(contrast=0.10, dehaze=0.20),
    "foliage": dict(saturation=0.26, dehaze=0.30, shadow_lift=0.03,
                    shadow_tone=(-0.01, 0.02, -0.01)),
}

# Subject-forward portrait/product look: background pushed back, subject popped.
SUBJECT_POP = {
    "background": dict(exposure=0.88, desaturate=0.35, contrast=-0.05, vignette=0.30),
    "subject": dict(dehaze=0.30, saturation=0.18, contrast=0.12, highlight_recovery=0.06),
}
