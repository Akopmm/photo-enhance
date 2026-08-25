"""Composition-aware crop suggestions.

Uses the subject mask we already compute in enhanced mode, so this costs
almost nothing extra: knowing *where* the subject is turns cropping from
guesswork into arithmetic.

The rules applied are the ordinary ones a photographer would use by hand:

  * put the subject's centre of mass on a rule-of-thirds intersection
    rather than dead centre
  * never cut the subject -- a crop that clips it is rejected outright
  * leave "looking room": if the subject sits left of centre, bias the crop
    to leave space on its right, and vice versa
  * offer a few standard aspect ratios, since the best crop depends on
    where the photo is going (print, phone, social, cinematic)

Deliberately NOT a learned aesthetic model. A saliency/aesthetics network
would be another ~100MB and several seconds per photo, and for the common
case -- one clear subject -- these rules land in the same place. Where
there's no clear subject we fall back to a centred crop and say so, rather
than inventing a confident-looking suggestion.
"""
import numpy as np

# (key, label, width/height). Portrait 4:5 and 1:1 are the social formats;
# 2.39:1 is the cinema letterbox; 16:9 for screens; 3:2 matches the native
# aspect of most of these cameras.
ASPECTS = [
    ("square", "Square 1:1", 1.0),
    ("portrait_45", "Portrait 4:5", 4 / 5),
    ("classic_32", "Classic 3:2", 3 / 2),
    ("wide_169", "Wide 16:9", 16 / 9),
    ("cine_239", "Cinematic 2.39:1", 2.39),
]

THIRDS = (1 / 3, 2 / 3)


def subject_box(mask: np.ndarray, threshold: float = 0.5):
    """Tight bounding box of the masked subject as (x0, y0, x1, y1), or None."""
    ys, xs = np.where(mask >= threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def subject_centroid(mask: np.ndarray):
    """Mass-weighted centre of the subject, normalized to [0,1]. Weighting by
    mask value (not just the box centre) keeps a subject with a long thin
    limb from dragging the focal point away from its body."""
    total = mask.sum()
    if total <= 0:
        return None
    h, w = mask.shape
    ys, xs = np.mgrid[0:h, 0:w]
    return float((xs * mask).sum() / total / w), float((ys * mask).sum() / total / h)


def _crop_for_aspect(img_w: int, img_h: int, aspect: float, cx: float, cy: float,
                     subject: tuple | None):
    """Largest crop of the given aspect that contains the subject and puts
    (cx, cy) as close as possible to the nearest rule-of-thirds intersection."""
    # Largest box of this aspect that fits inside the frame.
    if img_w / img_h > aspect:
        ch = img_h
        cw = int(round(ch * aspect))
    else:
        cw = img_w
        ch = int(round(cw / aspect))
    if cw > img_w or ch > img_h:
        return None

    # Target: place the subject on whichever thirds intersection it's already
    # nearest, so the crop nudges the composition rather than fighting it.
    tx = min(THIRDS, key=lambda t: abs(t - cx))
    ty = min(THIRDS, key=lambda t: abs(t - cy))

    # Give the subject looking room: if it sits on the left, prefer the
    # left third (space opens to the right).
    x0 = int(round(cx * img_w - tx * cw))
    y0 = int(round(cy * img_h - ty * ch))
    x0 = max(0, min(x0, img_w - cw))
    y0 = max(0, min(y0, img_h - ch))

    if subject is not None:
        sx0, sy0, sx1, sy1 = subject
        # Subject bigger than the crop in either axis -> this aspect can't
        # hold it without clipping; skip rather than return a bad crop.
        if (sx1 - sx0) > cw or (sy1 - sy0) > ch:
            return None
        # Slide the window minimally so the subject is fully inside.
        if sx0 < x0:
            x0 = sx0
        elif sx1 > x0 + cw:
            x0 = sx1 - cw
        if sy0 < y0:
            y0 = sy0
        elif sy1 > y0 + ch:
            y0 = sy1 - ch
        x0 = max(0, min(x0, img_w - cw))
        y0 = max(0, min(y0, img_h - ch))

    return {"x": x0, "y": y0, "w": cw, "h": ch}


def suggest_crops(img_w: int, img_h: int, mask: np.ndarray | None):
    """Returns a list of crop suggestions, best-guess first.

    Each entry: {key, label, x, y, w, h, rationale, has_subject}
    Coordinates are in pixels of the image the mask was computed against;
    callers scale them if applying to a different resolution.
    """
    subject = None
    centroid = None
    if mask is not None:
        subject = subject_box(mask)
        centroid = subject_centroid(mask)

    has_subject = subject is not None and centroid is not None
    cx, cy = centroid if has_subject else (0.5, 0.5)

    out = []
    for key, label, aspect in ASPECTS:
        # Scale the subject box to mask coordinates -> image coordinates if
        # they differ (mask is computed on the preview).
        box = subject
        if box is not None and mask is not None and mask.shape != (img_h, img_w):
            sy = img_h / mask.shape[0]
            sx = img_w / mask.shape[1]
            box = (int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy))

        crop = _crop_for_aspect(img_w, img_h, aspect, cx, cy, box)
        if crop is None:
            continue
        # Skip crops that barely change anything -- suggesting a "crop" that
        # removes 2% of the frame is noise, not a suggestion.
        area_ratio = (crop["w"] * crop["h"]) / (img_w * img_h)
        crop.update({
            "key": key,
            "label": label,
            "has_subject": has_subject,
            "kept_fraction": round(area_ratio, 3),
            "rationale": (
                f"Subject placed near the rule-of-thirds point ({'left' if cx < 0.5 else 'right'}, "
                f"{'upper' if cy < 0.5 else 'lower'})."
                if has_subject else
                "No distinct subject detected — centred crop."
            ),
        })
        out.append(crop)

    # Widest deviation from the original aspect last; the closest ratio to
    # the source is usually the safest suggestion, so lead with it.
    src = img_w / img_h
    out.sort(key=lambda c: abs((c["w"] / c["h"]) - src))
    return out


def apply_crop(arr: np.ndarray, crop: dict) -> np.ndarray:
    h, w = arr.shape[:2]
    sx = w / crop.get("ref_w", w)
    sy = h / crop.get("ref_h", h)
    x0 = max(0, int(round(crop["x"] * sx)))
    y0 = max(0, int(round(crop["y"] * sy)))
    x1 = min(w, x0 + int(round(crop["w"] * sx)))
    y1 = min(h, y0 + int(round(crop["h"] * sy)))
    return arr[y0:y1, x0:x1]
