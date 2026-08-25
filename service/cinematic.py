"""Cinematic looks.

Not segmentation and not learned -- these are just stronger, more opinionated
parameter sets for the existing grading engine. What actually reads as
"cinematic" in a still is a fairly small set of ingredients:

  * a filmic tone curve      -- lifted blacks, rolled-off highlights, so
                                nothing clips to pure black or pure white
  * split toning             -- cool shadows against warm highlights
  * restrained saturation    -- film stocks are less saturated than digital,
                                but with rich, specific colour in the midtones
  * halation                 -- light bleeding around bright areas
  * grain                    -- breaks up digital smoothness
  * (optionally) a 2.39:1 letterbox crop

All of these already exist as knobs in service/presets.py, so this file is
purely tuning -- no new machinery, and it works on any photo without needing
a mask or a model.
"""

CINEMATIC = [
    ("cine_teal_orange", "Cine · Teal & Orange", dict(
        dehaze=0.30, contrast=0.14, shadow_lift=0.06, black_crush=0.02,
        shadow_tone=(-0.055, 0.005, 0.075), highlight_tone=(0.075, 0.025, -0.055),
        desaturate=0.14, saturation=0.10, glow=0.10, grain=0.012, vignette=0.28,
    )),
    ("cine_bleach", "Cine · Bleach Bypass", dict(
        dehaze=0.55, contrast=0.34, black_crush=0.05, shadow_lift=0.03,
        desaturate=0.45, saturation=0.08,
        highlight_tone=(0.02, 0.02, 0.03), grain=0.02, vignette=0.30,
    )),
    ("cine_noir_blue", "Cine · Midnight Blue", dict(
        exposure=0.94, dehaze=0.25, contrast=0.22, black_crush=0.04,
        shadow_tone=(-0.05, -0.01, 0.09), highlight_tone=(0.02, 0.02, 0.02),
        desaturate=0.30, glow=0.14, grain=0.018, vignette=0.36,
    )),
    ("cine_warm_film", "Cine · Warm Film", dict(
        exposure=1.03, wb=(0.028, 0.008, -0.022), shadow_lift=0.10,
        contrast=0.05, black_crush=-0.03,
        shadow_tone=(0.01, 0.005, 0.015), highlight_tone=(0.045, 0.02, -0.03),
        desaturate=0.12, saturation=0.14, glow=0.20, grain=0.025, vignette=0.18,
    )),
    ("cine_desat_thriller", "Cine · Cold Thriller", dict(
        exposure=0.96, wb=(-0.02, 0.0, 0.025), dehaze=0.40,
        contrast=0.26, black_crush=0.05,
        shadow_tone=(-0.04, -0.005, 0.05), desaturate=0.38,
        grain=0.022, vignette=0.34,
    )),
    ("cine_golden_epic", "Cine · Golden Epic", dict(
        exposure=1.05, wb=(0.04, 0.014, -0.035), dehaze=0.22,
        shadow_lift=0.05, contrast=0.12,
        highlight_tone=(0.06, 0.03, -0.05), shadow_tone=(-0.02, 0.0, 0.03),
        saturation=0.16, glow=0.26, grain=0.015, vignette=0.22,
    )),
]


def letterbox(arr, ratio: float = 2.39):
    """Crop to a cinema aspect ratio (2.39:1 by default). Purely optional --
    it's the one 'cinematic' ingredient that throws away pixels."""
    h, w = arr.shape[:2]
    target_h = int(round(w / ratio))
    if target_h >= h:
        return arr
    top = (h - target_h) // 2
    return arr[top:top + target_h]
