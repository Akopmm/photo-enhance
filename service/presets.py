"""Deterministic look/style presets applied on top of the model's corrected
baseline image, producing several distinct Lightroom-style variants per
photo instead of training a separate model per style.

Reimplemented in PyTorch (was plain numpy) so the whole pipeline -- model
inference AND preset rendering -- is one tensor computation graph that can
run on optiplex's integrated GPU via OpenVINO (see model_runtime.py), not
just CPU. The box blur, previously a numpy cumsum trick, is now a separable
conv2d with a uniform kernel -- mathematically the same box filter, just
expressed as an op OpenVINO/any GPU backend can actually accelerate.
"""
import torch
import torch.nn.functional as F


def _luminance(a: torch.Tensor) -> torch.Tensor:
    # a: (B, 3, H, W) -> (B, 1, H, W)
    r, g, b = a[:, 0:1], a[:, 1:2], a[:, 2:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _box_blur(a: torch.Tensor, r: int) -> torch.Tensor:
    """a: (B, C, H, W). Separable box blur, edge-replicated border -- same
    result as the original numpy cumsum implementation."""
    k = 2 * r + 1
    c = a.shape[1]
    weight_h = a.new_full((c, 1, 1, k), 1.0 / k)
    weight_v = a.new_full((c, 1, k, 1), 1.0 / k)
    a = F.pad(a, (r, r, 0, 0), mode="replicate")
    a = F.conv2d(a, weight_h, groups=c)
    a = F.pad(a, (0, 0, r, r), mode="replicate")
    a = F.conv2d(a, weight_v, groups=c)
    return a


def _s_curve(x: torch.Tensor, strength: float) -> torch.Tensor:
    return x + strength * (x - 0.5) * (1 - torch.abs(2 * x - 1))


def apply_look(arr: torch.Tensor, p: dict) -> torch.Tensor:
    """arr: (B, 3, H, W) float32 in [0, 1] (or (3, H, W), auto-batched).
    Returns a tensor of the same shape."""
    squeeze = arr.dim() == 3
    a = arr.unsqueeze(0) if squeeze else arr
    a = a.clone()

    a = torch.clamp(a * p.get("exposure", 1.0), 0, 1)
    wb = p.get("wb")
    if wb:
        wb_t = a.new_tensor(wb).view(1, 3, 1, 1)
        a = torch.clamp(a + wb_t, 0, 1)

    lum = _luminance(a)
    dh = p.get("dehaze", 0.0)
    if dh:
        blurred = _box_blur(lum, 8)
        detail = lum - blurred
        lum2 = lum + detail * dh
        gain = torch.clamp(lum2, 0.001, 4) / torch.clamp(lum, 0.001, 4)
        a = torch.clamp(a * gain, 0, 1)
        lum = _luminance(a)

    hr = p.get("highlight_recovery", 0.0)
    if hr:
        knee = 0.78
        over = torch.clamp((lum - knee) / (1 - knee), 0, 1)
        lum2 = lum - over * hr
        gain = torch.clamp(lum2, 0.001, 4) / torch.clamp(lum, 0.001, 4)
        a = torch.clamp(a * gain, 0, 1)
        lum = _luminance(a)

    sl = p.get("shadow_lift", 0.0)
    if sl:
        a = torch.clamp(a * (1 - sl) + sl, 0, 1)

    bp = p.get("black_crush", 0.0)
    if bp:
        a = torch.clamp((a - bp) / (1 - bp), 0, 1)

    cs_strength = p.get("contrast", 0.0)
    if cs_strength:
        lum = _luminance(a)
        new_lum = torch.clamp(_s_curve(lum, cs_strength), 0.001, 1)
        a = torch.clamp(a * (new_lum / torch.clamp(lum, 0.001, 1)), 0, 1)

    lum = _luminance(a)
    sh_col = p.get("shadow_tone")
    hi_col = p.get("highlight_tone")
    if sh_col:
        w = torch.clamp(1 - lum / 0.5, 0, 1) ** 1.3
        a = torch.clamp(a + w * a.new_tensor(sh_col).view(1, 3, 1, 1), 0, 1)
    if hi_col:
        w = torch.clamp((lum - 0.5) / 0.5, 0, 1) ** 1.1
        a = torch.clamp(a + w * a.new_tensor(hi_col).view(1, 3, 1, 1), 0, 1)

    sat_amt = p.get("saturation", 0.0)
    if sat_amt:
        gray = _luminance(a)
        if p.get("vibrance_protect", True):
            mx = a.amax(dim=1, keepdim=True)
            mn = a.amin(dim=1, keepdim=True)
            sat = torch.where(mx > 0, (mx - mn) / torch.clamp(mx, min=1e-4), torch.zeros_like(mx))
            strength = sat_amt * (1 - sat)
            a = gray + (a - gray) * (1 + strength)
        else:
            a = gray + (a - gray) * (1 + sat_amt)
        a = torch.clamp(a, 0, 1)

    desat = p.get("desaturate", 0.0)
    if desat:
        gray = _luminance(a)
        a = torch.clamp(a * (1 - desat) + gray * desat, 0, 1)

    if p.get("bw", False):
        g = _luminance(a)
        a = g.expand(-1, 3, -1, -1).contiguous()

    glow = p.get("glow", 0.0)
    if glow:
        lum = _luminance(a)
        bright = torch.clamp((lum - 0.55) / 0.45, 0, 1)
        blurred = _box_blur(a * bright, 14)
        a = 1 - (1 - a) * (1 - blurred * glow)
        a = torch.clamp(a, 0, 1)

    grain = p.get("grain", 0.0)
    if grain:
        gen = torch.Generator(device="cpu").manual_seed(7)
        noise = torch.randn(a.shape[0], 1, a.shape[2], a.shape[3], generator=gen).to(a.device) * grain
        a = torch.clamp(a + noise, 0, 1)

    vig = p.get("vignette", 0.0)
    if vig:
        h, w = a.shape[2], a.shape[3]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=a.device),
            torch.linspace(-1, 1, w, device=a.device),
            indexing="ij",
        )
        dist = torch.sqrt(xx ** 2 + yy ** 2)
        v = 1 - torch.clamp(dist - 0.6, min=0) * vig
        a = torch.clamp(a * v.unsqueeze(0).unsqueeze(0), 0, 1)

    return a.squeeze(0) if squeeze else a


PRESETS = [
    ("natural_light", "Natural Light+", dict(
        exposure=1.05, wb=(0.005, 0.0, -0.01), dehaze=0.15,
        highlight_recovery=0.06, contrast=0.06, saturation=0.05,
    )),
    ("hdr_punch", "HDR Punch", dict(
        dehaze=0.75, highlight_recovery=0.18, shadow_lift=0.06,
        contrast=0.22, saturation=0.30, vignette=0.15,
    )),
    ("cinematic_teal_orange", "Cinematic Teal-Orange", dict(
        dehaze=0.35, black_crush=0.04, contrast=0.16,
        shadow_tone=(-0.05, 0.01, 0.06), highlight_tone=(0.06, 0.02, -0.05),
        saturation=0.05, desaturate=0.10, vignette=0.30,
    )),
    ("golden_hour_warm", "Golden Hour Warm", dict(
        exposure=1.03, wb=(0.035, 0.012, -0.03), dehaze=0.15,
        shadow_lift=0.03, contrast=0.08, saturation=0.10,
        glow=0.18, vignette=0.10,
    )),
    ("moody_matte_film", "Moody Matte Film", dict(
        dehaze=0.10, shadow_lift=0.10, contrast=-0.05,
        shadow_tone=(-0.01, 0.01, 0.02), desaturate=0.18,
        grain=0.02, vignette=0.20,
    )),
    ("black_white_dramatic", "B&W Dramatic", dict(
        dehaze=0.45, black_crush=0.05, contrast=0.28,
        bw=True, vignette=0.30,
    )),
    ("clean_commercial", "Clean Commercial", dict(
        exposure=1.08, wb=(-0.005, 0.0, 0.01), highlight_recovery=0.04,
        contrast=0.03, saturation=-0.08, shadow_lift=0.04,
    )),
    ("vibrant_punch", "Vibrant Punch", dict(
        exposure=1.02, wb=(0.015, 0.0, -0.01), dehaze=0.40,
        black_crush=0.03, contrast=0.20, saturation=0.35,
        vignette=0.18,
    )),
    ("bright_airy", "Bright & Airy", dict(
        exposure=1.12, wb=(0.008, 0.0, -0.01), shadow_lift=0.12,
        highlight_recovery=0.02, contrast=-0.04, saturation=-0.05,
    )),
    ("faded_retro", "Faded Retro", dict(
        exposure=1.02, wb=(0.02, 0.005, -0.015), shadow_lift=0.14,
        black_crush=-0.06, contrast=-0.08, desaturate=0.12,
        grain=0.035, vignette=0.12,
    )),
    ("deep_contrast_noir", "Deep Contrast Noir", dict(
        black_crush=0.08, contrast=0.40, shadow_lift=-0.02,
        bw=True, vignette=0.22,
    )),
    ("cool_arctic", "Cool Arctic", dict(
        wb=(-0.025, 0.0, 0.03), dehaze=0.20, contrast=0.10,
        shadow_tone=(-0.03, 0.0, 0.04), saturation=-0.10,
        vignette=0.14,
    )),
]
