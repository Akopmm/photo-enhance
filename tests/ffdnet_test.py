"""The balanced engine: FFDNet, wired the way the service will call it.

It exists because SCUNet costs 476s on a 26MP frame and this costs 42s for
output the crops could not tell apart. That claim rests on a noise level the
service estimates and then scales, so these tests pin the wiring around it:
the model loads, a tile keeps its shape, the level actually reaches the model,
and the routing sends "balanced" here rather than to SCUNet or the filter.

What they deliberately do NOT assert is denoising quality. That was settled by
measurement and by looking at crops, not by a threshold a test could check --
see research/denoise-speed/FINDINGS.md.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

import denoise as dn  # noqa: E402

HAVE_WEIGHTS = os.path.exists(os.path.join(
    os.path.dirname(__file__), "..", "service", "weights", "ffdnet_color.pth"))


def _noisy(h=256, w=256, sigma=0.03, seed=0):
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w, 3), np.float32) + 0.45
    base[:, w // 2:] = 0.65
    return np.clip(base + rng.normal(0, sigma, (h, w, 3)).astype(np.float32), 0, 1)


def test_the_model_loads_and_keeps_tile_shape():
    if not HAVE_WEIGHTS:
        print("  (skipped: weights not present)")
        return
    for h, w in [(256, 256), (255, 257), (64, 64)]:
        t = torch.rand(1, 3, h, w)
        out = dn._run_ffdnet(t, 8.0)
        assert tuple(out.shape) == (1, 3, h, w), f"{h}x{w} -> {tuple(out.shape)}"
        assert torch.isfinite(out).all()
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_odd_sizes_survive_the_pixel_shuffle():
    # FFDNet halves the spatial dimensions internally, so an odd tile has to
    # be padded and cropped back. Getting that wrong shifts the image by a
    # pixel, which is invisible in a metric and obvious as a seam once tiles
    # are blended.
    if not HAVE_WEIGHTS:
        print("  (skipped: weights not present)")
        return
    flat = torch.full((1, 3, 101, 103), 0.5)
    out = dn._run_ffdnet(flat, 6.0)
    assert tuple(out.shape) == (1, 3, 101, 103)
    assert float(out.std()) < 0.02, "a flat tile came back structured; padding is misaligned"


def test_the_noise_level_actually_reaches_the_model():
    # If the level were ignored or mis-scaled, the two results would match and
    # the whole calibration question would be silently moot.
    if not HAVE_WEIGHTS:
        print("  (skipped: weights not present)")
        return
    t = torch.from_numpy(_noisy()).permute(2, 0, 1)[None]
    low = dn._run_ffdnet(t, 2.0)
    high = dn._run_ffdnet(t, 25.0)
    assert float((low - high).abs().mean()) > 1e-3, "the sigma input changed nothing"
    assert float(high.std()) < float(low.std()), "a higher level denoised less"


def test_routing_picks_the_right_engine():
    calls = {"scunet": 0, "ffdnet": 0}
    real_run, real_ffd = dn._run, dn._run_ffdnet
    dn._run = lambda t: calls.__setitem__("scunet", calls["scunet"] + 1) or real_run(t)
    dn._run_ffdnet = lambda t, s: calls.__setitem__("ffdnet", calls["ffdnet"] + 1) or real_ffd(t, s)
    try:
        a = _noisy(600, 600)
        dn.denoise(a, method="fast")
        assert calls == {"scunet": 0, "ffdnet": 0}, f"fast ran a network: {calls}"
        if HAVE_WEIGHTS:
            dn.denoise(a, method="balanced")
            assert calls["ffdnet"] > 0 and calls["scunet"] == 0, f"balanced routed wrong: {calls}"
    finally:
        dn._run, dn._run_ffdnet = real_run, real_ffd


def test_the_level_is_measured_once_for_the_whole_frame():
    # Per-tile levels would smooth a tile of sky and a tile of foliage
    # differently, which shows up as blocking. Every tile must be told the same
    # number.
    if not HAVE_WEIGHTS:
        print("  (skipped: weights not present)")
        return
    seen = []
    real = dn._run_ffdnet
    dn._run_ffdnet = lambda t, s: seen.append(s) or real(t, s)
    try:
        dn.denoise(_noisy(900, 1400), method="balanced")
    finally:
        dn._run_ffdnet = real
    assert len(seen) > 1, "expected several tiles"
    assert len(set(seen)) == 1, f"tiles were given different noise levels: {set(seen)}"


def test_the_floor_stops_a_zero_level():
    a = np.full((300, 300, 3), 0.5, np.float32)     # perfectly clean
    seen = []
    real = dn._run_ffdnet
    dn._run_ffdnet = lambda t, s: seen.append(s) or torch.zeros_like(t)
    try:
        dn.denoise(a, method="balanced")
    finally:
        dn._run_ffdnet = real
    assert seen and seen[0] >= dn.FFDNET_MIN_SIGMA, f"level fell below the floor: {seen}"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print("all passed" if not failures else f"{failures} failed")
    sys.exit(1 if failures else 0)
