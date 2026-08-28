"""The fast denoiser: what it removes, what it deliberately leaves.

Measured at 0.09s against SCUNet's 16s on a 3200px frame. The temptation with
a result like that is to describe it as "almost as good", which is how the
research benchmark nearly ended up recommending it -- PSNR against SCUNet
could not distinguish "denoised" from "removed the colour noise and left the
grain".

So these tests assert both halves. It must take chroma noise out, and it must
leave luminance alone -- because that is the honest description of it, and a
future change that started smoothing luma would be a different filter wearing
this one's name and its timing claim.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

import denoise as dn  # noqa: E402


def _chroma(a):
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return np.stack([a[..., 2] - y, a[..., 0] - y], -1), y


def _noisy_patch(h=256, w=256, chroma_sigma=0.05, luma_sigma=0.02, seed=0):
    """A flat mid-grey with chroma blotches and luma grain on top."""
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), 0.45, np.float32)
    y_noise = rng.normal(0, luma_sigma, (h, w)).astype(np.float32)
    c_noise = rng.normal(0, chroma_sigma, (h, w, 2)).astype(np.float32)
    out = base + y_noise[..., None]
    out[..., 0] += c_noise[..., 1]
    out[..., 2] += c_noise[..., 0]
    return np.clip(out, 0, 1)


def test_chroma_noise_is_removed():
    a = _noisy_patch()
    before, _ = _chroma(a)
    after, _ = _chroma(dn.guided_chroma(a))
    drop = before.std() / max(after.std(), 1e-9)
    assert drop > 3.0, f"chroma noise barely changed: {before.std():.4f} -> {after.std():.4f}"


def test_luminance_is_left_alone():
    # The whole cost saving comes from not touching luma. If a change starts
    # smoothing it, this filter is no longer the thing that was measured.
    a = _noisy_patch()
    _, y_before = _chroma(a)
    _, y_after = _chroma(dn.guided_chroma(a))
    assert abs(float(y_before.std() - y_after.std())) < 0.002, (
        f"luma was altered: std {y_before.std():.4f} -> {y_after.std():.4f}")
    assert np.allclose(y_before, y_after, atol=2e-3), "luma changed pixel-wise"


def test_edges_are_not_bled_across():
    # A guided filter exists to avoid exactly this: colour smearing over a
    # hard boundary. Two saturated blocks side by side must stay separate.
    a = np.zeros((256, 256, 3), np.float32)
    a[:, :128] = (0.8, 0.2, 0.2)
    a[:, 128:] = (0.2, 0.2, 0.8)
    out = dn.guided_chroma(a)
    left, right = out[:, 40:88], out[:, 168:216]
    assert left[..., 0].mean() > left[..., 2].mean() + 0.3, "left block lost its red"
    assert right[..., 2].mean() > right[..., 0].mean() + 0.3, "right block lost its blue"


def test_shape_and_range_are_preserved():
    for h, w in [(256, 256), (200, 320), (37, 41), (8, 8), (1, 1)]:
        out = dn.guided_chroma(np.full((h, w, 3), 0.5, np.float32))
        assert out.shape == (h, w, 3), f"{h}x{w} -> {out.shape}"
        assert out.dtype == np.float32
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
        assert np.isfinite(out).all(), f"{h}x{w} produced non-finite values"


def test_a_clean_image_survives():
    a = np.tile(np.linspace(0.1, 0.9, 256, dtype=np.float32)[None, :, None], (256, 1, 3))
    out = dn.guided_chroma(a)
    assert np.allclose(a, out, atol=5e-3), "a noiseless gradient was altered"


def test_the_method_setting_chooses_the_engine():
    # Routing matters as much as the filter: "fast" must not reach the
    # network, and "quality" must not silently become the cheap filter.
    calls = []
    real_run = dn._run
    dn._run = lambda t: calls.append(1) or real_run(t)
    try:
        a = _noisy_patch(600, 600)          # big enough to tile
        dn.denoise(a, method="fast")
        assert not calls, "fast mode still ran the network"
    finally:
        dn._run = real_run


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
