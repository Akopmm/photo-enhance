"""The noise estimator, and the failure that made it worth testing.

It decides whether a photo is denoised at all. A frame measuring 3.37 on red
and 2.87 on blue was reported as 1.91, fell under the 3.0 threshold, and was
exported having never been denoised -- because the estimator averaged the
channels into a greyscale first, and sensor noise is largely independent
between channels, so that cancels roughly sqrt(3) of it. What it cancels most
is chroma noise, which is exactly the kind a warm indoor photo has.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

import denoise as dn  # noqa: E402


def _flat(h=256, w=256, level=0.45):
    return np.full((h, w, 3), level, np.float32)


def _add(img, sigma8, channels, seed=0):
    rng = np.random.default_rng(seed)
    out = img.copy()
    for c in channels:
        out[..., c] += rng.normal(0, sigma8 / 255.0, img.shape[:2]).astype(np.float32)
    return np.clip(out, 0, 1)


def test_a_clean_frame_reads_as_clean():
    assert dn.estimate_sigma(_flat()) < 0.5


def test_noise_in_one_channel_is_not_averaged_away():
    # THE regression. Noise in red alone must be reported near its true level,
    # not divided down by the two quiet channels beside it.
    got = dn.estimate_sigma(_add(_flat(), 6.0, [0]))
    assert got > 4.5, f"single-channel noise under-reported as {got:.2f}"


def test_chroma_noise_is_seen():
    # Independent noise in red and blue, none in green: a greyscale average
    # loses most of this, which is how the real photo slipped through.
    a = _add(_flat(), 5.0, [0], seed=1)
    a = _add(a, 5.0, [2], seed=2)
    got = dn.estimate_sigma(a)
    assert got > 3.5, f"chroma noise under-reported as {got:.2f}"


def test_it_tracks_the_noise_level():
    seen = [dn.estimate_sigma(_add(_flat(), s, [0, 1, 2])) for s in (2.0, 5.0, 10.0)]
    assert seen == sorted(seen), f"not monotonic: {seen}"
    assert abs(seen[1] - 5.0) < 2.0, f"5.0 reported as {seen[1]:.2f}"


def test_detail_is_not_mistaken_for_noise():
    # The reason it looks at flat blocks at all: a busy, perfectly clean frame
    # must not read as noisy, or every sharp photo would be smoothed.
    rng = np.random.default_rng(3)
    busy = np.zeros((256, 256, 3), np.float32)
    for _ in range(60):
        y, x = rng.integers(0, 200, 2)
        busy[y:y + 50, x:x + 50] = rng.random(3).astype(np.float32)
    assert dn.estimate_sigma(busy) < 1.5, "hard edges were counted as noise"


def test_small_frames_do_not_crash():
    for h, w in [(8, 8), (63, 63), (64, 64), (1, 1)]:
        v = dn.estimate_sigma(np.full((h, w, 3), 0.5, np.float32))
        assert v >= 0.0 and np.isfinite(v), f"{h}x{w} -> {v}"


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
