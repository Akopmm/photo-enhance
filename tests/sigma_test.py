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


def test_sampling_does_not_hide_the_noise():
    """A large frame is sampled, and sampling must not soften it.

    Decimation keeps each sampled pixel exactly as it was; a resize would
    average neighbours and report a cleaner photo than the one being
    exported. That distinction is the whole reason a noisy frame was skipped,
    so it is pinned here: the same noise must read the same at any size.
    """
    small = _add(_flat(512, 512), 6.0, [0, 1, 2], seed=5)
    big = np.repeat(np.repeat(small, 6, axis=0), 6, axis=1)   # 3072px, same pixels
    a, b = dn.estimate_sigma(small), dn.estimate_sigma(big)
    assert abs(a - b) < 1.5, f"sampling changed the reading: {a:.2f} vs {b:.2f}"
    assert b > 4.0, f"the large frame under-reported: {b:.2f}"


def test_sampling_keeps_it_cheap():
    import time
    big = _add(_flat(4000, 6000), 5.0, [0, 1, 2], seed=6)
    t0 = time.time()
    dn.estimate_sigma(big)
    took = time.time() - t0
    assert took < 2.0, f"estimating a 24MP frame took {took:.2f}s"


def test_small_frames_do_not_crash():
    for h, w in [(8, 8), (63, 63), (64, 64), (1, 1)]:
        v = dn.estimate_sigma(np.full((h, w, 3), 0.5, np.float32))
        assert v >= 0.0 and np.isfinite(v), f"{h}x{w} -> {v}"


def test_the_shadow_level_finds_noise_the_global_figure_misses():
    """A dark frame whose shadows are noisy and whose highlights are clean.

    This is the shape that broke a fixed multiplier: the global estimate is
    taken from the flattest blocks, which in such a frame are the calm bright
    areas, so it reports a level far below what the shadows carry. Two photos
    measured 8.6 and 10.2 globally and needed opposite treatment because one
    carried 24.4 in its shadows and the other 10.2.
    """
    rng = np.random.default_rng(7)
    h, w = 512, 512
    img = np.zeros((h, w, 3), np.float32)
    img[:, :w // 2] = 0.25          # shadow half
    img[:, w // 2:] = 0.75          # bright half
    # noise only in the shadows, which is how a sensor behaves
    img[:, :w // 2] += rng.normal(0, 8.0 / 255, (h, w // 2, 3)).astype(np.float32)
    img = np.clip(img, 0, 1)

    glob = dn.estimate_sigma(img)
    shad = dn.shadow_sigma(img)
    assert shad > glob * 1.5, (
        f"the shadow level did not find it: global {glob:.2f}, shadow {shad:.2f}")
    assert shad > 5.0, f"shadow noise of 8 reported as {shad:.2f}"


def test_the_shadow_level_never_undercuts_the_global_one():
    # A frame with no shadows at all must not be denoised more weakly than
    # before just because the band is empty.
    rng = np.random.default_rng(8)
    bright = np.clip(np.full((512, 512, 3), 0.8, np.float32)
                     + rng.normal(0, 6.0 / 255, (512, 512, 3)).astype(np.float32), 0, 1)
    assert dn.shadow_sigma(bright) >= dn.estimate_sigma(bright) - 1e-6


def test_the_shadow_level_matches_the_global_one_on_even_noise():
    # Noise spread evenly across brightness: the two should broadly agree, or
    # the shadow rule would be quietly over-denoising ordinary photos.
    rng = np.random.default_rng(9)
    ramp = np.repeat(np.linspace(0.05, 0.95, 512, dtype=np.float32)[None, :, None], 512, 0)
    img = np.clip(np.repeat(ramp, 3, axis=2)
                  + rng.normal(0, 5.0 / 255, (512, 512, 3)).astype(np.float32), 0, 1)
    glob, shad = dn.estimate_sigma(img), dn.shadow_sigma(img)
    assert shad < glob * 1.6, f"over-reads even noise: global {glob:.2f}, shadow {shad:.2f}"


def test_the_default_amount_ramps_from_the_threshold():
    """A photo that passes the gate must not default to doing nothing.

    The ramp used to start two above the threshold, which was tuned against
    an estimator that under-reported by ~1.7x. Once that was fixed, photos
    between the threshold and threshold+2 were marked "denoise available" and
    then defaulted to 0% -- available and inert, which is the worst of both.
    """
    import os
    import tempfile
    os.environ.setdefault("RENDER_STORAGE_DIR", tempfile.mkdtemp())
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))
    import pipeline

    threshold = 3.0
    assert pipeline._default_denoise_amount(threshold - 1) == 0.0, "below the gate should be 0"
    assert pipeline._default_denoise_amount(threshold) == 0.0, "at the gate should be 0"
    just_over = pipeline._default_denoise_amount(threshold + 0.4)
    assert just_over > 0.0, "a photo that passes the gate defaulted to no denoising"
    # and it must still ramp, not jump straight to full strength
    assert just_over < 0.4, f"too eager just over the gate: {just_over}"
    # A photo a clear step above the gate has to be actually cleaned, not
    # nudged: at 26% a frame at sigma 3.2 only reached 2.54 from 3.21, which
    # is what "denoised" looked like to the person who reported it.
    clearly_noisy = pipeline._default_denoise_amount(threshold + 1.2)
    assert clearly_noisy >= 0.6, f"a clearly noisy photo defaults to only {clearly_noisy:.0%}"
    assert pipeline._default_denoise_amount(threshold + 1.5) >= 0.85, "never reaches full strength"


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
