"""The masked sensor border some RAWs decode with, and what must survive it.

LibRaw returns a frame slightly taller than the visible image for a number of
bodies -- Sony ARW most visibly -- and the extra rows are the sensor's
optically masked border, i.e. solid black. Nothing downstream can tell them
from picture, so they end up in the thumbnail, in the model's input and in the
exported file as a black strip along one edge.

The trim that removes them is the kind of heuristic that is easy to make too
eager. Half of these cases exist for that direction: a night shot and a
deliberate letterbox have to come through untouched, and the first version of
the guard ate 2% off every edge of a uniformly dark frame.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

from pipeline import _trim_dead_border  # noqa: E402


def _photo(h=400, w=600, seed=0):
    return np.random.default_rng(seed).integers(40, 255, (h, w, 3), dtype=np.uint8)


def _black(h, w):
    return np.zeros((h, w, 3), np.uint8)


def test_removes_a_masked_strip_on_any_edge():
    photo = _photo()
    cases = {
        "bottom": np.vstack([photo, _black(6, 600)]),
        "top": np.vstack([_black(4, 600), photo]),
        "left": np.hstack([_black(400, 5), photo]),
        "right": np.hstack([photo, _black(400, 3)]),
    }
    for edge, padded in cases.items():
        out = _trim_dead_border(padded)
        assert out.shape == photo.shape, f"{edge} strip not removed: {out.shape}"
        assert np.array_equal(out, photo), f"{edge} trim moved the picture"


def test_a_clean_frame_is_returned_untouched():
    photo = _photo()
    out = _trim_dead_border(photo)
    assert out.shape == photo.shape
    assert np.array_equal(out, photo)


def test_the_real_geometry_from_a_sony_arw():
    # Measured from DSC00573.ARW: half-size decode 3584x2560 with 216 dead
    # rows at the bottom and 64 dead columns at the right, leaving a clean
    # 3:2. Scaled down here so the test stays fast.
    photo = _photo(586, 880)
    frame = np.vstack([np.hstack([photo, _black(586, 16)]), _black(54, 896)])
    out = _trim_dead_border(frame)
    assert out.shape == photo.shape, out.shape
    assert abs(out.shape[1] / out.shape[0] - 1.5) < 0.01


def test_a_very_dark_band_that_is_not_exactly_zero_survives():
    # The realistic false positive: a night sky, or heavy vignetting. Dark,
    # but carrying sensor noise, so never exactly zero. This is the case the
    # exact-zero test exists to protect.
    photo = _photo()
    murk = np.random.default_rng(2).integers(0, 3, (80, 600, 3), dtype=np.uint8)
    murk[murk == 0] = 1          # dark everywhere, zero nowhere
    boxed = np.vstack([photo, murk])
    assert _trim_dead_border(boxed).shape == boxed.shape


def test_a_black_run_wider_than_the_budget_is_left_alone():
    # The darkness never stops, so this is the picture, not a border.
    frame = np.vstack([_photo(100, 600), _black(400, 600)])
    assert _trim_dead_border(frame).shape == frame.shape


def test_a_uniformly_black_frame_is_untouched():
    assert _trim_dead_border(_black(400, 600)).shape == (400, 600, 3)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                # Not just AssertionError: a test that raises NameError or
                # ImportError has found something too, and should say which
                # test it was rather than dumping a bare traceback.
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print("all passed" if not failures else f"{failures} failed")
    sys.exit(1 if failures else 0)
