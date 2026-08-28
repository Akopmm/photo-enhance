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


def test_a_uniformly_dark_photo_is_not_eaten():
    # Every edge looks "dead" here. The trim must notice the darkness does not
    # stop, and leave the frame alone rather than cropping its budget off it.
    dark = np.random.default_rng(1).integers(0, 4, (400, 600, 3), dtype=np.uint8)
    assert _trim_dead_border(dark).shape == dark.shape


def test_a_deliberate_letterbox_survives():
    # Someone exported a 2.39:1 crop with real black bars. Those are content.
    photo = _photo()
    boxed = np.vstack([_black(60, 600), photo, _black(60, 600)])
    assert _trim_dead_border(boxed).shape == boxed.shape


def test_the_trim_is_bounded():
    # Even against an entirely black frame it can never remove more than its
    # budget, so a bug here cannot destroy a photo.
    out = _trim_dead_border(_black(400, 600))
    assert out.shape == (400, 600, 3)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print("all passed" if not failures else f"{failures} failed")
    sys.exit(1 if failures else 0)
