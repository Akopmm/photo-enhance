"""A bad tile from the accelerator must not reach the blender.

An import failed with `operands could not be broadcast together with shapes
(0,0,3) (512,512,1)`. That is the tile blender being handed an empty array
where a 512x512 tile belonged, and it says nothing about where the empty array
came from -- the accelerated path had returned it several frames earlier.

So the accelerated path now checks what it got. These tests pin both halves of
that: a good tile is passed through untouched, and a bad one falls back to the
CPU instead of travelling on to fail somewhere unreadable.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

import denoise  # noqa: E402


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


def _with_fake_accelerator(returns, monkey):
    """Point _run at a stub accelerator returning `returns`."""
    monkey["model"] = denoise._model
    monkey["ov"] = denoise._openvino_tile_model
    monkey["infer"] = denoise.ov_infer.infer
    denoise._model = lambda: _Identity()
    denoise._openvino_tile_model = lambda: object()
    denoise.ov_infer.infer = lambda comp, key, arr: returns


def _restore(monkey):
    denoise._model = monkey["model"]
    denoise._openvino_tile_model = monkey["ov"]
    denoise.ov_infer.infer = monkey["infer"]


def test_the_real_accelerator_lookup_runs():
    # The other tests stub _openvino_tile_model out, so none of them execute
    # it. A version shipped that referenced a lock it never defined: every
    # call raised NameError, every GPU denoise failed, and the whole suite
    # stayed green. This calls the real thing. It returns None on a machine
    # with no iGPU, which is fine -- the point is that it returns.
    result = denoise._openvino_tile_model()
    assert result is None or hasattr(result, "create_infer_request"), result


def test_a_good_tile_is_used_as_is():
    monkey = {}
    tile = torch.full((1, 3, denoise.TILE, denoise.TILE), 0.5)
    expected = (np.ones((1, 3, denoise.TILE, denoise.TILE), np.float32) * 0.25)
    _with_fake_accelerator(expected, monkey)
    try:
        out = denoise._run(tile)
        assert tuple(out.shape) == (1, 3, denoise.TILE, denoise.TILE), out.shape
        # It really used the accelerator's numbers, not the CPU fallback.
        assert abs(float(out.mean()) - 0.25) < 1e-6, float(out.mean())
    finally:
        _restore(monkey)


def test_an_empty_tile_falls_back_instead_of_propagating():
    # The exact shape from the failure: (1,3,0,0), which becomes (0,0,3) by
    # the time the blender multiplies it by a (512,512,1) window.
    monkey = {}
    tile = torch.full((1, 3, denoise.TILE, denoise.TILE), 0.5)
    _with_fake_accelerator(np.zeros((1, 3, 0, 0), np.float32), monkey)
    try:
        out = denoise._run(tile)
        assert tuple(out.shape) == (1, 3, denoise.TILE, denoise.TILE), (
            f"a bad tile escaped the guard: {tuple(out.shape)}")
        # The identity fallback ran, so the tile came back unchanged.
        assert abs(float(out.mean()) - 0.5) < 1e-6, float(out.mean())
    finally:
        _restore(monkey)


def test_a_wrong_sized_tile_also_falls_back():
    monkey = {}
    tile = torch.full((1, 3, denoise.TILE, denoise.TILE), 0.5)
    _with_fake_accelerator(np.zeros((1, 3, 256, 256), np.float32), monkey)
    try:
        out = denoise._run(tile)
        assert tuple(out.shape) == (1, 3, denoise.TILE, denoise.TILE), out.shape
    finally:
        _restore(monkey)


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
