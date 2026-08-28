"""Two renders at once must not collide inside OpenVINO.

A CompiledModel called directly runs on the model's own single InferRequest,
and OpenVINO refuses to enter one twice. Renders overlap by design --
max_concurrent_renders defaults to 2 -- so the second download died with
"Infer Request is busy" while the first was still denoising. The failure is
timing-dependent and leaves no trace in the model output, so only a
concurrent test finds it.

The model here is a deliberately slow scrap of arithmetic, not SCUNet:
converting SCUNet needs more than 12GB (see denoise.py) and the collision is
a property of how the request is used, not of which graph runs on it.
"""
import os
import sys
import threading

import numpy as np
import openvino as ov
import openvino.opset13 as op

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

import ov_infer  # noqa: E402

THREADS = 4


def _slow_model():
    p = op.parameter([1, 3, 256, 256], np.float32, name="x")
    y = p
    for _ in range(8):
        y = op.matmul(y, y, transpose_a=False, transpose_b=True)
    return ov.Core().compile_model(ov.Model([y], [p], "stress"), "CPU")


def _race(work):
    errors = []

    def run():
        try:
            work()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=run) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_concurrent_inference_does_not_collide():
    comp = _slow_model()
    x = (np.random.rand(1, 3, 256, 256) * 0.01).astype(np.float32)
    errors = _race(lambda: ov_infer.infer(comp, "test", x))
    assert not errors, f"concurrent inference failed: {errors}"


def test_the_control_still_fails_the_old_way():
    # Without this the test above proves nothing: if a future OpenVINO made
    # CompiledModel.__call__ thread-safe on its own, the fix would be dead
    # code and no one would know. Should this ever stop failing, the helper
    # can go -- but that must be a decision, not a silent drift.
    comp = _slow_model()
    x = (np.random.rand(1, 3, 256, 256) * 0.01).astype(np.float32)
    errors = _race(lambda: comp(x))
    assert any("busy" in e.lower() for e in errors), (
        f"expected 'Infer Request is busy' from the shared request, got: {errors}")


def test_results_survive_the_next_inference():
    # The request reuses its output buffer, so a result handed back without a
    # copy is quietly overwritten by the following tile.
    comp = _slow_model()
    a = (np.random.rand(1, 3, 256, 256) * 0.01).astype(np.float32)
    b = (np.random.rand(1, 3, 256, 256) * 0.5).astype(np.float32)
    first = ov_infer.infer(comp, "test", a)
    kept = first.copy()
    ov_infer.infer(comp, "test", b)
    assert np.array_equal(first, kept), "an earlier result was overwritten in place"


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
