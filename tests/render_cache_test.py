"""The download cache key must name everything that changes the file.

Two things it did not name. The denoise amount was only included when the
import's stored flag said denoising was available -- so once an explicit
request could bypass that flag, 0% and 100% collided on one entry and the
second download handed back the first one's file. And the engine was never in
the key at all, so switching between the denoisers and downloading again
returned the render made by the previous one: any comparison between them was
silently a comparison of one against itself.

Neither shows up as an error. The file arrives, it is just the wrong file.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

os.environ.setdefault("RENDER_STORAGE_DIR", __import__("tempfile").mkdtemp())

import denoise as dn  # noqa: E402


def _key(style, amount, method, size="medium", strength=1.0):
    """Rebuild the key the render uses, from the same inputs."""
    key = style
    if strength < 1.0:
        key = f"{key}__s{round(strength * 100):03d}"
    if float(amount) > 0:
        key = f"{key}__d{round(float(amount) * 100):03d}_{method[:1]}"
    return f"{key}__{size}"


def test_amount_changes_the_key():
    assert _key("punch", 0.0, "balanced") != _key("punch", 1.0, "balanced")
    assert _key("punch", 0.3, "balanced") != _key("punch", 0.9, "balanced")


def test_engine_changes_the_key():
    a = _key("punch", 1.0, "balanced")
    b = _key("punch", 1.0, "quality")
    c = _key("punch", 1.0, "fast")
    assert len({a, b, c}) == 3, f"engines share a cache entry: {a} {b} {c}"


def test_the_engines_have_distinct_initials():
    # The key uses the first letter, so two engines starting with the same one
    # would collide and this whole file would be decorative.
    names = ["quality", "balanced", "fast"]
    assert len({n[0] for n in names}) == len(names), f"initials collide: {names}"


def test_the_key_matches_what_the_render_builds():
    # Pinned against the real source, so this file cannot drift into testing
    # a formula the service no longer uses.
    import inspect
    import pipeline
    src = inspect.getsource(pipeline.render_full_style)
    assert 'if float(_dn_amt) > 0:' in src, "the amount is gated again"
    assert "_dn._configured_method()[:1]" in src, "the engine is missing from the key"


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
