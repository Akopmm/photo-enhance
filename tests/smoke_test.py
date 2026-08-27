"""Call every route the UI calls, with the parameters the UI actually sends.

Exists because a blind string edit twice added an argument to a call site
whose function had no such parameter. Both times the module imported fine,
parsed fine, and only failed at request time -- once shipping a broken editor
preview to production. Parsing is not enough; the routes have to be invoked.

Runs in CI against a temporary data directory, with auth stubbed out. It
asserts on status codes only: it is a wiring check, not a rendering test.
"""
import os
import sys
import tempfile

FAILS = []


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="pe-smoke-")
    os.environ["RENDER_STORAGE_DIR"] = os.path.join(tmp, "renders")
    os.environ.setdefault("PHOTO_ENHANCE_ADMIN_USER", "smoke")
    os.environ.setdefault("PHOTO_ENHANCE_ADMIN_PASSWORD", "smoke-password")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    here = os.path.join(root, "service")
    sys.path.insert(0, here)
    sys.path.insert(0, root)
    # main.py mounts StaticFiles("static") relatively, as it does in the
    # container where /app/service is the working directory.
    os.chdir(here)

    from fastapi.testclient import TestClient

    import auth
    import main as app_module
    import storage

    app = app_module.app
    client = TestClient(app)

    # A dependency override is NOT enough: auth runs as HTTP middleware, ahead
    # of the route's dependencies. Overriding it still returned 401 for every
    # call, so the route bodies never executed and the test passed while the
    # bug it exists to catch was still present. Log in for real instead.
    auth.create_user("smoke", "smoke-password", is_admin=True)
    r = client.post("/api/login", data={"username": "smoke", "password": "smoke-password"})
    if r.status_code != 200:
        print(f"  could not log in for the smoke test: {r.status_code} {r.text[:120]}")
        return 1
    if not client.cookies.get(auth.SESSION_COOKIE):
        print("  logged in but no session cookie was set")
        return 1

    imp = storage.create_import("smoke.cr3", "upload", owner="smoke")
    storage.save_preview(imp, "punch", "Punch", b"")
    storage.save_denoise_info(imp, {"available": True, "sigma": 7.0, "amount": 0.9})

    # (method, path, query/form) -- exactly the shapes index.html sends.
    checks = [
        ("GET", "/api/me", None),
        ("GET", "/api/styles", None),
        ("GET", "/api/sizes", None),
        ("GET", "/api/gallery", None),
        ("GET", f"/api/gallery/{imp}", None),
        # the editor hero: every combination of controls it can send
        ("GET", f"/api/gallery/{imp}/punch_preview.jpg", {}),
        ("GET", f"/api/gallery/{imp}/punch_preview.jpg", {"strength": "0.50"}),
        ("GET", f"/api/gallery/{imp}/punch_preview.jpg", {"denoise": "0.00"}),
        ("GET", f"/api/gallery/{imp}/punch_preview.jpg",
         {"strength": "0.40", "denoise": "0.75"}),
        ("GET", f"/api/gallery/{imp}/punch_thumb.jpg", {"strength": "0.60"}),
        ("GET", f"/api/gallery/{imp}/motion_wiggle.gif", None),
        ("GET", f"/api/gallery/{imp}/motion_turn.gif", None),
        # the download job, with every parameter the editor can attach
        ("POST", "/api/render", {"import_id": imp, "style_key": "punch"}),
        ("POST", "/api/render", {"import_id": imp, "style_key": "punch",
                                 "strength": "0.5", "denoise": "0.8",
                                 "size": "medium", "crop_rect": "0.1,0.1,0.5,0.5"}),
    ]

    for method, path, payload in checks:
        try:
            if method == "GET":
                r = client.get(path, params=payload or {})
            else:
                r = client.post(path, data=payload or {})
        except Exception as e:  # noqa: BLE001
            FAILS.append(f"{method} {path} raised {type(e).__name__}: {e}")
            continue
        # 404/502 are legitimate here (no real image bytes behind the import);
        # 500 means the route itself is broken, which is what we are hunting.
        if r.status_code == 401:
            # Would mean the login above silently stopped working, and every
            # later assertion would be vacuous -- exactly the false pass this
            # test was written to avoid.
            FAILS.append(f"{method} {path} -> 401, the smoke session is not authenticated")
        elif r.status_code >= 500:
            FAILS.append(f"{method} {path} -> {r.status_code} {r.text[:160]}")
        else:
            print(f"  ok  {method:<4} {path}{'?' + str(payload) if payload else ''} -> {r.status_code}")

    storage.delete_import(imp)
    if FAILS:
        print("\nSMOKE FAILURES:")
        for f in FAILS:
            print("  -", f)
        return 1
    print("\nall routes wired correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
