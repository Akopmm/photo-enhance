"""End-to-end API test: every capability the service claims, against a live server.

Complements the two narrower suites:
  smoke_test.py    every route the UI calls, in-process, wiring only
  pending_test.py  one photo appears once in the gallery while importing

This one runs against a *running instance* over HTTP, so it exercises the
real middleware, real sessions, real background tasks and the real pipeline —
the things a TestClient quietly bypasses. It creates its own admin, so point
it at a throwaway data directory, never a live one.

    python e2e_test.py                       # against http://127.0.0.1:5055
    python e2e_test.py --base http://host:5054

The photo is generated, not fetched, so the test carries no assets and can
run anywhere.
"""
import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((bool(cond), name, detail))
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{'  — ' + detail if detail and not cond else ''}",
          flush=True)
    return bool(cond)


class Client:
    """One user's session. Separate cookie jars prove isolation is real."""

    def __init__(self, base):
        self.base = base.rstrip("/")
        self.jar = CookieJar()
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))

    def request(self, method, path, data=None, files=None, raw=False):
        url = self.base + path
        body, headers = None, {}
        if files:
            boundary = "----e2e"
            buf = io.BytesIO()
            for k, v in (data or {}).items():
                buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
            for k, (fn, content, ctype) in files.items():
                buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fn}\"\r\n"
                          f"Content-Type: {ctype}\r\n\r\n".encode())
                buf.write(content); buf.write(b"\r\n")
            buf.write(f"--{boundary}--\r\n".encode())
            body = buf.getvalue()
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        elif data is not None:
            body = urllib.parse.urlencode(data).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with self.op.open(req, timeout=600) as r:
                payload = r.read()
                return r.status, (payload if raw else _maybe_json(payload))
        except urllib.error.HTTPError as e:
            return e.code, _maybe_json(e.read())

    def get(self, p, **kw):  return self.request("GET", p, **kw)
    def post(self, p, **kw): return self.request("POST", p, **kw)
    def delete(self, p):     return self.request("DELETE", p)


def _maybe_json(b):
    try:
        return json.loads(b)
    except Exception:
        return b[:200]


def make_photo():
    """A deterministic JPEG with a clear subject, so segmentation has something
    to find and the region recipes are actually exercised."""
    import numpy as np
    from PIL import Image
    h, w = 900, 1350
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.float32)
    img[..., 2] = 0.55 - 0.25 * (yy / h)          # sky gradient at the top
    img[..., 1] = 0.35 + 0.25 * (yy / h)
    img[..., 0] = 0.30
    ground = yy > h * 0.62
    img[ground] = (0.22, 0.34, 0.15)              # foliage-ish
    blob = ((yy - h * 0.62) ** 2 / (h * 0.26) ** 2 + (xx - w * 0.5) ** 2 / (w * 0.13) ** 2) < 1
    img[blob] = (0.80, 0.72, 0.60)                # a subject
    rng = np.random.default_rng(11)
    img = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)
    buf = io.BytesIO()
    Image.fromarray((img * 255).astype("uint8"), "RGB").save(buf, "JPEG", quality=94)
    return buf.getvalue()


def wait_for_import(c, timeout=600):
    end = time.time() + timeout
    while time.time() < end:
        st, g = c.get("/api/gallery")
        if isinstance(g, list):
            done = [i for i in g if not i.get("pending")]
            if done:
                return done[0]
        time.sleep(3)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5055")
    a = ap.parse_args()
    ADMIN, APW = "e2e_admin", "e2e-admin-password"
    OTHER, OPW = "e2e_other", "e2e-other-password"
    admin, other, anon = Client(a.base), Client(a.base), Client(a.base)

    print("\n-- ops --")
    st, h = admin.get("/health")
    check("GET /health responds 200", st == 200, str(st))
    check("/health reports a mode", isinstance(h, dict) and "mode" in h, str(h)[:80])

    print("\n-- first run and auth --")
    st, b = anon.get("/api/bootstrap-state")
    fresh = isinstance(b, dict) and b.get("needs_setup")
    check("a fresh instance asks for setup", fresh, f"needs_setup={b}")
    if not fresh:
        print("     (instance already has users — point --base at a throwaway one)")
        return 2
    st, _ = anon.get("/api/gallery")
    check("gallery refuses an anonymous caller", st == 401, f"got {st}")
    st, _ = admin.post("/api/setup", data={"username": ADMIN, "password": APW})
    check("POST /api/setup creates the first admin", st == 200, str(st))
    st, _ = Client(a.base).post("/api/setup", data={"username": "sneak", "password": "x" * 12})
    check("a second setup is refused", st >= 400, f"got {st}")
    st, _ = Client(a.base).post("/api/login", data={"username": ADMIN, "password": "wrong"})
    check("login rejects a bad password", st == 401, f"got {st}")
    st, me = admin.get("/api/me")
    check("/api/me reports the admin", isinstance(me, dict) and me.get("is_admin"), str(me)[:90])
    check("no Immich key configured yet", me.get("immich_api_key_set") is False)

    print("\n-- immich credentials --")
    st, me2 = admin.post("/api/me/immich", data={"immich_url": "http://immich.invalid:2283",
                                                 "immich_api_key": "SECRET-KEY-VALUE-1234567890"})
    check("saving Immich credentials succeeds", st == 200, str(st))
    body = json.dumps(me2)
    check("the API key is never echoed back", "SECRET-KEY-VALUE-1234567890" not in body)
    check("only a masked preview is returned", bool(me2.get("immich_api_key_preview")), str(me2)[:90])
    st, _ = admin.get("/api/immich/albums")
    check("an unreachable Immich fails cleanly, not 500", st in (400, 502), f"got {st}")

    print("\n-- users and permissions --")
    st, _ = admin.post("/api/users", data={"username": OTHER, "password": OPW})
    check("admin can create a second user", st == 200, str(st))
    st, _ = other.post("/api/login", data={"username": OTHER, "password": OPW})
    check("the second user can log in", st == 200, str(st))
    st, u = other.get("/api/users")
    check("a non-admin cannot list users", st == 403, f"got {st}")
    st, _ = other.post("/api/settings", data={"jpeg_quality": "50"})
    check("a non-admin cannot change instance settings", st == 403, f"got {st}")

    print("\n-- settings --")
    st, s = admin.get("/api/settings")
    check("admin can read settings", st == 200 and isinstance(s, dict), str(st))
    st, _ = admin.post("/api/settings", data={"jpeg_quality": "91"})
    st, s2 = admin.get("/api/settings")
    check("a settings change persists", s2.get("jpeg_quality") == 91, f"got {s2.get('jpeg_quality')}")
    # Regression: a partial POST used to switch every masking toggle off,
    # because an absent checkbox was read as "unchecked" rather than "not in
    # this request". Enhanced mode then silently produced no masks at all.
    untouched = all(s2.get(k) == s.get(k) for k in
                    ("subject_masking", "sky_masking", "depth_masking", "cinematic_presets"))
    check("a partial settings update leaves other toggles alone", untouched,
          f"before={[s.get(k) for k in ('subject_masking','sky_masking','depth_masking','cinematic_presets')]} "
          f"after={[s2.get(k) for k in ('subject_masking','sky_masking','depth_masking','cinematic_presets')]}")
    # And a real form submit must still be able to turn one off.
    st, _ = admin.post("/api/settings", data={"_checkbox_fields": "subject_masking,sky_masking",
                                              "subject_masking": "on"})
    st, s3 = admin.get("/api/settings")
    check("a full form submit can still switch a toggle off",
          s3.get("subject_masking") is True and s3.get("sky_masking") is False,
          f"subject={s3.get('subject_masking')} sky={s3.get('sky_masking')}")
    admin.post("/api/settings", data={
        "jpeg_quality": str(s.get("jpeg_quality", 95)),
        "_checkbox_fields": "subject_masking,sky_masking,depth_masking,cinematic_presets",
        **{k: "on" for k in ("subject_masking", "sky_masking", "depth_masking", "cinematic_presets")
           if s.get(k)}})

    print("\n-- catalogue endpoints --")
    st, styles = admin.get("/api/styles")
    check("/api/styles lists looks", st == 200 and len(styles) > 0, f"{st}, {len(styles) if hasattr(styles,'__len__') else '?'}")
    st, sizes = admin.get("/api/sizes")
    names = [x.get("key") for x in sizes] if isinstance(sizes, list) else []
    check("/api/sizes offers all four presets",
          set(names) >= {"original", "large", "medium", "small"}, str(names))

    print("\n-- ingest --")
    st, r = admin.post("/api/import/upload", files={"file": ("e2e.jpg", make_photo(), "image/jpeg")})
    check("upload is accepted and queued", st == 200 and isinstance(r, dict) and r.get("queued"), str(r)[:90])
    st, g = admin.get("/api/gallery")
    check("it appears immediately as pending", any(i.get("pending") for i in g) if isinstance(g, list) else False)
    imp = wait_for_import(admin)
    if not check("the import finishes", imp is not None):
        return 1
    iid = imp["id"]
    check("styles were produced", len(imp.get("styles") or []) > 0, f"{len(imp.get('styles') or [])}")
    check("it appears exactly once when finished",
          len([i for i in admin.get('/api/gallery')[1] if i.get('id') == iid]) == 1)

    print("\n-- isolation --")
    st, og = other.get("/api/gallery")
    check("another user's gallery is empty", og == [], str(og)[:80])
    st, _ = other.get(f"/api/gallery/{iid}")
    check("another user cannot open the import", st == 404, f"got {st}")
    st, _ = other.get(f"/api/gallery/{iid}/original_thumb.jpg")
    check("another user cannot fetch its thumbnail", st == 404, f"got {st}")

    print("\n-- editor surfaces --")
    style = (imp.get("styles") or [{}])[0].get("key")
    st, body = admin.get(f"/api/gallery/{iid}/original_thumb.jpg", raw=True)
    check("original thumbnail renders", st == 200 and body[:2] == b"\xff\xd8", f"{st}")
    st, body = admin.get(f"/api/gallery/{iid}/{style}_thumb.jpg", raw=True)
    check("look thumbnail renders", st == 200 and body[:2] == b"\xff\xd8", f"{st}")
    st, body = admin.get(f"/api/gallery/{iid}/{style}_preview.jpg", raw=True)
    check("editor preview renders", st == 200 and body[:2] == b"\xff\xd8", f"{st}")
    st, b1 = admin.get(f"/api/gallery/{iid}/{style}_preview.jpg?strength=0.20", raw=True)
    st2, b2 = admin.get(f"/api/gallery/{iid}/{style}_preview.jpg?strength=1.00", raw=True)
    check("the strength slider changes the image", st == 200 and st2 == 200 and b1 != b2)
    st, b3 = admin.get(f"/api/gallery/{iid}/{style}_preview.jpg?denoise=0.00", raw=True)
    st2, b4 = admin.get(f"/api/gallery/{iid}/{style}_preview.jpg?denoise=1.00", raw=True)
    check("the denoise slider is accepted", st == 200 and st2 == 200, f"{st}/{st2}")

    print("\n-- render and download --")
    st, job = admin.post("/api/render", data={"import_id": iid, "style_key": style,
                                              "size": "small", "strength": "1.0"})
    jid = job.get("job_id") if isinstance(job, dict) else None
    check("a render job is accepted", st == 200 and jid, str(job)[:90])
    done, last = False, None
    end = time.time() + 900
    while jid and time.time() < end:
        st, j = admin.get(f"/api/render/{jid}")
        last = j
        if isinstance(j, dict) and j.get("state") in ("done", "error", "failed", "ready"):
            done = j.get("state") in ("done", "ready"); break
        time.sleep(2)
    check("the render completes", done, str(last)[:110])
    st, out = admin.get(f"/api/gallery/{iid}/{style}.jpg?size=small", raw=True)
    check("the rendered file downloads as a JPEG",
          st == 200 and out[:2] == b"\xff\xd8", f"{st}, {len(out) if isinstance(out,bytes) else '?'} bytes")

    print("\n-- crops --")
    st, cropped = admin.get(f"/api/gallery/{iid}/{style}_preview.jpg?crop_rect=0.1,0.1,0.6,0.6", raw=True)
    check("a crop rectangle is applied", st == 200 and cropped[:2] == b"\xff\xd8", f"{st}")

    print("\n-- delete --")
    st, _ = other.delete(f"/api/gallery/{iid}")
    check("another user cannot delete it", st == 404, f"got {st}")
    st, _ = admin.delete(f"/api/gallery/{iid}")
    check("the owner can delete it", st == 200, f"got {st}")
    st, g = admin.get("/api/gallery")
    check("it is gone from the gallery", all(i.get("id") != iid for i in g) if isinstance(g, list) else False)

    print("\n-- session --")
    st, _ = admin.post("/api/me/password", data={"current": APW, "new": APW + "2"})
    check("password change works", st == 200, f"got {st}")
    st, _ = admin.post("/api/logout")
    st, _ = admin.get("/api/gallery")
    check("after logout the session is dead", st == 401, f"got {st}")
    st, _ = admin.post("/api/login", data={"username": ADMIN, "password": APW + "2"})
    check("the new password works", st == 200, f"got {st}")

    passed = sum(1 for okc, _, _ in RESULTS if okc)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    for okc, name, detail in RESULTS:
        if not okc:
            print(f"  FAILED: {name}  {detail}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
