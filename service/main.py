import logging

import asyncio
import time
import uuid
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

import auth
import immich_client
import pipeline
import settings
import storage
from model_runtime import runtime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("photo-enhance.main")

app = FastAPI(title="photo-enhance")

# Routes reachable without a session. Everything else requires login.
PUBLIC_PATHS = {"/health", "/login", "/api/login", "/api/setup", "/api/bootstrap-state"}
PUBLIC_PREFIXES = ("/static/",)


@app.on_event("startup")
async def startup():
    auth.bootstrap_from_env()
    # Upgrading a pre-multi-user instance: imports written before ownership
    # existed match no user, so the gallery would look empty even though
    # every file is intact. Hand them to the first admin.
    admins = [u["username"] for u in auth.list_users() if u["is_admin"]]
    if admins:
        n = storage.claim_unowned(admins[0])
        if n:
            logger.info("claimed %d pre-existing import(s) for admin %s", n, admins[0])
    runtime.start_watchdog()


def current_user(request: Request) -> str:
    user = auth.read_session(request.cookies.get(auth.SESSION_COOKIE, ""))
    if not user:
        raise HTTPException(401, "not authenticated")
    return user


def require_admin(request: Request) -> str:
    user = current_user(request)
    u = auth.get_user(user) or {}
    if not u.get("is_admin"):
        raise HTTPException(403, "admin only")
    return user


@app.middleware("http")
async def auth_and_cache(request: Request, call_next):
    path = request.url.path

    is_public = path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)
    if not is_public:
        user = auth.read_session(request.cookies.get(auth.SESSION_COOKIE, ""))
        if not user:
            # Browsers navigating to a page get redirected to the login
            # screen; API calls get a clean 401 to handle in JS.
            if path.startswith("/api/"):
                return JSONResponse({"detail": "not authenticated"}, status_code=401)
            return RedirectResponse("/login", status_code=302)

    response = await call_next(request)
    # /api/* JSON changes constantly (gallery contents, settings) and a stale
    # cached response is indistinguishable from a real one. Rendered JPEGs
    # are immutable per import_id, so those stay cacheable.
    if path.startswith("/api/") and not path.endswith(".jpg"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": runtime.status(),
        "mode": settings.get("mode"),
        "max_concurrent_jobs": settings.get("max_concurrent_jobs"),
        "max_concurrent_renders": settings.get("max_concurrent_renders"),
        "users_configured": auth.any_users(),
    }


# ---- auth ----

@app.get("/api/bootstrap-state")
async def bootstrap_state():
    """Lets the login page show a first-run 'create admin' form instead of a
    login box when the instance has no users yet."""
    return {"needs_setup": not auth.any_users()}


@app.post("/api/setup")
async def first_run_setup(username: str = Form(...), password: str = Form(...)):
    # Only valid while the instance has no users -- otherwise anyone could
    # create themselves an admin account.
    if auth.any_users():
        raise HTTPException(409, "already initialised")
    if len(password) < 8:
        raise HTTPException(400, "password must be at least 8 characters")
    auth.create_user(username, password, is_admin=True)
    # The startup migration runs before any user exists, so on a first-run
    # form signup it finds no admin to hand legacy imports to. Claim them
    # here too, or an upgraded single-user instance shows an empty gallery
    # while every file is still on disk.
    claimed = storage.claim_unowned(username.strip().lower())
    if claimed:
        logger.info("claimed %d pre-existing import(s) for new admin %s", claimed, username)
    return _login_response(username)


@app.post("/api/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if not auth.verify(username, password):
        raise HTTPException(401, "invalid username or password")
    return _login_response(username)


def _login_response(username: str) -> JSONResponse:
    resp = JSONResponse({"ok": True, "username": username.lower()})
    resp.set_cookie(
        auth.SESSION_COOKIE, auth.issue_session(username),
        max_age=auth.SESSION_TTL, httponly=True, samesite="lax",
    )
    return resp


@app.post("/api/logout")
async def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(auth.SESSION_COOKIE)
    return resp


@app.get("/api/me")
async def me(user: str = Depends(current_user)):
    u = auth.get_user(user) or {}
    return {
        "username": user,
        "is_admin": bool(u.get("is_admin")),
        "immich_url": u.get("immich_url", ""),
        "immich_api_key_set": bool(u.get("immich_api_key")),
        "immich_api_key_preview": auth._mask(u.get("immich_api_key", "")),
    }


@app.post("/api/me/immich")
async def set_my_immich(immich_url: str = Form(""), immich_api_key: str = Form(""),
                        user: str = Depends(current_user)):
    auth.update_user_immich(user, immich_url or None, immich_api_key or None)
    return await me(user)


@app.post("/api/me/password")
async def change_my_password(current: str = Form(...), new: str = Form(...),
                             user: str = Depends(current_user)):
    if not auth.verify(user, current):
        raise HTTPException(401, "current password is incorrect")
    if len(new) < 8:
        raise HTTPException(400, "password must be at least 8 characters")
    auth.set_password(user, new)
    return {"ok": True}


# ---- settings (instance-wide; admin only) ----

@app.get("/api/settings")
async def get_settings(user: str = Depends(current_user)):
    data = settings.public_dict()
    data["is_admin"] = bool((auth.get_user(user) or {}).get("is_admin"))
    return data


@app.post("/api/settings")
async def post_settings(request: Request, user: str = Depends(require_admin)):
    form = dict(await request.form())
    for flag in ("subject_masking", "sky_masking", "depth_masking", "cinematic_presets"):
        form[flag] = flag in form  # unchecked checkboxes simply aren't submitted
    return settings.update(form)


@app.get("/api/users")
async def get_users(user: str = Depends(require_admin)):
    return auth.list_users()


@app.post("/api/users")
async def add_user(username: str = Form(...), password: str = Form(...),
                   is_admin: str = Form(""), user: str = Depends(require_admin)):
    if len(password) < 8:
        raise HTTPException(400, "password must be at least 8 characters")
    if not auth.create_user(username, password, is_admin=bool(is_admin)):
        raise HTTPException(409, "that username already exists")
    return auth.list_users()


@app.delete("/api/users/{username}")
async def remove_user(username: str, user: str = Depends(require_admin)):
    if not auth.delete_user(username):
        raise HTTPException(400, "cannot delete (unknown user, or the last admin)")
    return auth.list_users()


# ---- Immich browse (server-side proxy; the browser never sees the API key) ----

@app.get("/api/immich/albums")
async def immich_albums(user: str = Depends(current_user)):
    try:
        return await immich_client.list_albums(user)
    except immich_client.NoCredentials as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"could not reach Immich: {e}")


@app.get("/api/immich/search")
async def immich_search(album_id: str | None = None, page: int = 1, size: int = 100,
                        user: str = Depends(current_user)):
    try:
        result = await immich_client.search_assets(user, album_id=album_id, page=page, size=size)
    except immich_client.NoCredentials as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"could not reach Immich: {e}")
    page_data = result.get("assets", {})
    return {
        "items": [
            {"id": a["id"], "originalFileName": a.get("originalFileName"),
             "takenAt": a.get("fileCreatedAt")}
            for a in page_data.get("items", [])
        ],
        "nextPage": page_data.get("nextPage"),
    }


@app.get("/api/immich/thumbnail/{asset_id}")
async def immich_thumbnail(asset_id: str, user: str = Depends(current_user)):
    try:
        data, content_type = await immich_client.get_thumbnail_bytes(user, asset_id)
    except immich_client.NoCredentials as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"could not reach Immich: {e}")
    return Response(content=data, media_type=content_type)


# ---- ingest ----

@app.post("/api/import/immich/{asset_id}")
async def import_from_immich(asset_id: str, user: str = Depends(current_user)):
    # Admission is taken BEFORE the fetch, so 30 queued clicks hold 30 queue
    # slots rather than 30 full-resolution originals in memory.
    async with pipeline.admission():
        try:
            data, filename = await immich_client.get_original_bytes(user, asset_id)
        except immich_client.NoCredentials as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(502, f"could not fetch original from Immich: {e}")
        import_id = await pipeline.process_preview(
            data, filename, source_type="immich", username=user, immich_asset_id=asset_id)
    return {"import_id": import_id}


@app.post("/api/import/upload")
async def import_upload(file: UploadFile = File(...), user: str = Depends(current_user)):
    async with pipeline.admission():
        data = await file.read()
        import_id = await pipeline.process_preview(
            data, file.filename, source_type="upload", username=user)
    return {"import_id": import_id}


# ---- gallery (scoped to the signed-in user) ----

def _owned(import_id: str, user: str):
    if not storage.owns(import_id, user):
        # 404 rather than 403 -- don't confirm that someone else's import
        # exists to a user who isn't allowed to see it.
        raise HTTPException(404, "not found")


@app.get("/api/gallery")
async def gallery_list(user: str = Depends(current_user)):
    return storage.list_imports(owner=user)


@app.get("/api/gallery/{import_id}")
async def gallery_get(import_id: str, user: str = Depends(current_user)):
    _owned(import_id, user)
    return storage.get_import(import_id)


@app.get("/api/gallery/{import_id}/original_thumb.jpg")
async def gallery_original_thumb(import_id: str, user: str = Depends(current_user)):
    _owned(import_id, user)
    return FileResponse(storage.original_thumb_path(import_id), media_type="image/jpeg")


@app.get("/api/gallery/{import_id}/crop_{crop_key}_thumb.jpg")
async def gallery_crop_thumb(import_id: str, crop_key: str, user: str = Depends(current_user)):
    _owned(import_id, user)
    return FileResponse(storage.crop_thumb_path(import_id, crop_key), media_type="image/jpeg")


@app.get("/api/gallery/{import_id}/{style_key}_thumb.jpg")
async def gallery_thumb(import_id: str, style_key: str, strength: float = 1.0,
                        user: str = Depends(current_user)):
    _owned(import_id, user)
    # A strength below 1 re-composites the preview live from the stored
    # baseline + masks (cheap at 480px), so the slider actually shows what it
    # will do instead of silently only affecting the download.
    if strength < 1.0:
        data = await pipeline.render_preview_at_strength(import_id, style_key, strength)
        if data is not None:
            return Response(content=data, media_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})
    return FileResponse(storage.thumb_path(import_id, style_key), media_type="image/jpeg")


@app.get("/api/styles")
async def styles_catalogue(user: str = Depends(current_user)):
    """Which looks exist and how they group. The editor shows one group at a
    time -- 24 thumbnails at once is a wall, and the groups mean something:
    `smart` looks are computed for THIS photo's content, the rest are fixed."""
    from cinematic import CINEMATIC
    from presets import PRESETS
    cine = {k for k, _l, _p in CINEMATIC}
    out = []
    for key, label, _params in pipeline._style_list():
        out.append({"key": key, "label": label,
                    "group": "cinematic" if key in cine else "signature"})
    return JSONResponse(out)


@app.get("/api/gallery/{import_id}/{style_key}_preview.jpg")
async def gallery_preview(import_id: str, style_key: str, strength: float = 1.0,
                          denoise: float | None = None,
                          user: str = Depends(current_user)):
    """Big editor preview: always rendered live from the stored baseline, so
    it reflects the current look AND strength immediately. Falls back to the
    small stored thumbnail for imports made before baselines were kept."""
    _owned(import_id, user)
    data = await pipeline.render_preview_at_strength(import_id, style_key, strength,
                                                    denoise_amount=denoise)
    if data is not None:
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    return FileResponse(storage.thumb_path(import_id, style_key), media_type="image/jpeg")


@app.get("/api/gallery/{import_id}/{style_key}.jpg")
async def gallery_render_full(import_id: str, style_key: str, crop: str | None = None,
                              strength: float = 1.0, crop_rect: str | None = None,
                              denoise: float | None = None,
                              user: str = Depends(current_user)):
    _owned(import_id, user)
    try:
        path = await pipeline.render_full_style(import_id, style_key, crop_key=crop,
                                                strength=strength, crop_rect=crop_rect,
                                                denoise_amount=denoise)
    except (FileNotFoundError, KeyError):
        raise HTTPException(404, "unknown import or style")
    except Exception as e:
        raise HTTPException(502, f"could not render full-resolution image: {e}")
    return FileResponse(path, media_type="image/jpeg")


# ---- render jobs ----
#
# A full-resolution render takes ~12s, and a plain blocking GET can report
# nothing while it happens -- the UI could only spin. These endpoints start
# the work and let the page poll for real stages. Jobs live in memory: this
# is a single-process service and a lost job on restart just means clicking
# download again, which is cheaper than persisting them.
_render_jobs: dict[str, dict] = {}
_RENDER_JOB_TTL = 900


def _reap_jobs():
    now = time.time()
    for jid in [j for j, v in _render_jobs.items() if now - v["updated"] > _RENDER_JOB_TTL]:
        _render_jobs.pop(jid, None)


@app.post("/api/render")
async def render_start(import_id: str = Form(...), style_key: str = Form(...),
                       crop: str | None = Form(None), strength: float = Form(1.0),
                       crop_rect: str | None = Form(None),
                       denoise: float | None = Form(None),
                       user: str = Depends(current_user)):
    _owned(import_id, user)
    _reap_jobs()
    job_id = uuid.uuid4().hex
    job = {"state": "queued", "pct": 0, "message": "Starting", "owner": user,
           "updated": time.time(), "url": None, "error": None}
    _render_jobs[job_id] = job

    def on_progress(pct, message):
        job.update(pct=pct, message=message, state="running", updated=time.time())

    async def run():
        try:
            await pipeline.render_full_style(import_id, style_key, crop_key=crop,
                                             strength=strength, progress=on_progress,
                                             crop_rect=crop_rect, denoise_amount=denoise)
            job.update(state="done", pct=100, message="Ready", updated=time.time(),
                       url=f"/api/gallery/{import_id}/{style_key}.jpg"
                           + _render_query(crop, strength, crop_rect, denoise))
        except (FileNotFoundError, KeyError) as e:
            job.update(state="error", error=f"unknown import or style: {e}", updated=time.time())
        except Exception as e:  # noqa: BLE001
            job.update(state="error", error=str(e), updated=time.time())

    asyncio.create_task(run())
    return JSONResponse({"job_id": job_id})


def _render_query(crop, strength, crop_rect=None, denoise=None):
    params = []
    if denoise is not None:
        params.append(f"denoise={denoise:.2f}")
    if crop_rect:
        params.append(f"crop_rect={crop_rect}")
    elif crop:
        params.append(f"crop={crop}")
    if strength is not None and strength < 1.0:
        params.append(f"strength={strength:.2f}")
    return ("?" + "&".join(params)) if params else ""


@app.get("/api/render/{job_id}")
async def render_status(job_id: str, user: str = Depends(current_user)):
    job = _render_jobs.get(job_id)
    if not job or job["owner"] != user:
        raise HTTPException(404, "unknown job")
    return JSONResponse({k: job[k] for k in ("state", "pct", "message", "url", "error")})


@app.delete("/api/gallery/{import_id}")
async def gallery_delete(import_id: str, user: str = Depends(current_user)):
    _owned(import_id, user)
    storage.delete_import(import_id)
    return JSONResponse({"deleted": import_id})


# ---- pages ----

@app.get("/login")
async def login_page():
    return FileResponse("static/login.html")


@app.get("/settings")
async def settings_page():
    return FileResponse("static/settings.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")
