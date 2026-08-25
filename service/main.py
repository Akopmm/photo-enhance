import logging

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
async def gallery_thumb(import_id: str, style_key: str, user: str = Depends(current_user)):
    _owned(import_id, user)
    return FileResponse(storage.thumb_path(import_id, style_key), media_type="image/jpeg")


@app.get("/api/gallery/{import_id}/{style_key}.jpg")
async def gallery_render_full(import_id: str, style_key: str, crop: str | None = None,
                              strength: float = 1.0,
                              user: str = Depends(current_user)):
    _owned(import_id, user)
    try:
        path = await pipeline.render_full_style(import_id, style_key, crop_key=crop,
                                                strength=strength)
    except (FileNotFoundError, KeyError):
        raise HTTPException(404, "unknown import or style")
    except Exception as e:
        raise HTTPException(502, f"could not render full-resolution image: {e}")
    return FileResponse(path, media_type="image/jpeg")


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
