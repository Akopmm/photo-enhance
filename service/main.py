import logging

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import immich_client
import pipeline
import storage
from model_runtime import runtime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("photo-enhance.main")

app = FastAPI(title="photo-enhance")


@app.middleware("http")
async def no_store_api_responses(request: Request, call_next):
    """All /api/* JSON responses change frequently (gallery contents, Immich
    browse results) and must never be cached by the browser -- a stale cached
    /api/gallery response is indistinguishable from a real one and silently
    shows the wrong imports. Rendered JPEGs under /api/gallery/*/*.jpg are
    immutable once written (each import_id is a fresh uuid), so those are
    left cacheable.
    """
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/api/") and not path.endswith(".jpg"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.on_event("startup")
async def startup():
    runtime.start_watchdog()


@app.get("/health")
async def health():
    return {"status": "ok", "model": runtime.status()}


# ---- Immich browse (server-side proxy; the browser never sees the API key) ----

@app.get("/api/immich/albums")
async def immich_albums():
    try:
        return await immich_client.list_albums()
    except Exception as e:
        raise HTTPException(502, f"could not reach Immich: {e}")


@app.get("/api/immich/search")
async def immich_search(album_id: str | None = None, page: int = 1, size: int = 100):
    try:
        result = await immich_client.search_assets(album_id=album_id, page=page, size=size)
    except Exception as e:
        raise HTTPException(502, f"could not reach Immich: {e}")
    asset_page = result.get("assets", {})
    items = asset_page.get("items", [])
    next_page = asset_page.get("nextPage")
    return {
        "items": [
            {"id": a["id"], "originalFileName": a.get("originalFileName"), "takenAt": a.get("fileCreatedAt")}
            for a in items
        ],
        "nextPage": next_page,
    }


@app.get("/api/immich/thumbnail/{asset_id}")
async def immich_thumbnail(asset_id: str):
    try:
        data, content_type = await immich_client.get_thumbnail_bytes(asset_id)
    except Exception as e:
        raise HTTPException(502, f"could not reach Immich: {e}")
    return Response(content=data, media_type=content_type)


# ---- Ingest ----

@app.post("/api/import/immich/{asset_id}")
async def import_from_immich(asset_id: str):
    try:
        data, filename = await immich_client.get_original_bytes(asset_id)
    except Exception as e:
        raise HTTPException(502, f"could not fetch original from Immich: {e}")
    import_id = await pipeline.process(data, filename)
    return {"import_id": import_id}


@app.post("/api/import/upload")
async def import_upload(file: UploadFile = File(...)):
    data = await file.read()
    import_id = await pipeline.process(data, file.filename)
    return {"import_id": import_id}


# ---- Gallery ----

@app.get("/api/gallery")
async def gallery_list():
    return storage.list_imports()


@app.get("/api/gallery/{import_id}")
async def gallery_get(import_id: str):
    meta = storage.get_import(import_id)
    if not meta:
        raise HTTPException(404, "not found")
    return meta


@app.get("/api/gallery/{import_id}/{style_key}.jpg")
async def gallery_render(import_id: str, style_key: str):
    path = storage.render_path(import_id, style_key)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/gallery/{import_id}/{style_key}_thumb.jpg")
async def gallery_thumb(import_id: str, style_key: str):
    path = storage.thumb_path(import_id, style_key)
    return FileResponse(path, media_type="image/jpeg")


@app.delete("/api/gallery/{import_id}")
async def gallery_delete(import_id: str):
    storage.delete_import(import_id)
    return JSONResponse({"deleted": import_id})


app.mount("/", StaticFiles(directory="static", html=True), name="static")
