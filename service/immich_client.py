"""Thin async client for the Immich REST API, used server-side only -- the
browser UI never sees the Immich API key, it just talks to our own /immich/*
proxy routes.

Endpoints below were verified against a live Immich 3.1.0 instance's real
OpenAPI spec (GET /api/spec.json), not guessed from memory:
  - auth: header "x-api-key"
  - GET  /api/albums                 -> list albums
  - POST /api/search/metadata        -> paginated asset search (by album,
                                         date range, camera make/model, etc.),
                                         returns {assets: {items: [...]}, ...}
  - GET  /api/assets/{id}/thumbnail  -> thumbnail bytes, for browse grids
  - GET  /api/assets/{id}/original   -> original file bytes, untouched
                                         (this is what returns CR3/ARW as-shot)
"""
import os
import re
import urllib.parse

import httpx

IMMICH_URL = os.environ.get("IMMICH_URL", "http://192.168.0.123:2283").rstrip("/")
IMMICH_API_KEY = os.environ.get("IMMICH_API_KEY", "")


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=f"{IMMICH_URL}/api",
        headers={"x-api-key": IMMICH_API_KEY},
        timeout=60.0,
    )


async def list_albums() -> list[dict]:
    async with _client() as c:
        r = await c.get("/albums")
        r.raise_for_status()
        return r.json()


async def search_assets(album_id: str | None = None, page: int = 1, size: int = 60) -> dict:
    body = {"page": page, "size": size, "type": "IMAGE"}
    if album_id:
        body["albumIds"] = [album_id]
    async with _client() as c:
        r = await c.post("/search/metadata", json=body)
        r.raise_for_status()
        return r.json()


async def get_thumbnail_bytes(asset_id: str) -> tuple[bytes, str]:
    async with _client() as c:
        r = await c.get(f"/assets/{asset_id}/thumbnail")
        r.raise_for_status()
        return r.content, r.headers.get("content-type", "image/jpeg")


def _filename_from_content_disposition(cd: str) -> str | None:
    # RFC 5987 form Immich actually sends: filename*=UTF-8''IMG_1662.CR3
    m = re.search(r"filename\*=(?:UTF-8|utf-8)''([^;]+)", cd)
    if m:
        return urllib.parse.unquote(m.group(1))
    # plain form fallback: filename="IMG_1662.CR3"
    m = re.search(r'filename="?([^";]+)"?', cd)
    if m:
        return m.group(1)
    return None


async def get_original_bytes(asset_id: str) -> tuple[bytes, str]:
    async with _client() as c:
        r = await c.get(f"/assets/{asset_id}/original")
        r.raise_for_status()
        filename = _filename_from_content_disposition(r.headers.get("content-disposition", ""))
        return r.content, filename or f"{asset_id}.bin"
