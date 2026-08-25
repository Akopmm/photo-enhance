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

import auth


class NoCredentials(RuntimeError):
    """Raised when the calling user hasn't configured their Immich key yet."""


def creds_for(username: str) -> tuple[str, str]:
    """Each user holds their own Immich URL + API key, so imports read from
    that person's own library and one user's key is never used for another."""
    user = auth.get_user(username) or {}
    url = str(user.get("immich_url") or "").rstrip("/")
    key = user.get("immich_api_key") or ""
    if not url or not key:
        raise NoCredentials("No Immich URL/API key configured for this user (see Settings)")
    return url, key


def _client(username: str) -> httpx.AsyncClient:
    url, key = creds_for(username)
    return httpx.AsyncClient(
        base_url=f"{url}/api",
        headers={"x-api-key": key},
        timeout=60.0,
    )


async def list_albums(username: str) -> list[dict]:
    async with _client(username) as c:
        r = await c.get("/albums")
        r.raise_for_status()
        return r.json()


async def search_assets(username: str, album_id: str | None = None, page: int = 1, size: int = 60) -> dict:
    body = {"page": page, "size": size, "type": "IMAGE"}
    if album_id:
        body["albumIds"] = [album_id]
    async with _client(username) as c:
        r = await c.post("/search/metadata", json=body)
        r.raise_for_status()
        return r.json()


async def get_thumbnail_bytes(username: str, asset_id: str) -> tuple[bytes, str]:
    async with _client(username) as c:
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


async def get_original_bytes(username: str, asset_id: str) -> tuple[bytes, str]:
    async with _client(username) as c:
        r = await c.get(f"/assets/{asset_id}/original")
        r.raise_for_status()
        filename = _filename_from_content_disposition(r.headers.get("content-disposition", ""))
        return r.content, filename or f"{asset_id}.bin"
