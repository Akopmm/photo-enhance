"""Persistent gallery storage.

Renders are kept indefinitely (no auto-expiry) until deleted via the UI.

Preview-first design: at import time we only ever save small preview JPEGs
(fast, cheap). A full-resolution render for a given style is generated
on-demand the first time it's requested (see pipeline.render_full_style)
and cached here afterwards. To regenerate at full-res later we need the
original bytes again:
  - Immich imports: nothing extra to store, just the asset_id -- refetch
    from Immich on demand, it already keeps the original forever.
  - Plain uploads: nothing else has a copy, so the uploaded original is
    persisted here (source.<ext>) until the import is deleted.
"""
import json
import os
import shutil
import time
import uuid

ROOT = os.environ.get("RENDER_STORAGE_DIR", os.path.join(os.path.dirname(__file__), "data", "renders"))


def _import_dir(import_id: str) -> str:
    return os.path.join(ROOT, import_id)


def create_import(source_name: str, source_type: str, immich_asset_id: str | None = None,
                  owner: str = "") -> str:
    """source_type: 'immich' or 'upload'. `owner` scopes the import to one
    user -- galleries are per-user, and every read path checks ownership
    (see owns()) rather than relying on uuids being unguessable."""
    import_id = uuid.uuid4().hex
    d = _import_dir(import_id)
    os.makedirs(d, exist_ok=True)
    meta = {
        "id": import_id,
        "owner": owner,
        "source_name": source_name,
        "source_type": source_type,
        "immich_asset_id": immich_asset_id,
        "created_at": time.time(),
        "styles": [],
    }
    _write_meta(import_id, meta)
    return import_id


def _write_meta(import_id: str, meta: dict):
    with open(os.path.join(_import_dir(import_id), "meta.json"), "w") as f:
        json.dump(meta, f)


def _read_meta(import_id: str) -> dict | None:
    path = os.path.join(_import_dir(import_id), "meta.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_source_upload(import_id: str, raw_bytes: bytes, filename: str):
    ext = os.path.splitext(filename)[1]
    with open(os.path.join(_import_dir(import_id), f"source{ext}"), "wb") as f:
        f.write(raw_bytes)
    meta = _read_meta(import_id)
    meta["source_upload_ext"] = ext
    _write_meta(import_id, meta)


def get_source_upload_bytes(import_id: str) -> tuple[bytes, str] | None:
    """Returns (bytes, filename) for a persisted upload, or None if this
    import wasn't an upload (e.g. it's an Immich import instead)."""
    meta = _read_meta(import_id)
    ext = meta.get("source_upload_ext")
    if not ext:
        return None
    path = os.path.join(_import_dir(import_id), f"source{ext}")
    with open(path, "rb") as f:
        return f.read(), meta["source_name"]


def save_original_thumb(import_id: str, thumb_bytes: bytes):
    with open(os.path.join(_import_dir(import_id), "original_thumb.jpg"), "wb") as f:
        f.write(thumb_bytes)


def original_thumb_path(import_id: str) -> str:
    return os.path.join(_import_dir(import_id), "original_thumb.jpg")


def save_preview(import_id: str, style_key: str, style_label: str, thumb_bytes: bytes):
    with open(os.path.join(_import_dir(import_id), f"{style_key}_thumb.jpg"), "wb") as f:
        f.write(thumb_bytes)
    meta = _read_meta(import_id)
    if not any(s["key"] == style_key for s in meta["styles"]):
        meta["styles"].append({"key": style_key, "label": style_label})
    _write_meta(import_id, meta)


def save_full_render(import_id: str, style_key: str, jpeg_bytes: bytes):
    with open(os.path.join(_import_dir(import_id), f"{style_key}.jpg"), "wb") as f:
        f.write(jpeg_bytes)


def full_render_path(import_id: str, style_key: str) -> str:
    return os.path.join(_import_dir(import_id), f"{style_key}.jpg")


def full_render_exists(import_id: str, style_key: str) -> bool:
    return os.path.exists(full_render_path(import_id, style_key))


def thumb_path(import_id: str, style_key: str) -> str:
    return os.path.join(_import_dir(import_id), f"{style_key}_thumb.jpg")


def get_import(import_id: str) -> dict | None:
    return _read_meta(import_id)


def list_imports(owner: str | None = None) -> list[dict]:
    if not os.path.isdir(ROOT):
        return []
    items = []
    for name in os.listdir(ROOT):
        meta = _read_meta(name)
        if not meta:
            continue
        if owner is not None and meta.get("owner", "") != owner:
            continue
        items.append(meta)
    items.sort(key=lambda m: m["created_at"], reverse=True)
    return items


def save_mask(import_id: str, name: str, png_bytes: bytes):
    """Masks are computed once at import and reused for the full-resolution
    render, so the download grades through exactly the mask the preview was
    judged on. Stored as PNG bytes; masking.encode/decode own the format."""
    with open(os.path.join(_import_dir(import_id), f"mask_{name}.png"), "wb") as f:
        f.write(png_bytes)


def load_mask(import_id: str, name: str) -> bytes | None:
    path = os.path.join(_import_dir(import_id), f"mask_{name}.png")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def save_scene(import_id: str, scene: dict, recipes: list):
    """`recipes` is the list of region-recipe keys that were actually offered
    for this import. Persisting it is what stops the download path from
    re-deciding availability and 500ing on a style the gallery is showing."""
    meta = _read_meta(import_id)
    if not meta:
        return
    meta["scene"] = scene
    meta["region_recipes"] = recipes
    _write_meta(import_id, meta)


def save_crop_thumb(import_id: str, crop_key: str, jpeg_bytes: bytes):
    with open(os.path.join(_import_dir(import_id), f"crop_{crop_key}_thumb.jpg"), "wb") as f:
        f.write(jpeg_bytes)


def crop_thumb_path(import_id: str, crop_key: str) -> str:
    return os.path.join(_import_dir(import_id), f"crop_{crop_key}_thumb.jpg")


def save_crops(import_id: str, crops: list, ref_w: int, ref_h: int):
    """Crop suggestions are computed against the preview, so the reference
    dimensions travel with them -- callers scale to whatever resolution they
    are actually cropping."""
    meta = _read_meta(import_id)
    if not meta:
        return
    meta["crops"] = crops
    meta["crop_ref"] = {"w": ref_w, "h": ref_h}
    _write_meta(import_id, meta)


def owns(import_id: str, username: str) -> bool:
    meta = _read_meta(import_id)
    return bool(meta) and meta.get("owner", "") == username


def claim_unowned(username: str) -> int:
    """Assign imports that predate multi-user support to `username`.

    Galleries are filtered by owner, so without this, upgrading an existing
    single-user instance would make every previously-imported photo silently
    disappear -- the files are still there, they just match nobody. Runs once
    at startup for the first admin. Returns how many were claimed.
    """
    if not os.path.isdir(ROOT) or not username:
        return 0
    claimed = 0
    for name in os.listdir(ROOT):
        meta = _read_meta(name)
        if meta and not meta.get("owner"):
            meta["owner"] = username
            _write_meta(name, meta)
            claimed += 1
    return claimed


def delete_import(import_id: str):
    d = _import_dir(import_id)
    if os.path.isdir(d):
        shutil.rmtree(d)
