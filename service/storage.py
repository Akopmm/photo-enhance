"""Persistent gallery storage for rendered JPEGs. Renders are kept
indefinitely (per project decision -- no auto-expiry); the user deletes via
the UI when they want. Only the small rendered JPEGs persist here, not the
original RAW/JPEG source -- that's decoded, processed, and discarded
in-memory per request (see pipeline.py), keeping disk usage to the renders
only.
"""
import json
import os
import shutil
import time
import uuid

ROOT = os.environ.get("RENDER_STORAGE_DIR", os.path.join(os.path.dirname(__file__), "data", "renders"))


def _import_dir(import_id: str) -> str:
    return os.path.join(ROOT, import_id)


def create_import(source_name: str) -> str:
    import_id = uuid.uuid4().hex
    d = _import_dir(import_id)
    os.makedirs(d, exist_ok=True)
    meta = {"id": import_id, "source_name": source_name, "created_at": time.time(), "styles": []}
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


def save_render(import_id: str, style_key: str, style_label: str, jpeg_bytes: bytes, thumb_bytes: bytes):
    d = _import_dir(import_id)
    with open(os.path.join(d, f"{style_key}.jpg"), "wb") as f:
        f.write(jpeg_bytes)
    with open(os.path.join(d, f"{style_key}_thumb.jpg"), "wb") as f:
        f.write(thumb_bytes)
    meta = _read_meta(import_id)
    if not any(s["key"] == style_key for s in meta["styles"]):
        meta["styles"].append({"key": style_key, "label": style_label})
    _write_meta(import_id, meta)


def render_path(import_id: str, style_key: str) -> str:
    return os.path.join(_import_dir(import_id), f"{style_key}.jpg")


def thumb_path(import_id: str, style_key: str) -> str:
    return os.path.join(_import_dir(import_id), f"{style_key}_thumb.jpg")


def get_import(import_id: str) -> dict | None:
    return _read_meta(import_id)


def list_imports() -> list[dict]:
    if not os.path.isdir(ROOT):
        return []
    items = []
    for name in os.listdir(ROOT):
        meta = _read_meta(name)
        if meta:
            items.append(meta)
    items.sort(key=lambda m: m["created_at"], reverse=True)
    return items


def delete_import(import_id: str):
    d = _import_dir(import_id)
    if os.path.isdir(d):
        shutil.rmtree(d)
