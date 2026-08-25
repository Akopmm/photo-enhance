"""Runtime-editable settings, persisted to the data volume.

Environment variables remain the defaults (so a fresh container still works
with no setup), but anything saved through the settings page overrides them
and survives restarts and image updates.

Secret handling: the Immich API key is stored here in plain text, which is
the same trust level as an env var on the host -- but it is NEVER sent to
the browser. `public_dict()` returns a masked preview only. Submitting an
empty key in the settings form means "leave unchanged" rather than "clear",
so the UI never needs to round-trip the real value.
"""
import json
import os
import threading

DATA_DIR = os.environ.get("RENDER_STORAGE_DIR", os.path.join(os.path.dirname(__file__), "data", "renders"))
SETTINGS_PATH = os.path.join(os.path.dirname(DATA_DIR.rstrip("/")), "settings.json")

_lock = threading.Lock()
_cache = None

# mode: "classic"  -> global LUT + full-frame style presets only (what the
#                     stable service does today; cheap, no extra models)
#       "enhanced" -> additionally runs segmentation to grade regions
#                     separately (subject vs background, sky vs ground)
DEFAULTS = {
    "mode": "classic",
    "immich_url": os.environ.get("IMMICH_URL", "http://192.168.0.123:2283"),
    "immich_api_key": os.environ.get("IMMICH_API_KEY", ""),
    "subject_masking": True,
    "sky_masking": True,
    "cinematic_presets": True,
    "max_concurrent_jobs": int(os.environ.get("MAX_CONCURRENT_JOBS", "3")),
    "idle_unload_minutes": float(os.environ.get("IDLE_UNLOAD_MINUTES", "15")),
    "jpeg_quality": 95,
    "thumb_long_edge": 480,
}

# Fields the settings page may write. Anything else in a POST is ignored, so
# a malformed or hostile request can't inject arbitrary keys.
EDITABLE = {
    "mode": str,
    "immich_url": str,
    "immich_api_key": str,
    "subject_masking": bool,
    "sky_masking": bool,
    "cinematic_presets": bool,
    "max_concurrent_jobs": int,
    "idle_unload_minutes": float,
    "jpeg_quality": int,
}

VALID_MODES = ("classic", "enhanced")


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    data = dict(DEFAULTS)
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH) as f:
                data.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
        except (json.JSONDecodeError, OSError):
            # A corrupt settings file must not take the service down --
            # fall back to defaults rather than refusing to start.
            pass
    _cache = data
    return _cache


def get(key: str):
    return _load().get(key, DEFAULTS.get(key))


def all_settings() -> dict:
    return dict(_load())


def _mask_key(key: str) -> str:
    if not key:
        return ""
    return f"…{key[-4:]}" if len(key) > 4 else "…"


def public_dict() -> dict:
    """Settings safe to hand to a browser -- the API key is masked."""
    d = dict(_load())
    d["immich_api_key_set"] = bool(d.get("immich_api_key"))
    d["immich_api_key_preview"] = _mask_key(d.get("immich_api_key", ""))
    d.pop("immich_api_key", None)
    return d


def update(patch: dict) -> dict:
    """Apply and persist a settings patch. Returns the public view."""
    with _lock:
        data = dict(_load())
        for key, caster in EDITABLE.items():
            if key not in patch:
                continue
            value = patch[key]
            # Empty API key means "keep the existing one" -- the browser is
            # never given the real value, so it can't send it back.
            if key == "immich_api_key" and not str(value).strip():
                continue
            try:
                data[key] = bool(value) if caster is bool else caster(value)
            except (TypeError, ValueError):
                continue

        if data.get("mode") not in VALID_MODES:
            data["mode"] = DEFAULTS["mode"]
        data["max_concurrent_jobs"] = max(1, min(int(data["max_concurrent_jobs"]), 8))
        data["jpeg_quality"] = max(60, min(int(data["jpeg_quality"]), 100))
        data["idle_unload_minutes"] = max(0.0, float(data["idle_unload_minutes"]))

        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        tmp = SETTINGS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, SETTINGS_PATH)  # atomic; never leaves a half-written file

        global _cache
        _cache = data
    return public_dict()
