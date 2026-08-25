"""Users, password hashing and session cookies.

Deliberately stdlib-only (hashlib.scrypt + hmac) -- this service already
pulls in torch and transformers; adding an auth framework for what is a
handful of accounts on a private tailnet isn't worth the dependency.

Security notes, stated plainly rather than assumed:
  * Passwords are stored as scrypt hashes with a per-user random salt.
    Never in plain text, never recoverable.
  * Sessions are HMAC-signed cookies (username + expiry + signature). The
    signing secret is generated once and kept in the data volume; rotating
    it invalidates every session.
  * Cookies are HttpOnly and SameSite=Lax. Secure is set when the request
    arrives over HTTPS (Tailscale serve terminates TLS, so this is on in
    real use but off for plain-HTTP local testing).
  * Each user holds their OWN Immich API key. One user's key is never
    exposed to another, and no key is ever sent to any browser -- only a
    masked preview.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time

import settings as global_settings

DATA_ROOT = os.path.dirname(global_settings.SETTINGS_PATH)
USERS_PATH = os.path.join(DATA_ROOT, "users.json")
SECRET_PATH = os.path.join(DATA_ROOT, "session_secret")

SESSION_COOKIE = "pe_session"
SESSION_TTL = 30 * 24 * 3600  # 30 days

_lock = threading.Lock()
_users_cache = None
_secret = None


# ---------------------------------------------------------------- secret

def _session_secret() -> bytes:
    global _secret
    if _secret is None:
        os.makedirs(DATA_ROOT, exist_ok=True)
        if os.path.exists(SECRET_PATH):
            with open(SECRET_PATH, "rb") as f:
                _secret = f.read().strip()
        if not _secret:
            _secret = base64.b64encode(secrets.token_bytes(32))
            with open(SECRET_PATH, "wb") as f:
                f.write(_secret)
            os.chmod(SECRET_PATH, 0o600)
    return _secret


# ---------------------------------------------------------------- users

def _load_users() -> dict:
    global _users_cache
    if _users_cache is not None:
        return _users_cache
    users = {}
    if os.path.exists(USERS_PATH):
        try:
            with open(USERS_PATH) as f:
                users = json.load(f)
        except (json.JSONDecodeError, OSError):
            users = {}
    _users_cache = users
    return _users_cache


def _save_users(users: dict):
    global _users_cache
    os.makedirs(DATA_ROOT, exist_ok=True)
    tmp = USERS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(users, f, indent=2)
    os.replace(tmp, USERS_PATH)
    os.chmod(USERS_PATH, 0o600)
    _users_cache = users


def _hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return base64.b64encode(dk).decode()


def create_user(username: str, password: str, is_admin: bool = False) -> bool:
    """Returns False if the username already exists."""
    username = username.strip().lower()
    if not username or not password:
        return False
    with _lock:
        users = dict(_load_users())
        if username in users:
            return False
        salt = secrets.token_bytes(16)
        users[username] = {
            "salt": base64.b64encode(salt).decode(),
            "password_hash": _hash_password(password, salt),
            "is_admin": bool(is_admin),
            "immich_url": global_settings.DEFAULTS["immich_url"],
            "immich_api_key": "",
            "created_at": time.time(),
        }
        _save_users(users)
    return True


def set_password(username: str, password: str) -> bool:
    username = username.strip().lower()
    with _lock:
        users = dict(_load_users())
        if username not in users or not password:
            return False
        salt = secrets.token_bytes(16)
        users[username]["salt"] = base64.b64encode(salt).decode()
        users[username]["password_hash"] = _hash_password(password, salt)
        _save_users(users)
    return True


def verify(username: str, password: str) -> bool:
    username = (username or "").strip().lower()
    user = _load_users().get(username)
    if not user:
        # Still burn comparable time so a missing username isn't obviously
        # faster than a wrong password (basic timing-leak hygiene).
        hashlib.scrypt(b"x", salt=b"x" * 16, n=2 ** 14, r=8, p=1, dklen=32)
        return False
    salt = base64.b64decode(user["salt"])
    return hmac.compare_digest(_hash_password(password, salt), user["password_hash"])


def get_user(username: str) -> dict | None:
    u = _load_users().get((username or "").strip().lower())
    return dict(u) if u else None


def list_users() -> list[dict]:
    out = []
    for name, u in sorted(_load_users().items()):
        out.append({
            "username": name,
            "is_admin": bool(u.get("is_admin")),
            "immich_url": u.get("immich_url", ""),
            "immich_api_key_set": bool(u.get("immich_api_key")),
            "immich_api_key_preview": _mask(u.get("immich_api_key", "")),
        })
    return out


def delete_user(username: str) -> bool:
    username = (username or "").strip().lower()
    with _lock:
        users = dict(_load_users())
        if username not in users:
            return False
        # Refuse to remove the last admin -- otherwise the instance becomes
        # permanently unadministrable.
        if users[username].get("is_admin"):
            admins = [n for n, u in users.items() if u.get("is_admin")]
            if len(admins) <= 1:
                return False
        users.pop(username)
        _save_users(users)
    return True


def update_user_immich(username: str, url: str | None = None, api_key: str | None = None) -> bool:
    """Per-user Immich credentials. An empty api_key means 'leave unchanged',
    matching the settings-page contract (the real key is never sent to the
    browser, so the browser can't echo it back)."""
    username = (username or "").strip().lower()
    with _lock:
        users = dict(_load_users())
        if username not in users:
            return False
        if url is not None and str(url).strip():
            users[username]["immich_url"] = str(url).strip().rstrip("/")
        if api_key is not None and str(api_key).strip():
            users[username]["immich_api_key"] = str(api_key).strip()
        _save_users(users)
    return True


def _mask(key: str) -> str:
    if not key:
        return ""
    return f"…{key[-4:]}" if len(key) > 4 else "…"


def any_users() -> bool:
    return bool(_load_users())


def bootstrap_from_env():
    """Create the first admin from env vars if no users exist yet, so a fresh
    deployment is reachable without a chicken-and-egg problem."""
    if any_users():
        return
    user = os.environ.get("PHOTO_ENHANCE_ADMIN_USER", "").strip()
    pw = os.environ.get("PHOTO_ENHANCE_ADMIN_PASSWORD", "").strip()
    if user and pw:
        create_user(user, pw, is_admin=True)


# ---------------------------------------------------------------- sessions

def issue_session(username: str) -> str:
    exp = int(time.time()) + SESSION_TTL
    payload = f"{username.lower()}:{exp}"
    sig = hmac.new(_session_secret(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}:{sig}"


def read_session(token: str) -> str | None:
    """Returns the username if the token is valid and unexpired."""
    if not token:
        return None
    try:
        username, exp_s, sig = token.rsplit(":", 2)
        exp = int(exp_s)
    except (ValueError, AttributeError):
        return None
    if exp < time.time():
        return None
    expected = hmac.new(_session_secret(), f"{username}:{exp}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    return username if get_user(username) else None
