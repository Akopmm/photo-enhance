"""Mutation-test the test suite: does e2e_test.py actually have teeth?

A passing test suite proves nothing on its own — a check that cannot fail
looks exactly like a check that always passes. This breaks the product in
specific, realistic ways and requires the suite to FAIL each time. A mutation
that survives means whatever was supposed to cover it is decorative.

    python mutation_test.py

Runs against a throwaway copy of the tree, never the working one, and starts
its own server on a spare port.

Two findings from writing it, both worth keeping in mind:

  * A mutation that changes nothing observable proves nothing about the
    tests. Adding a path to PUBLIC_PATHS does NOT expose it, because routes
    also carry Depends(current_user) — auth is enforced twice. The first
    version of that mutation "survived" and looked like a test gap; it was a
    property of the product.
  * The suite really did have a hole: strength is clamped separately in the
    preview path and the download path, and only the preview was checked, so
    an export that ignored the slider passed clean.
"""
import os, shutil, subprocess, sys, tempfile, time, signal, json, urllib.request

TESTS = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(TESTS), "service")
PY = sys.executable
PORT = int(os.environ.get("MUTATION_TEST_PORT", "5077"))

# (name, file, find, replace, which check should die)
MUTATIONS = [
    # NB: adding a path to PUBLIC_PATHS is NOT enough to expose it -- the
    # routes also carry Depends(current_user), so auth is enforced twice.
    # Breaking the middleware alone changes nothing observable, which is a
    # property of the product, not a hole in the suite. Break the dependency.
    ("auth: session check disabled entirely", "main.py",
     "def current_user(request: Request) -> str:",
     "def current_user(request: Request) -> str:\n    return 'e2e_admin'  # MUTANT",
     "gallery refuses an anonymous caller"),
     

    ("isolation: ownership check removed", "main.py",
     "def _owned(import_id: str, user: str):\n    if not storage.owns(import_id, user):",
     "def _owned(import_id: str, user: str):\n    if False:",
     "another user cannot open the import"),

    ("privacy: the Immich API key is echoed back", "main.py",
     '"immich_api_key_preview": auth._mask(u.get("immich_api_key", "")),',
     '"immich_api_key_preview": u.get("immich_api_key", ""),',
     "the API key is never echoed back"),

    ("settings: partial POST clobbers checkboxes again", "main.py",
     'declared = form.pop("_checkbox_fields", "")\n    for flag in [f.strip() for f in str(declared).split(",") if f.strip()]:',
     'form.pop("_checkbox_fields", "")\n    for flag in ("subject_masking", "sky_masking", "depth_masking", "cinematic_presets"):',
     "a partial settings update leaves other toggles alone"),

    ("permissions: non-admin may change settings", "main.py",
     "async def post_settings(request: Request, user: str = Depends(require_admin)):",
     "async def post_settings(request: Request, user: str = Depends(current_user)):",
     "a non-admin cannot change instance settings"),

    ("editor: the strength parameter is ignored", "pipeline.py",
     "    strength = min(max(float(strength), 0.0), 1.0)",
     "    strength = 1.0",
     "the strength slider changes the image"),
]


def run(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def start(workdir, data):
    env = dict(os.environ, RENDER_STORAGE_DIR=data)
    p = subprocess.Popen([PY, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
                          "--port", str(PORT), "--log-level", "error"],
                         cwd=workdir, env=env, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()
            return p
        except Exception:
            time.sleep(1)
    return p


def stop(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        pass
    time.sleep(1)


def suite(workdir):
    # e2e_test.py is a pure HTTP client, so it runs from the real tests/
    # directory and talks to the mutated server over the wire. It does not
    # need to be inside the copied tree — and after tests moved out of
    # service/ it no longer could be.
    r = run(f'"{PY}" "{TESTS}/e2e_test.py" --base http://127.0.0.1:{PORT}', cwd=workdir)
    out = r.stdout + r.stderr
    failed = [l.strip()[5:].split("  —")[0].strip()
              for l in out.splitlines() if l.strip().startswith("FAIL")]
    return r.returncode, failed, out


def make_env():
    # The layout matters: model_runtime does `from shared.model import ...`,
    # and shared/ lives beside service/, not inside it. Copying service/ alone
    # gives a tree that cannot import — which is how this harness first failed.
    work = tempfile.mkdtemp(prefix="mut-src-")
    root = os.path.dirname(SRC)
    shutil.copytree(SRC, f"{work}/service", ignore=shutil.ignore_patterns(".venv", "data", "__pycache__", "weights"))
    shutil.copytree(f"{root}/shared", f"{work}/shared", ignore=shutil.ignore_patterns("__pycache__"))
    os.symlink(f"{SRC}/weights", f"{work}/service/weights")
    data = tempfile.mkdtemp(prefix="mut-data-")
    os.makedirs(f"{data}/_config", exist_ok=True)
    json.dump({"mode": "classic", "jpeg_quality": 95, "max_concurrent_jobs": 2,
               "subject_masking": True, "sky_masking": True, "depth_masking": True,
               "cinematic_presets": True},
              open(f"{data}/_config/settings.json", "w"))
    return f"{work}/service", data


def main():
    print("=== baseline: unmutated source must PASS ===", flush=True)
    wd, data = make_env()
    p = start(wd, data); code, failed, out = suite(wd); stop(p)
    total = out.count("  ok ") + len(failed)
    print(f"  baseline: exit={code}, {len(failed)} failures, ~{total} checks", flush=True)
    if code != 0:
        print("  BASELINE IS RED — cannot mutation-test against it.")
        print("  failures:", failed[:4])
        print("  suite output tail:", out.strip().splitlines()[-6:])
        return 1

    results = []
    for name, fname, find, repl, expect in MUTATIONS:
        wd, data = make_env()
        path = f"{wd}/{fname}"
        s = open(path).read()
        if s.count(find) != 1:
            results.append((name, "SKIPPED", f"anchor matched {s.count(find)}x", expect));
            print(f"\n--- {name}\n  SKIPPED: anchor matched {s.count(find)}x", flush=True); continue
        open(path, "w").write(s.replace(find, repl))
        print(f"\n--- {name}", flush=True)
        p = start(wd, data); code, failed, out = suite(wd); stop(p)
        caught = code != 0
        hit = expect in failed
        results.append((name, "CAUGHT" if caught else "SURVIVED",
                        ("expected check died" if hit else f"died elsewhere: {failed[:3]}") if caught else "",
                        expect))
        print(f"  suite exit={code}  failures={failed[:3]}", flush=True)
        print(f"  -> {'CAUGHT' if caught else 'SURVIVED (test gap!)'}"
              f"{' (by the intended check)' if hit else ''}", flush=True)

    print("\n\n=== kill matrix ===")
    for name, verdict, note, expect in results:
        mark = {"CAUGHT": "killed  ", "SURVIVED": "SURVIVED", "SKIPPED": "skipped "}[verdict]
        print(f"  {mark} {name}")
        if verdict != "CAUGHT":
            print(f"           expected to trip: {expect}   {note}")
    killed = sum(1 for _, v, _, _ in results if v == "CAUGHT")
    print(f"\n  {killed}/{len(results)} mutations caught")
    return 0 if killed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
