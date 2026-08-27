"""A photo being processed must appear exactly ONCE in /api/gallery.

The storage row is created partway through process_preview, while the pending
placeholder is still registered -- so without the claim/filter the same photo
is returned twice for the rest of its import.
"""
import asyncio, os, sys, tempfile

tmp = tempfile.mkdtemp(prefix="pe-claim-")
os.environ["RENDER_STORAGE_DIR"] = os.path.join(tmp, "renders")
os.environ.setdefault("PHOTO_ENHANCE_ADMIN_USER", "claimtest")
os.environ.setdefault("PHOTO_ENHANCE_ADMIN_PASSWORD", "claimtest-password")
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
here = os.path.join(root, "service")
sys.path.insert(0, here); sys.path.insert(0, root); os.chdir(here)

import main as app_module
import storage

FAILS = []


def check(label, cond):
    print(f"  {'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAILS.append(label)


async def run():
    user = "claimtest"
    row = storage.create_import("claimed.cr3", "upload", owner=user)
    released = asyncio.Event()
    started = asyncio.Event()

    async def work(claim):
        claim(row)            # exactly what pipeline does after create_import
        started.set()
        await released.wait()

    pending_id = app_module._start_import(user, "claimed.cr3", work)
    await started.wait()

    g = await app_module.gallery_list(user=user)
    ids = [i["id"] for i in g]
    print(f"    gallery while processing: {ids}")
    check("the half-built row is hidden while its placeholder is live", row not in ids)
    check("the placeholder is shown instead", pending_id in ids)
    check("the photo appears exactly once", len(ids) == 1)

    released.set()
    for _ in range(200):                      # let run()'s finally execute
        await asyncio.sleep(0.01)
        if not app_module._pending_imports:
            break

    g = await app_module.gallery_list(user=user)
    ids = [i["id"] for i in g]
    print(f"    gallery after finishing:  {ids}")
    check("the real row appears once processing completes", ids == [row])

    storage.delete_import(row)

asyncio.run(run())
print("\nCLAIM TEST FAILURES: " + str(FAILS) if FAILS else "\nclaim/filter behaves correctly")
sys.exit(1 if FAILS else 0)
