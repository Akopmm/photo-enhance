"""Phase 1: bulk-download raw FiveK source files (input DNG + Expert-C TIFF16)
from MIT's authoritative source, keeping them on disk for reuse.

Split from processing (process_raw.py) on purpose: downloading is
network/I-O bound and worth parallelizing heavily and resuming freely, while
processing (RAW demosaic + resize) is CPU-bound and cheap to rerun locally if
the processing logic ever needs a fix -- no need to re-download 175GB from
MIT again just because a resize/crop bug gets found later.

Usage:
    python3 download_raw.py --n_images 5000 --out data/fiveK_raw --workers 8
"""
import argparse
import os
import random
import shutil
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import certifi

BASE_URL = "https://data.csail.mit.edu/graphics/fivek/img"
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def _download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def fetch_one(name, dng_dir, tif_dir):
    dng_path = os.path.join(dng_dir, name + ".dng")
    tif_path = os.path.join(tif_dir, name + ".tif")
    if os.path.exists(dng_path) and os.path.exists(tif_path):
        return name, "skip"
    try:
        if not os.path.exists(dng_path):
            tmp = dng_path + ".part"
            _download(f"{BASE_URL}/dng/{name}.dng", tmp)
            os.rename(tmp, dng_path)
        if not os.path.exists(tif_path):
            tmp = tif_path + ".part"
            _download(f"{BASE_URL}/tiff16_c/{name}.tif", tmp)
            os.rename(tmp, tif_path)
        return name, "ok"
    except Exception as e:  # noqa: BLE001
        for p in (dng_path + ".part", tif_path + ".part"):
            if os.path.exists(p):
                os.remove(p)
        return name, f"error: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names_file", type=str, default="fivek_all_names.txt")
    ap.add_argument("--n_images", type=int, default=5000)
    ap.add_argument("--out", type=str, default="data/fiveK_raw")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.names_file) as f:
        all_names = [line.strip() for line in f if line.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(all_names)
    names = all_names[: args.n_images]

    dng_dir = os.path.join(args.out, "dng")
    tif_dir = os.path.join(args.out, "tiff16_c")
    os.makedirs(dng_dir, exist_ok=True)
    os.makedirs(tif_dir, exist_ok=True)

    ok, skipped, errors = 0, 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_one, n, dng_dir, tif_dir): n for n in names}
        for i, fut in enumerate(as_completed(futures)):
            name, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                errors.append((name, status))
            if (i + 1) % 50 == 0 or (i + 1) == len(names):
                print(f"[{i + 1}/{len(names)}] ok={ok} skipped={skipped} errors={len(errors)}", flush=True)

    print(f"\nDone. {ok} downloaded, {skipped} already present, {len(errors)} failed.")
    if errors:
        print("Failed (rerun this script to retry -- it resumes automatically):")
        for name, err in errors[:20]:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
