"""Phase 2: turn locally-downloaded raw FiveK files (from download_raw.py)
into the 480p JPEG pairs dataset.py expects. Pure local processing, no
network -- safe to rerun as many times as needed (e.g. after a processing
bug fix) without touching MIT's server again.

Input rendering matches how real-world inference will process a user's own
CR3/ARW file later (rawpy default demosaic + camera white balance + sRGB
gamma, no creative grading), so the model trains on the same kind of "flat"
input it will see in production.

Usage:
    python3 process_raw.py --raw_dir data/fiveK_raw --out data/fiveK --workers 4
"""
import argparse
import hashlib
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rawpy
from PIL import Image

TARGET_LONG_EDGE = 500  # "480p"-equivalent, matches the original preprocessed release


def _resize_long_edge(img: Image.Image, long_edge: int) -> Image.Image:
    w, h = img.size
    scale = long_edge / max(w, h)
    if scale >= 1:
        return img
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def process_one(name, dng_dir, tif_dir, input_dir, expert_dir):
    dng_path = os.path.join(dng_dir, name + ".dng")
    tif_path = os.path.join(tif_dir, name + ".tif")
    input_out = os.path.join(input_dir, name + ".jpg")
    expert_out = os.path.join(expert_dir, name + ".jpg")

    if not (os.path.exists(dng_path) and os.path.exists(tif_path)):
        return name, "missing raw"

    try:
        with rawpy.imread(dng_path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
                output_color=rawpy.ColorSpace.sRGB,
            )
        input_img = Image.fromarray(rgb)
        input_img = _resize_long_edge(input_img, TARGET_LONG_EDGE)
        input_img.save(input_out, quality=95)

        # Resize the target to input_img's *exact* pixel size -- the DNG
        # demosaic and the Adobe-rendered TIFF aren't guaranteed to share
        # native dimensions, and a 1px mismatch breaks the per-pixel loss.
        expert_img = Image.open(tif_path).convert("RGB")
        expert_img = expert_img.resize(input_img.size, Image.LANCZOS)
        expert_img.save(expert_out, quality=95)

        return name, "ok"
    except Exception as e:  # noqa: BLE001
        return name, f"error: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/fiveK_raw")
    ap.add_argument("--out", type=str, default="data/fiveK")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--test_fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dng_dir = os.path.join(args.raw_dir, "dng")
    tif_dir = os.path.join(args.raw_dir, "tiff16_c")
    names = sorted(f[:-4] for f in os.listdir(dng_dir) if f.endswith(".dng"))

    input_dir = os.path.join(args.out, "input", "JPG", "480p")
    expert_dir = os.path.join(args.out, "expertC", "JPG", "480p")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(expert_dir, exist_ok=True)

    ok, errors = 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_one, n, dng_dir, tif_dir, input_dir, expert_dir): n
            for n in names
        }
        for i, fut in enumerate(as_completed(futures)):
            name, status = fut.result()
            if status == "ok":
                ok += 1
            else:
                errors.append((name, status))
            if (i + 1) % 50 == 0 or (i + 1) == len(names):
                print(f"[{i + 1}/{len(names)}] ok={ok} errors={len(errors)}", flush=True)

    # Scan the OUTPUT dir, not this batch's raw dir: the dataset is fetched in
    # disk-safe batches (raws deleted after each), so the split has to be
    # rebuilt from everything processed so far, not just what's on disk now.
    good_names = sorted(
        f[:-4] for f in os.listdir(input_dir)
        if f.endswith(".jpg") and os.path.exists(os.path.join(expert_dir, f))
    )

    # Deterministic per-name assignment, NOT a re-randomized sample: with
    # incremental batches, re-sampling each time would shuffle images between
    # train and test as the set grows, leaking test images into training and
    # making eval numbers meaningless across runs. Hashing the name pins each
    # image to one side forever, regardless of dataset size or batch order.
    def is_test(name: str) -> bool:
        h = hashlib.sha1(f"{args.seed}:{name}".encode()).digest()
        return (int.from_bytes(h[:4], "big") % 1000) < round(args.test_fraction * 1000)

    test_names = {n for n in good_names if is_test(n)}
    train_names = [n for n in good_names if n not in test_names]

    with open(os.path.join(args.out, "train_input.txt"), "w") as f:
        f.write("\n".join(train_names) + "\n")
    with open(os.path.join(args.out, "train_label.txt"), "w") as f:
        f.write("")  # kept for dataset.py compatibility; all train names go in train_input.txt
    with open(os.path.join(args.out, "test.txt"), "w") as f:
        f.write("\n".join(sorted(test_names)) + "\n")

    print(f"\nDone. {len(good_names)} usable images ({len(train_names)} train / {len(test_names)} test).")
    if errors:
        print(f"{len(errors)} failed:")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
