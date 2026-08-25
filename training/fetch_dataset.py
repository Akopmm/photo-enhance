"""Fetch and process the whole FiveK dataset in disk-safe batches.

The raw source files are huge (~58MB per image pair: a ~5MB DNG plus a
~30-50MB 16-bit TIFF), so the full 5,000-image set is ~290GB -- far more
than a laptop usually has spare. The processed 480p training pairs, by
contrast, are ~120KB per image (~600MB for the full set).

So this downloads a batch, processes it into the training set, deletes that
batch's raws, and moves on. Peak extra disk usage is one batch, not the
whole dataset.

HARD SAFETY RULE: before each batch it checks actual free disk space and
refuses to continue below --min_free_gb. Filling the boot disk of the
machine this runs on is a real failure mode (it happened once during this
project -- an unbounded download ran the disk down to 121MB free), so the
guard aborts rather than trying to be clever.

Usage:
    python3 fetch_dataset.py --target 5000 --batch 600
"""
import argparse
import os
import shutil
import subprocess
import sys


def free_gb(path: str) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / (1024 ** 3)


def processed_count(out_dir: str) -> int:
    d = os.path.join(out_dir, "input", "JPG", "480p")
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith(".jpg")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=5000, help="total images to end up with")
    ap.add_argument("--batch", type=int, default=600, help="images per download batch")
    ap.add_argument("--out", type=str, default="data/fiveK")
    ap.add_argument("--raw_dir", type=str, default="data/fiveK_raw")
    ap.add_argument("--min_free_gb", type=float, default=60.0,
                    help="abort before a batch if free space is under this")
    ap.add_argument("--start_skip", type=int, default=0,
                    help="how many names of the shuffled list are already fetched")
    ap.add_argument("--dl_workers", type=int, default=8)
    ap.add_argument("--proc_workers", type=int, default=6)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable

    skip = args.start_skip
    while skip < args.target:
        have = processed_count(args.out)
        free = free_gb(here)
        print(f"\n=== batch @ skip={skip} | processed so far: {have} | free: {free:.1f}GB ===",
              flush=True)

        if free < args.min_free_gb:
            print(f"ABORT: only {free:.1f}GB free, below --min_free_gb={args.min_free_gb}. "
                  f"Stopping cleanly with {have} images processed.", flush=True)
            return 1

        n = min(args.batch, args.target - skip)

        rc = subprocess.call([
            py, os.path.join(here, "download_raw.py"),
            "--skip", str(skip), "--n_images", str(n),
            "--out", args.raw_dir, "--workers", str(args.dl_workers),
        ], cwd=here)
        if rc != 0:
            print(f"download batch failed (rc={rc}) -- stopping", flush=True)
            return rc

        rc = subprocess.call([
            py, os.path.join(here, "process_raw.py"),
            "--raw_dir", args.raw_dir, "--out", args.out,
            "--workers", str(args.proc_workers),
        ], cwd=here)
        if rc != 0:
            print(f"processing batch failed (rc={rc}) -- stopping", flush=True)
            return rc

        # Reclaim the batch's raws immediately -- this is what keeps peak
        # usage at one batch instead of the whole dataset.
        for sub in ("dng", "tiff16_c"):
            d = os.path.join(here, args.raw_dir, sub)
            if os.path.isdir(d):
                shutil.rmtree(d)
        print(f"reclaimed raws for this batch; free now: {free_gb(here):.1f}GB", flush=True)

        skip += n

    print(f"\nDONE. {processed_count(args.out)} processed image pairs in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
