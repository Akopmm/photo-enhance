"""The first benchmark answered the wrong question. This one answers the right one.

bench.py compared against "decode 26MP, denoise 26MP", which is what the
service does for size=original ONLY. Every other preset already resizes right
after the colour model, so a Medium render denoises a 3200px frame -- 35 tiles,
not 140. Half-size decoding therefore saves half a second and nothing else.

So the question is really two questions:

  * Original (140 tiles): can it be cut without losing full resolution?
  * The presets (35 tiles at Medium): can that be cut too?

Both are attacked the same way -- denoise smaller, carry back only the
correction -- but they have to be measured at their own working size, which is
what this does.
"""
import os
import sys
import time

import numpy as np
import rawpy
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "..", "service"))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench  # noqa: E402
import denoise as dn  # noqa: E402
import pipeline  # noqa: E402

RAW = bench.RAW
MEDIUM = 3200


def tiles_for(h, w):
    step = dn.TILE - dn.OVERLAP
    return (len(range(0, max(h - dn.OVERLAP, 1), step))
            * len(range(0, max(w - dn.OVERLAP, 1), step)))


def main():
    full = bench.decode(RAW)
    t = torch.from_numpy(full).permute(2, 0, 1).unsqueeze(0).contiguous()
    work = bench.to_a(pipeline._resize_tensor(t, MEDIUM))
    h, w = work.shape[:2]
    print(f"Medium working size {w}x{h} -> {tiles_for(h, w)} tiles\n")

    print("reference: denoise at the working size (what ships today)", flush=True)
    t0 = time.time()
    ref = dn.denoise(work)
    t_ref = time.time() - t0
    print(f"  {t_ref:.1f}s\n")

    rows = [("reference: denoise at 3200", t_ref, ref)]
    for label, fn in [
        ("residual 1/2 of working size", lambda: bench.denoise_residual(work, 2)),
        ("residual 1/4 of working size", lambda: bench.denoise_residual(work, 4)),
        ("guided chroma at working size", lambda: bench.guided_chroma(work)),
        ("guided chroma + residual 1/2",
         lambda: bench.denoise_residual(bench.guided_chroma(work), 2)),
    ]:
        t0 = time.time()
        img = fn()
        secs = time.time() - t0
        rows.append((label, secs, img))
        print(f"  {label:32s} {secs:6.2f}s", flush=True)

    print(f"\n{'variant':34s} {'time':>8s} {'speedup':>8s} {'vs ref':>11s} {'vs untouched':>13s}")
    print("-" * 80)
    for label, secs, img in rows:
        vs = "reference" if label.startswith("reference") else f"{bench.psnr(img, ref):6.2f} dB"
        print(f"{label:34s} {secs:7.2f}s {t_ref / secs:7.1f}x {vs:>11s} "
              f"{bench.psnr(img, work):9.2f} dB")

    np.save(os.path.join(HERE, "med_ref.npy"), ref)
    np.save(os.path.join(HERE, "med_work.npy"), work)
    for label, _, img in rows[1:]:
        key = label.split()[0] + "_" + label.split()[1]
        np.save(os.path.join(HERE, f"med_{key}.npy"), img)
    print("\nsaved arrays for crop comparison")


if __name__ == "__main__":
    main()
