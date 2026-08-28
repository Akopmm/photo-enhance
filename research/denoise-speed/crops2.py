"""100% crops at the Medium working size -- the path most renders take."""
import os, sys, time
import numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "..", "service"))
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bench, denoise as dn, pipeline

CROP = 380


def main():
    bench._require_photo()
    full = bench.decode(bench.RAW)
    t = torch.from_numpy(full).permute(2,0,1).unsqueeze(0).contiguous()
    work = bench.to_a(pipeline._resize_tensor(t, 3200))

    cache = os.path.join(HERE, "med_ref.npy")
    ref = np.load(cache) if os.path.exists(cache) else dn.denoise(work)
    np.save(cache, ref)

    variants = [
        ("untouched", work),
        ("SCUNet 16s (ships today)", ref),
        ("guided chroma 0.09s", bench.guided_chroma(work)),
        ("residual 1/2  5.1s", bench.denoise_residual(work, 2)),
    ]
    picks = __import__("crops").noisiest_patches(work, n=3, size=CROP)
    pad, header = 8, 24
    cols, rows = len(variants), len(picks)
    sheet = Image.new("RGB", (cols*CROP + (cols+1)*pad, rows*(CROP+header) + (rows+1)*pad), (16,18,21))
    d = ImageDraw.Draw(sheet)
    for r,(py,px) in enumerate(picks):
        for c,(label,img) in enumerate(variants):
            patch = np.clip(img[py:py+CROP, px:px+CROP], 0, 1)
            x = pad + c*(CROP+pad); y = pad + r*(CROP+header+pad)
            d.text((x, y+5), label, fill=(200,205,212))
            sheet.paste(Image.fromarray((patch*255).astype(np.uint8)), (x, y+header))
    out = os.path.join(HERE, "compare_medium.png"); sheet.save(out)
    print("wrote", out, sheet.size)


if __name__ == "__main__":
    main()
