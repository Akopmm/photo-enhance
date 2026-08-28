"""Same network, applied only where it is the only thing that works.

The research established two facts that together suggest a different split:
  * guided chroma removes colour noise essentially for free, and
  * what SCUNet uniquely contributes is the LUMA grain the filter cannot touch.

So give each half to whichever handles it: chroma to the filter at full
resolution, luma to the network at reduced resolution as a residual. The
network then runs on a quarter of the pixels and only ever sees luminance.
"""
import os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "service"))
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bench, denoise as dn, pipeline

def split(a):
    y = 0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]
    return y, a[...,2]-y, a[...,0]-y

def join(y, cb, cr):
    out = np.empty(y.shape + (3,), np.float32)
    out[...,0] = y + cr; out[...,2] = y + cb
    out[...,1] = (y - 0.299*out[...,0] - 0.114*out[...,2]) / 0.587
    return np.clip(out, 0, 1)

def hybrid(a, factor=2):
    """Chroma by filter at full res; luma by the network at 1/factor."""
    chroma_done = dn.guided_chroma(a)          # colour noise gone, luma intact
    y, cb, cr = split(chroma_done)
    h, w = y.shape
    t = torch.from_numpy(y)[None, None]
    small = F.interpolate(t, scale_factor=1/factor, mode="area")
    # SCUNet wants three channels; luma replicated is what it gets.
    rgb = small.repeat(1, 3, 1, 1)[0].permute(1, 2, 0).numpy()
    cleaned = dn.denoise(np.ascontiguousarray(rgb), method="quality")
    y_small_clean = cleaned.mean(2)
    resid = torch.from_numpy(y_small_clean - small[0,0].numpy())[None, None]
    up = F.interpolate(resid, size=(h, w), mode="bilinear", align_corners=False)[0,0].numpy()
    return join(np.clip(y + up, 0, 1), cb, cr)

def main():
    bench._require_photo()
    full = bench.decode(bench.RAW)
    t = torch.from_numpy(full).permute(2,0,1).unsqueeze(0).contiguous()
    work = bench.to_a(pipeline._resize_tensor(t, 3200))
    ref_path = os.path.join(HERE, "med_ref.npy")
    ref = np.load(ref_path)

    rows = [("SCUNet at 3200 (ships today)", 16.03, ref)]
    for label, fn in [("hybrid: chroma filter + luma net 1/2", lambda: hybrid(work, 2)),
                      ("hybrid: chroma filter + luma net 1/4", lambda: hybrid(work, 4)),
                      ("guided chroma only", lambda: dn.guided_chroma(work))]:
        t0 = time.time(); img = fn(); secs = time.time()-t0
        rows.append((label, secs, img)); print(f"  {label:38s} {secs:6.2f}s", flush=True)

    print(f"\n{'variant':40s} {'time':>7s} {'speedup':>8s} {'vs ref':>10s} {'strength':>10s}")
    print("-"*80)
    for label, secs, img in rows:
        vs = "reference" if label.startswith("SCUNet") else f"{bench.psnr(img, ref):6.2f} dB"
        print(f"{label:40s} {secs:6.2f}s {16.03/secs:7.1f}x {vs:>10s} {bench.psnr(img, work):7.2f} dB")

    from PIL import Image, ImageDraw
    CROP = 380
    import crops as C
    picks = C.noisiest_patches(work, n=3, size=CROP)
    variants = [("untouched", work)] + rows
    pad, header = 8, 24
    sheet = Image.new("RGB", (len(variants)*CROP+(len(variants)+1)*pad, 3*(CROP+header)+4*pad), (16,18,21))
    d = ImageDraw.Draw(sheet)
    for r,(py,px) in enumerate(picks):
        for c,item in enumerate(variants):
            label, img = (item[0], item[1]) if len(item)==2 else (item[0], item[2])
            patch = np.clip(img[py:py+CROP, px:px+CROP], 0, 1)
            x = pad + c*(CROP+pad); y = pad + r*(CROP+header+pad)
            d.text((x, y+5), label[:44], fill=(200,205,212))
            sheet.paste(Image.fromarray((patch*255).astype(np.uint8)), (x, y+header))
    out = os.path.join(HERE, "compare_hybrid.png"); sheet.save(out); print("\nwrote", out)


if __name__ == "__main__":
    main()
