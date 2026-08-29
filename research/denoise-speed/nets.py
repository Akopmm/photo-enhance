"""Cheaper networks against SCUNet: same job, less work per pixel.

Reducing the resolution the network sees is structurally dead -- grain lives
in exactly the detail that downsampling discards. So the remaining lever is a
network that costs less per pixel while still seeing every pixel.

  SCUNet  17.9M params, NO internal downsampling -- the whole cost problem
  FFDNet   0.85M, works on a 2x2 pixel-shuffled sub-image: a quarter of the
           spatial positions, but full-resolution detail still reaches it
  DRUNet   32.6M, but a U-Net that downsamples internally, so parameters and
           cost are not the same question

One caveat that the numbers alone will not show: the SCUNet checkpoint in the
service is `color_real`, trained on REAL camera noise. FFDNet and DRUNet here
are the AWGN checkpoints -- trained on synthetic Gaussian noise. That is a
different distribution from a real sensor, so their scores may flatter them
while the crops disagree. The crops decide.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
# Weights are large and not committed. Point WEIGHTS at a directory
# holding ffdnet_color.pth and drunet_color.pth from the KAIR releases.
MODELS = os.environ.get("WEIGHTS", os.path.join(HERE, "weights"))
sys.path.insert(0, MODELS)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "service"))

import bench  # noqa: E402
import denoise as dn  # noqa: E402
import pipeline  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load_ffdnet():
    # Imported here, not at module scope: these come from KAIR and are not
    # vendored, so the file must stay importable without them for CI's
    # drift check.
    from network_ffdnet import FFDNet
    m = FFDNet(in_nc=3, out_nc=3, nc=96, nb=12, act_mode="R")
    m.load_state_dict(torch.load(os.path.join(MODELS, "ffdnet_color.pth"), map_location="cpu"))
    return m.eval().to(DEV)


def load_drunet():
    from network_unet import UNetRes
    m = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode="R",
                downsample_mode="strideconv", upsample_mode="convtranspose", bias=False)
    m.load_state_dict(torch.load(os.path.join(MODELS, "drunet_color.pth"), map_location="cpu"))
    return m.eval().to(DEV)


def _pad_to(t, mult):
    h, w = t.shape[-2:]
    ph, pw = (-h) % mult, (-w) % mult
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode="reflect")
    return t, h, w


@torch.no_grad()
def run_ffdnet(m, arr, sigma8):
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)[None].to(DEV)
    t, h, w = _pad_to(t, 2)
    s = torch.full((1, 1, 1, 1), sigma8 / 255.0, device=DEV)
    out = m(t, s)[:, :, :h, :w]
    return out.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def run_drunet(m, arr, sigma8):
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)[None].to(DEV)
    t, h, w = _pad_to(t, 8)
    noise = torch.full((1, 1, t.shape[2], t.shape[3]), sigma8 / 255.0, device=DEV)
    out = m(torch.cat([t, noise], 1))[:, :, :h, :w]
    return out.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()


def main():
    bench._require_photo()
    full = bench.decode(bench.RAW)
    t = torch.from_numpy(full).permute(2, 0, 1).unsqueeze(0).contiguous()
    work = bench.to_a(pipeline._resize_tensor(t, 3200))
    sigma = dn.estimate_sigma(work)
    print(f"working size {work.shape[1]}x{work.shape[0]}, estimated sigma {sigma:.2f} (8-bit)\n")

    ref_path = os.path.join(HERE, "med_ref.npy")
    if os.path.exists(ref_path):
        ref, t_ref = np.load(ref_path), 16.03
        print("SCUNet reference from cache")
    else:
        t0 = time.time(); ref = dn.denoise(work, method="quality"); t_ref = time.time() - t0
        np.save(ref_path, ref)
    print(f"  SCUNet (color_real)            {t_ref:7.2f}s\n")

    rows = [("SCUNet 17.9M (ships today)", t_ref, ref)]
    ffd, dru = load_ffdnet(), load_drunet()

    for label, fn in [
        (f"FFDNet 0.85M  sigma={sigma:.1f}", lambda: run_ffdnet(ffd, work, sigma)),
        (f"FFDNet 0.85M  sigma={sigma*2:.1f}", lambda: run_ffdnet(ffd, work, sigma * 2)),
        (f"DRUNet 32.6M  sigma={sigma:.1f}", lambda: run_drunet(dru, work, sigma)),
        (f"DRUNet 32.6M  sigma={sigma*2:.1f}", lambda: run_drunet(dru, work, sigma * 2)),
    ]:
        fn()  # warm
        t0 = time.time(); img = fn(); secs = time.time() - t0
        rows.append((label, secs, img))
        print(f"  {label:30s} {secs:7.2f}s", flush=True)

    print(f"\n{'variant':32s} {'time':>8s} {'speedup':>8s} {'vs SCUNet':>11s} {'strength':>10s}")
    print("-" * 76)
    for label, secs, img in rows:
        vs = "reference" if label.startswith("SCUNet") else f"{bench.psnr(img, ref):6.2f} dB"
        print(f"{label:32s} {secs:7.2f}s {t_ref / secs:7.1f}x {vs:>11s} "
              f"{bench.psnr(img, work):7.2f} dB")

    from PIL import Image, ImageDraw
    import crops as C
    CROP = 380
    picks = C.noisiest_patches(work, n=3, size=CROP)
    show = [("untouched", work)] + [(l, i) for l, _, i in rows]
    pad, header = 8, 24
    sheet = Image.new("RGB", (len(show) * CROP + (len(show) + 1) * pad,
                              3 * (CROP + header) + 4 * pad), (16, 18, 21))
    d = ImageDraw.Draw(sheet)
    for r, (py, px) in enumerate(picks):
        for c, (label, img) in enumerate(show):
            patch = np.clip(img[py:py + CROP, px:px + CROP], 0, 1)
            x = pad + c * (CROP + pad)
            y = pad + r * (CROP + header + pad)
            d.text((x, y + 5), label[:42], fill=(200, 205, 212))
            sheet.paste(Image.fromarray((patch * 255).astype(np.uint8)), (x, y + header))
    out = os.path.join(HERE, "compare_nets.png")
    sheet.save(out)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
