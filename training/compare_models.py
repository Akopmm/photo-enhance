"""Compare a trained checkpoint against the original authors' pretrained
weights on the same held-out test set, with the same metric.

Caveats this script does NOT hide:
  - The pretrained model was trained on the full MIT-Adobe FiveK, which
    contains the same photographs as our test split (different rendering
    pipeline, but the same source images). So it may effectively have seen
    our test photos. That biases the comparison IN ITS FAVOUR, not ours.
  - Our checkpoint is selected as "best on the test set", which makes its
    number slightly optimistic. Both effects are worth remembering before
    treating a small PSNR gap as decisive.
  - PSNR rewards matching Expert C specifically. It is not a measure of
    "looks good"; always eyeball real photos too (see --dump_dir).

Usage:
    python3 compare_models.py --checkpoint checkpoints/best.pt
    python3 compare_models.py --checkpoint checkpoints/best.pt --dump_dir /tmp/cmp
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from dataset import FiveKDataset  # noqa: E402
from shared.model import AdaptiveLUTModel  # noqa: E402

PRETRAINED_LUTS = os.environ.get("PRETRAINED_LUTS", "/tmp/orig_pretrained/LUTs_sRGB.pth")
PRETRAINED_CLF = os.environ.get("PRETRAINED_CLF", "/tmp/orig_pretrained/classifier_sRGB.pth")


def psnr(pred, target):
    pred = pred.clamp(0, 1)
    mse = ((pred * 255).round() - (target * 255).round()).pow(2).mean().item()
    if mse == 0:
        return 99.0
    return 10 * math.log10(255.0 * 255.0 / mse)


def load_pretrained():
    luts = torch.load(PRETRAINED_LUTS, map_location="cpu")
    clf = torch.load(PRETRAINED_CLF, map_location="cpu")
    sd = {f"luts.{i}.lut": luts[i]["LUT"] for i in ("0", "1", "2")}
    sd.update({f"predictor.{k}": v for k, v in clf.items()})
    m = AdaptiveLUTModel(n_luts=3, lut_dim=33)
    m.load_state_dict(sd)
    m.eval()
    return m


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    a = ckpt.get("args", {})
    m = AdaptiveLUTModel(n_luts=a.get("n_luts", 3), lut_dim=a.get("lut_dim", 33))
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m, ckpt


@torch.no_grad()
def score(model, loader, dump_dir=None, tag=""):
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        out, _ = model.infer(batch["input"])
        total += psnr(out, batch["target"])
        n += 1
        if dump_dir and i < 8:
            os.makedirs(dump_dir, exist_ok=True)
            arr = (out.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(dump_dir, f"{i:02d}_{tag}.jpg"), quality=92)
            if tag == "ours":  # write the shared reference images once
                inp = (batch["input"].squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                tgt = (batch["target"].squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(inp).save(os.path.join(dump_dir, f"{i:02d}_input.jpg"), quality=92)
                Image.fromarray(tgt).save(os.path.join(dump_dir, f"{i:02d}_expertC.jpg"), quality=92)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    ap.add_argument("--data_root", type=str, default="data/fiveK")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole test split")
    ap.add_argument("--dump_dir", type=str, default=None)
    args = ap.parse_args()

    ds = FiveKDataset(args.data_root, split="test")
    if args.limit:
        ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    print(f"test images: {len(ds)}")

    ours, ckpt = load_checkpoint(args.checkpoint)
    a = ckpt.get("args", {})
    print(f"our checkpoint: epoch {ckpt.get('epoch')}, "
          f"trained on {a.get('data_root')} n_luts={a.get('n_luts', 3)}")

    ours_psnr = score(ours, loader, args.dump_dir, "ours")
    pre_psnr = score(load_pretrained(), loader, args.dump_dir, "pretrained")

    print()
    print(f"  ours       : {ours_psnr:.2f} dB")
    print(f"  pretrained : {pre_psnr:.2f} dB")
    print(f"  delta      : {ours_psnr - pre_psnr:+.2f} dB")
    if args.dump_dir:
        print(f"\nsample renders written to {args.dump_dir}")


if __name__ == "__main__":
    main()
