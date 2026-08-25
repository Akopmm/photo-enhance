"""Train the adaptive-LUT enhancement model on FiveK (Expert C target).

Mirrors the original Image-Adaptive-3DLUT paper's training recipe (Adam,
lr=1e-4, MSE pixel loss + LUT smoothness/monotonicity regularization,
per-epoch PSNR eval on held-out images), but runs on Apple Silicon via MPS
instead of CUDA, with the grid_sample-based model in model.py.

Training crops are resized to a fixed size (see dataset.py's TRAIN_CROP_SIZE)
so batch_size can be >1 -- each step then submits real parallel work to the
GPU instead of one tiny image at a time, which for a model this small was
almost pure per-step dispatch overhead rather than compute.
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset import FiveKDataset
from shared.model import AdaptiveLUTModel, TotalVariation3D


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def psnr(pred, target):
    # pred can legitimately land slightly outside [0,1] (real overshoot from
    # contrast/highlight correction) -- clamp before scoring, same as any
    # real display/export path would, or MSE gets inflated by out-of-range
    # values that were never actually visible in the rendered image.
    pred = pred.clamp(0, 1)
    mse = ((pred * 255).round() - (target * 255).round()).pow(2).mean().item()
    if mse == 0:
        return 99.0
    import math
    return 10 * math.log10(255.0 * 255.0 / mse)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        img_in = batch["input"].to(device)
        img_out = batch["target"].to(device)
        pred, _ = model.infer(img_in)
        total += psnr(pred, img_out)
        n += 1
    model.train()
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/fiveK")
    ap.add_argument("--n_epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_smooth", type=float, default=1e-4)
    ap.add_argument("--lambda_monotonicity", type=float, default=10.0)
    ap.add_argument("--n_luts", type=int, default=3)
    ap.add_argument("--lut_dim", type=int, default=33)
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=5, help="run the (slow, one-image-at-a-time) test-set PSNR eval every N epochs")
    ap.add_argument("--eval_subset", type=int, default=250,
                    help="cap eval to this many test images (they're evaluated one at a time at "
                         "native resolution, so a large test set makes eval dominate epoch time). "
                         "A fixed prefix, so the number stays comparable across epochs/runs.")
    ap.add_argument("--cosine", action="store_true", default=True,
                    help="cosine-anneal the LR to ~0 over the run (default on). A flat LR was "
                         "what stalled the first attempt well short of convergence.")
    ap.add_argument("--no_cosine", dest="cosine", action="store_false")
    ap.add_argument("--resume", type=str, default=None, help="path to a checkpoint .pt to resume from")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_ds = FiveKDataset(args.data_root, split="train")
    test_ds = FiveKDataset(args.data_root, split="test")
    if args.eval_subset and len(test_ds) > args.eval_subset:
        test_ds = torch.utils.data.Subset(test_ds, range(args.eval_subset))
    print(f"train: {len(train_ds)} images, eval: {len(test_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)

    model = AdaptiveLUTModel(n_luts=args.n_luts, lut_dim=args.lut_dim).to(device)
    model.init_weights()
    tv = TotalVariation3D()

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=args.lr * 0.01)
        if args.cosine else None
    )
    if scheduler and start_epoch:
        for _ in range(start_epoch):
            scheduler.step()

    mse_loss_fn = torch.nn.MSELoss()

    print_every = max(1, len(train_loader) // 3)  # a few progress lines per epoch regardless of batch count
    best_psnr = 0.0
    for epoch in range(start_epoch, args.n_epochs):
        t0 = time.time()
        for i, batch in enumerate(train_loader):
            img_in = batch["input"].to(device)
            img_out = batch["target"].to(device)

            optimizer.zero_grad()
            pred, weights = model(img_in)

            mse = mse_loss_fn(pred, img_out)
            weights_norm = (weights ** 2).mean()

            tv_total = 0.0
            mono_total = 0.0
            for lut_module in model.luts:
                tv_i, mono_i = tv(lut_module.lut)
                tv_total = tv_total + tv_i
                mono_total = mono_total + mono_i

            loss = mse + args.lambda_smooth * (weights_norm + tv_total) + args.lambda_monotonicity * mono_total
            loss.backward()
            optimizer.step()

            # .item()/psnr() force a CPU-GPU sync (measured ~1.5s/epoch on 42
            # batches) -- only pay that cost on the batches we actually print,
            # not every single step.
            if i % print_every == 0:
                elapsed = time.time() - t0
                batch_psnr = psnr(pred.detach(), img_out)
                print(
                    f"epoch {epoch} batch {i}/{len(train_loader)} "
                    f"mse={mse.item():.5f} tv={float(tv_total.detach()):.5f} mono={float(mono_total.detach()):.5f} "
                    f"psnr={batch_psnr:.2f} elapsed={elapsed:.1f}s"
                )

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.n_epochs - 1)
        if do_eval:
            # The only part of an epoch that's still one-image-at-a-time (real
            # photos, varying native sizes -- can't batch without resizing away
            # the point of measuring true PSNR), so it dominates wall-clock time
            # if run every epoch. Running it every eval_every epochs instead
            # keeps the fast batched training loop the bottleneck, not this.
            test_psnr = evaluate(model, test_loader, device)
            is_best = test_psnr > best_psnr
            best_psnr = max(best_psnr, test_psnr)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"== epoch {epoch} done in {time.time() - t0:.1f}s, lr={lr_now:.2e}, "
                  f"test PSNR={test_psnr:.2f} (best={best_psnr:.2f}) ==", flush=True)
        else:
            test_psnr = None
            is_best = False
            print(f"== epoch {epoch} done in {time.time() - t0:.1f}s (no eval this epoch) ==", flush=True)

        ckpt = {"model": model.state_dict(), "epoch": epoch, "test_psnr": test_psnr,
                "best_psnr": best_psnr, "args": vars(args)}
        torch.save(ckpt, os.path.join(args.checkpoint_dir, "last.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))

        if scheduler:
            scheduler.step()


if __name__ == "__main__":
    main()
