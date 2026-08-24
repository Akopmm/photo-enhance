"""Image-adaptive 3D LUT color enhancement model.

Reimplementation of the architecture from Zeng et al., "Learning Image-Adaptive
3D Lookup Tables for High Performance Photo Enhancement in Real-time" (Apache-2.0,
https://github.com/HuiZeng/Image-Adaptive-3DLUT). The original repo applies its
3D LUT via a compiled CUDA/C++ trilinear-interpolation extension, which only runs
on CUDA GPUs. This version applies the LUT with torch.nn.functional.grid_sample
instead, a stock op that runs identically on CPU, Apple MPS, and CUDA -- so the
same weights train on Apple Silicon and run inference on a CPU-only host with no
GPU dependency anywhere.

Model = a small CNN ("Classifier") that looks at a downsampled image and predicts
N scalar blend weights, plus N learnable 3D LUTs (one identity-initialized, the
rest zero-initialized). The blend weights combine the N LUTs into one image-
specific LUT, which is then applied to the full-resolution input in a single
trilinear sample. Total size is a few hundred thousand parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _discriminator_block(in_filters, out_filters, normalization=False):
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1), nn.LeakyReLU(0.2)]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers


class WeightPredictor(nn.Module):
    """Small CNN predicting `n_luts` blend weights from a downsampled image.

    Same shape as the original repo's `Classifier`: 5 stride-2 conv blocks
    (3->16->32->64->128->128) over a 256x256 input, then a kernel-8 conv that
    collapses the final 8x8 feature map to a single 1x1 spatial output.
    """

    def __init__(self, n_luts=3):
        super().__init__()
        self.n_luts = n_luts
        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode="bilinear", align_corners=False),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *_discriminator_block(16, 32, normalization=True),
            *_discriminator_block(32, 64, normalization=True),
            *_discriminator_block(64, 128, normalization=True),
            *_discriminator_block(128, 128, normalization=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, n_luts, 8, padding=0),
        )

    def forward(self, x):
        out = self.model(x)  # (B, n_luts, 1, 1)
        return out.view(out.size(0), self.n_luts)


def _init_weights(m):
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.xavier_normal_(m.weight.data)
    elif "InstanceNorm2d" in name or "BatchNorm2d" in name:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class LUT3D(nn.Module):
    """A single learnable 3D color lookup table, shape (3, dim, dim, dim).

    Axis convention: LUT[:, i, j, k] is the (R,G,B) output for input index
    (i,j,k) along the (B,G,R) axes respectively (note the order -- R varies
    along the *last* tensor axis, B along the *first*), indices normalized
    over [0, dim-1] <-> input value [0, 1]. This matches the axis convention
    of the original Image-Adaptive-3DLUT repo's own IdentityLUT33.txt
    (verified directly against its contents, not assumed) so pretrained
    weights from that repo load and run correctly here, not just structurally
    but numerically.
    """

    def __init__(self, dim=33, identity=True):
        super().__init__()
        self.dim = dim
        if identity:
            coords = torch.linspace(0, 1, dim)
            r = coords.view(1, 1, dim).expand(dim, dim, dim)
            g = coords.view(1, dim, 1).expand(dim, dim, dim)
            b = coords.view(dim, 1, 1).expand(dim, dim, dim)
            lut = torch.stack([r, g, b], dim=0)  # (3, dim, dim, dim)
        else:
            lut = torch.zeros(3, dim, dim, dim)
        self.lut = nn.Parameter(lut)


def apply_lut(lut, img):
    """Apply a (3, dim, dim, dim) LUT to a (B, 3, H, W) image in [0, 1] via
    trilinear grid_sample. Returns (B, 3, H, W)."""
    B, C, H, W = img.shape
    volume = lut.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, 3, dim, dim, dim)

    r = img[:, 0].clamp(0, 1)
    g = img[:, 1].clamp(0, 1)
    b = img[:, 2].clamp(0, 1)

    # grid_sample's grid axis order (x, y, z) samples the input volume's
    # (W, H, D) axes. Our volume's D/H/W axes are B/G/R respectively (see
    # LUT3D docstring), so grid x<-R, y<-G, z<-B.
    grid = torch.stack([r * 2 - 1, g * 2 - 1, b * 2 - 1], dim=-1)  # (B, H, W, 3)
    grid = grid.unsqueeze(1)  # (B, 1, H, W, 3) -> D_out=1

    out = F.grid_sample(volume, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.squeeze(2)  # (B, 3, H, W)


class AdaptiveLUTModel(nn.Module):
    """Full model: predicts per-image blend weights over `n_luts` basis LUTs,
    fuses them into one LUT, applies it to the input image."""

    def __init__(self, n_luts=3, lut_dim=33):
        super().__init__()
        self.n_luts = n_luts
        self.predictor = WeightPredictor(n_luts=n_luts)
        # First LUT starts as identity (a no-op color transform); the rest
        # start at zero, so the model starts close to "pass the image through
        # unchanged" and learns corrections from there.
        self.luts = nn.ModuleList(
            [LUT3D(dim=lut_dim, identity=(i == 0)) for i in range(n_luts)]
        )

    def init_weights(self):
        self.predictor.apply(_init_weights)
        # Bias the predictor's initial output toward selecting the identity
        # LUT (weight ~1 on LUT0, ~0 elsewhere) so training starts from a
        # near-identity transform rather than a random color mess.
        final_conv = self.predictor.model[-1]
        with torch.no_grad():
            final_conv.bias.zero_()
            final_conv.bias[0] = 1.0

    def fused_lut(self, weights):
        """weights: (n_luts,) for a single image -> combined (3,dim,dim,dim) LUT."""
        fused = weights[0] * self.luts[0].lut
        for i in range(1, self.n_luts):
            fused = fused + weights[i] * self.luts[i].lut
        return fused

    def forward(self, img):
        """Training-time forward: applies each basis LUT to the batch, then
        blends the *output images* per-sample by the predicted weights. This
        matches the original paper's training path and lets each sample in a
        batch use different weights without building one fused LUT per sample."""
        weights = self.predictor(img)  # (B, n_luts)
        outputs = [apply_lut(self.luts[i].lut, img) for i in range(self.n_luts)]
        combined = torch.zeros_like(img)
        for i in range(self.n_luts):
            w = weights[:, i].view(-1, 1, 1, 1)
            combined = combined + w * outputs[i]
        return combined, weights

    @torch.no_grad()
    def infer(self, img):
        """Inference-time forward for a single image (B=1): fuse the LUT
        tensors first, then apply once. Equivalent to forward() but does one
        trilinear pass instead of n_luts of them -- cheaper for a lone image."""
        weights = self.predictor(img).squeeze(0)  # (n_luts,)
        lut = self.fused_lut(weights)
        return apply_lut(lut, img), weights


class TotalVariation3D(nn.Module):
    """Smoothness + monotonicity regularizer on a 3D LUT, matching the
    original paper: penalizes large jumps between adjacent LUT cells (TV) and
    penalizes non-monotonic transitions along each axis (mn), which keeps the
    learned color transform visually smooth instead of noisy/banded."""

    def forward(self, lut):
        dr = lut[:, :, :, :-1] - lut[:, :, :, 1:]
        dg = lut[:, :, :-1, :] - lut[:, :, 1:, :]
        db = lut[:, :-1, :, :] - lut[:, 1:, :, :]
        tv = (dr ** 2).mean() + (dg ** 2).mean() + (db ** 2).mean()
        mono = F.relu(dr).mean() + F.relu(dg).mean() + F.relu(db).mean()
        return tv, mono
