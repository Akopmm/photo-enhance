"""FiveK paired dataset loader.

Expects the preprocessed 480p sRGB FiveK release used by the Image-Adaptive-3DLUT
paper (input JPEGs + Expert-C-retouched JPEGs, already resized/aligned), laid out as:

  <root>/train_input.txt        # basenames (no extension), one per line
  <root>/train_label.txt        # more basenames, combined with train_input.txt for training
  <root>/test.txt                # held-out basenames
  <root>/input/JPG/480p/<name>.jpg
  <root>/expertC/JPG/480p/<name>.jpg

This is the exact structure the dataset's Google Drive/OneDrive/Baidu release
already ships in (see download_fivek.py), so it's used as-is rather than
reshuffled into a different convention.
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# Fixed size every training crop gets resized to after the random-ratio crop.
# The original repo's own code has this exact option commented out
# (`resized_crop(..., (320,320))`) but ships with variable-size crops and
# batch_size=1 by default. Fixing the size here means samples in a batch
# share a shape, so batch_size can be >1 -- each training step then submits
# real parallel work to the GPU instead of one tiny image at a time, which
# for a model this small is almost pure per-step dispatch overhead rather
# than compute. Eval/test stays at full resolution (see train.py) since
# that's evaluated one image at a time regardless.
TRAIN_CROP_SIZE = 256


class FiveKDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = root
        self.split = split

        def read_list(name):
            path = os.path.join(root, name)
            with open(path) as f:
                return [line.strip() for line in f if line.strip()]

        if split == "train":
            names = read_list("train_input.txt") + read_list("train_label.txt")
        elif split == "test":
            names = read_list("test.txt")
        else:
            raise ValueError(f"unknown split: {split}")

        self.inputs = [os.path.join(root, "input", "JPG", "480p", n + ".jpg") for n in names]
        self.targets = [os.path.join(root, "expertC", "JPG", "480p", n + ".jpg") for n in names]
        self.names = names

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        img_in = Image.open(self.inputs[idx]).convert("RGB")
        img_out = Image.open(self.targets[idx]).convert("RGB")

        if self.split == "train":
            w, h = img_in.size
            ratio_h = random.uniform(0.6, 1.0)
            ratio_w = random.uniform(0.6, 1.0)
            crop_h, crop_w = round(h * ratio_h), round(w * ratio_w)
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            img_in = TF.crop(img_in, top, left, crop_h, crop_w)
            img_out = TF.crop(img_out, top, left, crop_h, crop_w)

            size = (TRAIN_CROP_SIZE, TRAIN_CROP_SIZE)
            img_in = TF.resize(img_in, size)
            img_out = TF.resize(img_out, size)

            if random.random() > 0.5:
                img_in = TF.hflip(img_in)
                img_out = TF.hflip(img_out)

            img_in = TF.adjust_brightness(img_in, random.uniform(0.8, 1.2))
            img_in = TF.adjust_saturation(img_in, random.uniform(0.8, 1.2))

        return {
            "input": TF.to_tensor(img_in),
            "target": TF.to_tensor(img_out),
            "name": self.names[idx],
        }
