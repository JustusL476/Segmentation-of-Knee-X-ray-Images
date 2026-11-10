"""dataset.py - read-image function and the Dataset class."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_image(path, size, is_mask=False):
    img = Image.open(path).convert("L")
    if is_mask:
        img = img.resize((size, size), Image.NEAREST)
        arr = np.array(img, dtype=np.float32)
        return (arr > 0).astype(np.float32)
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


class FemurDataset(Dataset):
    def __init__(self, csv_path, data_dir, img_size=256, augment=False):
        self.df = pd.read_csv(csv_path, keep_default_na=False)
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def has_masks(self):
        return not (self.df["masks"] == "None").any()

    def _augment(self, img, mask):
        if random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        a = random.uniform(0.85, 1.15)
        b = random.uniform(-0.08, 0.08)
        img = np.clip(img * a + b, 0.0, 1.0)
        s = int(self.img_size * 0.1)
        dx, dy = random.randint(-s, s), random.randint(-s, s)
        img = np.roll(img, (dy, dx), axis=(0, 1))
        mask = np.roll(mask, (dy, dx), axis=(0, 1))
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = read_image(self.data_dir / row["xrays"], self.img_size)
        if row["masks"] == "None": 
            mask = np.zeros_like(img)
        else:
            mask = read_image(self.data_dir / row["masks"], self.img_size, is_mask=True)
            if self.augment:
                img, mask = self._augment(img, mask)
        img_t = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)    
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)  
        return img_t, mask_t, row["xrays"]
