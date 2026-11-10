"""main.py - builds the data loaders and starts the training."""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from args import get_args
from dataset import FemurDataset
from trainer import Trainer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = get_args()
    seed_everything(args.seed)
    csv_dir = Path(args.csv_dir) if args.csv_dir else Path(args.data_dir) / "CSVs"

    train_ds = FemurDataset(csv_dir / "train.csv", args.data_dir, args.img_size, augment=bool(args.augment))
    val_ds = FemurDataset(csv_dir / "val.csv", args.data_dir, args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=len(train_ds) > args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    Trainer(args, train_loader, val_loader).fit()


if __name__ == "__main__":
    main()
