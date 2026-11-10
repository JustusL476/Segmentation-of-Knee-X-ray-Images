"""args.py - command-line arguments that configure the whole pipeline."""
import argparse


def get_args():
    p = argparse.ArgumentParser(description="Femur segmentation with U-Net")

    # directories
    p.add_argument("--data_dir", type=str, default="data",
                   help="Folder that contains xrays/, masks/ and CSVs/")
    p.add_argument("--csv_dir", type=str, default=None,
                   help="Folder with train/val/test CSVs (default: <data_dir>/CSVs)")
    p.add_argument("--output_dir", type=str, default="outputs",
                   help="Where checkpoints, curves and predictions are saved")

    # data
    p.add_argument("--img_size", type=int, default=256, help="Images/masks are resized to img_size x img_size")
    p.add_argument("--augment", type=int, default=1, help="1 = augment training data, 0 = off")
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (keep 0 on Windows)")

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--bw", type=int, default=24, help="U-Net basic width (doubled at each level)")
    p.add_argument("--depth", type=int, default=6,
                   help="U-Net depth; img_size should be divisible by 2**depth (use 5 for a lighter/faster model)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")

    # fine-tuning / resuming
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a .pt file to start from (fine-tuning) or to evaluate")
    p.add_argument("--freeze_encoder", type=int, default=0,
                   help="1 = freeze encoder when fine-tuning from a checkpoint")

    # evaluation
    p.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for predicted masks")
    p.add_argument("--num_vis", type=int, default=8, help="How many example predictions to plot")

    return p.parse_args()
