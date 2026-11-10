"""evaluate.py - final evaluation: Dice score, predicted masks and visualizations."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from args import get_args
from dataset import FemurDataset
from model import UNet
from trainer import get_device
from utils import dice_score_from_logits


@torch.no_grad()
def predict_split(model, loader, device, threshold):
    model.eval()
    names, imgs_all, preds_all, gts_all, dices = [], [], [], [], []
    for imgs, masks, paths in loader:
        logits = model(imgs.to(device))
        probs = torch.sigmoid(logits).cpu()
        preds = (probs > threshold).float()
        for i in range(imgs.size(0)):
            names.append(paths[i])
            imgs_all.append(imgs[i, 0].numpy())
            preds_all.append(preds[i, 0].numpy())
            gts_all.append(masks[i, 0].numpy())
            dices.append(dice_score_from_logits(logits[i:i + 1].cpu(), masks[i:i + 1]))
    return names, imgs_all, preds_all, gts_all, dices


def save_predictions(out_dir, names, preds):
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, p in zip(names, preds):
        Image.fromarray((p * 255).astype(np.uint8)).save(out_dir / f"{Path(n).stem}_pred.png")


def plot_examples(path, imgs, preds, gts, dices, labeled, k):
    k = min(k, len(imgs))
    cols = 3 if labeled else 2
    fig, ax = plt.subplots(k, cols, figsize=(3.2 * cols, 3.2 * k), squeeze=False)
    for r in range(k):
        ax[r, 0].imshow(imgs[r], cmap="gray"); ax[r, 0].set_title("X-ray")
        if labeled:
            ax[r, 1].imshow(gts[r], cmap="gray"); ax[r, 1].set_title("Ground truth")
            ax[r, 2].imshow(preds[r], cmap="gray"); ax[r, 2].set_title(f"Prediction (Dice {dices[r]:.3f})")
        else:
            ax[r, 1].imshow(imgs[r], cmap="gray")
            ax[r, 1].imshow(np.ma.masked_where(preds[r] == 0, preds[r]), cmap="autumn", alpha=0.5)
            ax[r, 1].set_title("Predicted overlay")
        for a in ax[r]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    args = get_args()
    device = get_device(args.device)
    out = Path(args.output_dir)
    ckpt_path = args.checkpoint or out / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = UNet(bw=args.bw, depth=args.depth, n_inputs=1, n_classes=1).to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    print(f"Loaded {ckpt_path}")

    csv_dir = Path(args.csv_dir) if args.csv_dir else Path(args.data_dir) / "CSVs"
    rows = []
    for split in ("train", "val", "test"):
        ds = FemurDataset(csv_dir / f"{split}.csv", args.data_dir, args.img_size, augment=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        names, imgs, preds, gts, dices = predict_split(model, loader, device, args.threshold)
        labeled = ds.has_masks()
        save_predictions(out / "predictions" / split, names, preds)
        plot_examples(out / f"examples_{split}.png", imgs, preds, gts, dices, labeled, args.num_vis)
        if labeled:
            print(f"{split:5s}: mean Dice = {np.mean(dices):.4f}  (std {np.std(dices):.4f}, n={len(dices)})")
            rows += [{"split": split, "image": n, "dice": d} for n, d in zip(names, dices)]
        else:
            print(f"{split:5s}: no ground-truth masks -> saved {len(names)} predicted masks + overlays "
                  f"(Dice not computable)")
    if rows:
        pd.DataFrame(rows).to_csv(out / "dice_per_image.csv", index=False)
    print(f"Results saved in {out}")


if __name__ == "__main__":
    main()
