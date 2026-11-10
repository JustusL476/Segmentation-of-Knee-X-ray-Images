"""trainer.py - model initialization, training loop and validation loop."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from model import UNet
from utils import dice_loss_from_logits, dice_score_from_logits


def get_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Trainer:
    def __init__(self, args, train_loader, val_loader):
        self.args = args
        self.device = get_device(args.device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.out = Path(args.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        # ---- model initialization ----
        self.model = UNet(bw=args.bw, depth=args.depth, n_inputs=1, n_classes=1).to(self.device)
        if args.checkpoint:  # fine-tuning: start from saved weights
            ckpt = torch.load(args.checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
            print(f"Loaded weights from {args.checkpoint}")
        if args.freeze_encoder:
            for p in self.model.encoder_parameters():
                p.requires_grad = False
            print("Encoder frozen")

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode="max", factor=0.5, patience=5)
        self.bce = nn.BCEWithLogitsLoss()
        self.history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
        self.best_dice = -1.0

    def _loss(self, logits, masks):
        return self.bce(logits, masks) + dice_loss_from_logits(logits, masks)

    def _run_epoch(self, loader, train):
        self.model.train(train)
        tot_loss = tot_dice = n = 0
        with torch.set_grad_enabled(train):
            for imgs, masks, _ in loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                logits = self.model(imgs)
                loss = self._loss(logits, masks)
                if train:
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                b = imgs.size(0)
                tot_loss += loss.item() * b
                tot_dice += dice_score_from_logits(logits.detach(), masks) * b
                n += b
        return tot_loss / n, tot_dice / n

    def fit(self):
        print(f"Training on {self.device} | train={len(self.train_loader.dataset)} "
              f"val={len(self.val_loader.dataset)}")
        for ep in range(1, self.args.epochs + 1):
            tl, td = self._run_epoch(self.train_loader, train=True)
            vl, vd = self._run_epoch(self.val_loader, train=False)
            self.sched.step(vd)
            for k, v in zip(self.history, (tl, vl, td, vd)):
                self.history[k].append(v)
            tag = ""
            if vd > self.best_dice:  # keep the best model by validation Dice
                self.best_dice = vd
                torch.save({"model": self.model.state_dict(), "args": vars(self.args),
                            "epoch": ep, "val_dice": vd}, self.out / "best_model.pt")
                tag = "  <- best"
            print(f"Epoch {ep:3d}/{self.args.epochs} | train loss {tl:.4f} dice {td:.4f} | "
                  f"val loss {vl:.4f} dice {vd:.4f}{tag}")
        self._save_history()
        print(f"Best validation Dice: {self.best_dice:.4f}  (saved to {self.out / 'best_model.pt'})")

    def _save_history(self):
        (self.out / "history.json").write_text(json.dumps(self.history, indent=2))
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(self.history["train_loss"], label="train"); ax[0].plot(self.history["val_loss"], label="val")
        ax[0].set_title("Loss (BCE + Dice)"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(self.history["train_dice"], label="train"); ax[1].plot(self.history["val_dice"], label="val")
        ax[1].set_title("Dice score"); ax[1].set_xlabel("epoch"); ax[1].legend()
        fig.tight_layout()
        fig.savefig(self.out / "training_curves.png", dpi=150)
        plt.close(fig)
