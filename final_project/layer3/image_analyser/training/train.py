"""
Transfer learning training loop for the PropertyImageClassifier.

Usage (recommended — two-phase training):
    python training/train.py --data_dir ./data --epochs 20 --unfreeze_epoch 7 --batch_size 32

Two-phase strategy:
  Phase 1 (epochs 1 to unfreeze_epoch): backbone fully frozen, only heads train.
  Phase 2 (unfreeze_epoch onward): last ResNet block (layer4) unfreezes with a
  10x lower LR — this fine-tunes high-level visual features for real estate images
  and significantly improves accuracy over heads-only training.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.model import build_model
from training.dataset import PropertyImageDataset

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")


def _unfreeze_last_block(model, backbone_lr: float, head_lr: float):
    """Unfreeze ResNet layer4 and return a new optimizer with differential LRs."""
    for name, param in model.backbone.named_parameters():
        if "7" in name:  # layer4 is the 8th child (index 7) in the Sequential backbone
            param.requires_grad = True

    return torch.optim.Adam([
        {"params": [p for n, p in model.backbone.named_parameters() if p.requires_grad], "lr": backbone_lr},
        {"params": model.room_head.parameters(), "lr": head_lr},
        {"params": model.condition_head.parameters(), "lr": head_lr},
    ])


def train(data_dir: str, epochs: int, batch_size: int, lr: float, device: str, unfreeze_epoch: int):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_ds = PropertyImageDataset(data_dir, split="train")
    val_ds = PropertyImageDataset(data_dir, split="test")
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(freeze_backbone=True).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    room_criterion = nn.CrossEntropyLoss()
    condition_criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, "model.pth")
    phase = 1

    for epoch in range(1, epochs + 1):

        # Switch to phase 2: unfreeze last backbone block
        if epoch == unfreeze_epoch and unfreeze_epoch > 0:
            print(f"\n--- Phase 2: unfreezing ResNet layer4 (backbone_lr={lr/10:.5f}) ---\n")
            optimizer = _unfreeze_last_block(model, backbone_lr=lr / 10, head_lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            phase = 2

        # --- Train ---
        model.train()
        running_loss = 0.0
        for imgs, room_labels, cond_labels in train_loader:
            imgs = imgs.to(device)
            room_labels = room_labels.to(device)
            cond_labels = cond_labels.to(device)

            optimizer.zero_grad()
            room_logits, cond_logits = model(imgs)
            loss = room_criterion(room_logits, room_labels) + 0.5 * condition_criterion(cond_logits, cond_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_ds)

        # --- Validate ---
        model.eval()
        room_correct = 0
        cond_correct = 0
        total = 0
        with torch.no_grad():
            for imgs, room_labels, cond_labels in val_loader:
                imgs = imgs.to(device)
                room_labels = room_labels.to(device)
                cond_labels = cond_labels.to(device)
                room_logits, cond_logits = model(imgs)
                room_correct += (room_logits.argmax(1) == room_labels).sum().item()
                cond_correct += (cond_logits.argmax(1) == cond_labels).sum().item()
                total += imgs.size(0)

        val_room_acc = room_correct / total
        val_cond_acc = cond_correct / total
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{epochs} [phase {phase}] | Loss: {epoch_loss:.4f} | "
            f"Room Acc: {val_room_acc:.3f} | Cond Acc: {val_cond_acc:.3f}"
        )

        if val_room_acc > best_val_acc:
            best_val_acc = val_room_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  → Saved best checkpoint (room acc={best_val_acc:.3f})")

    print(f"\nTraining complete. Best room val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint saved to: {best_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unfreeze_epoch", type=int, default=7,
                        help="Epoch at which to unfreeze ResNet layer4. Set to 0 to keep backbone fully frozen.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device} | Unfreeze at epoch: {args.unfreeze_epoch}")
    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.device, args.unfreeze_epoch)
