"""
Transfer learning training loop for the PropertyImageClassifier.

Usage:
    python training/train.py --data_dir ./data --epochs 20 --batch_size 32

The script:
  1. Loads a ResNet-50 backbone (frozen) with dual classification heads.
  2. Trains on the combined room-type + condition loss.
  3. Saves the best checkpoint (by validation room accuracy) to checkpoints/model.pth.
  4. Prints a final accuracy report.
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


def train(data_dir: str, epochs: int, batch_size: int, lr: float, device: str):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_ds = PropertyImageDataset(data_dir, split="train")
    val_ds = PropertyImageDataset(data_dir, split="test")
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(freeze_backbone=True).to(device)

    # Only train the heads initially
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    room_criterion = nn.CrossEntropyLoss()
    condition_criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, "model.pth")

    for epoch in range(1, epochs + 1):
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
            f"Epoch {epoch:3d}/{epochs} | Loss: {epoch_loss:.4f} | "
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.device)
