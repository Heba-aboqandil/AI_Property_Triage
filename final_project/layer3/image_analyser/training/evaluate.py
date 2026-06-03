"""
Evaluate the trained model on the test split and print a classification report.

Usage:
    python training/evaluate.py --data_dir ./data --checkpoint checkpoints/model.pth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from app.model import build_model, ROOM_CLASSES, CONDITION_CLASSES
from training.dataset import PropertyImageDataset


def evaluate(data_dir: str, checkpoint: str, device: str):
    test_ds = PropertyImageDataset(data_dir, split="test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    model = build_model(freeze_backbone=False).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_room_preds, all_room_true = [], []
    all_cond_preds, all_cond_true = [], []

    with torch.no_grad():
        for imgs, room_labels, cond_labels in test_loader:
            imgs = imgs.to(device)
            room_logits, cond_logits = model(imgs)
            all_room_preds.extend(room_logits.argmax(1).cpu().tolist())
            all_room_true.extend(room_labels.tolist())
            all_cond_preds.extend(cond_logits.argmax(1).cpu().tolist())
            all_cond_true.extend(cond_labels.tolist())

    print("=== Room Type Classification Report ===")
    print(classification_report(all_room_true, all_room_preds, target_names=ROOM_CLASSES))

    cond_names = [str(s) for s in CONDITION_CLASSES]
    print("=== Condition Score Classification Report ===")
    print(classification_report(all_cond_true, all_cond_preds, target_names=cond_names))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--checkpoint", default="checkpoints/model.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    evaluate(args.data_dir, args.checkpoint, args.device)
