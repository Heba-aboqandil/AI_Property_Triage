"""
Custom dataset for property image classification.

Expected directory structure:
    data/
      train/
        kitchen/      (images)
        bathroom/
        living_room/
        bedroom/
        exterior/
        other/
      val/
        kitchen/
        ...
      test/
        kitchen/
        ...

Each image filename must include a condition score suffix: _c1 through _c5.
Example:  kitchen_001_c4.jpg  →  room=kitchen, condition=4

If the suffix is absent the condition label defaults to 3 (average).
"""

import os
import re
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from app.model import ROOM_CLASSES, CONDITION_CLASSES

ROOM_TO_IDX = {name: i for i, name in enumerate(ROOM_CLASSES)}
CONDITION_TO_IDX = {score: i for i, score in enumerate(CONDITION_CLASSES)}

_CONDITION_RE = re.compile(r"_c([1-5])", re.IGNORECASE)

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PropertyImageDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train"):
        self.root = Path(root_dir) / split
        assert self.root.exists(), f"Directory not found: {self.root}"

        self.transform = TRAIN_TRANSFORM if split == "train" else EVAL_TRANSFORM
        self.samples: list[tuple[Path, int, int]] = []  # (path, room_idx, condition_idx)

        for room_name in ROOM_CLASSES:
            room_dir = self.root / room_name
            if not room_dir.exists():
                continue
            for img_path in room_dir.iterdir():
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                    continue
                room_idx = ROOM_TO_IDX[room_name]
                match = _CONDITION_RE.search(img_path.stem)
                condition_score = int(match.group(1)) if match else 3
                condition_idx = CONDITION_TO_IDX[condition_score]
                self.samples.append((img_path, room_idx, condition_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, room_idx, condition_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, room_idx, condition_idx
