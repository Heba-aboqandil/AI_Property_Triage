"""
Inference module: load the trained checkpoint and run predictions on an image URL.

Returns room_type, condition_score (1–5 or None), and confidence.
When the top-class confidence < CONFIDENCE_THRESHOLD, the prediction is
considered uncertain and condition_score is set to None.
"""

import os
import io
from functools import lru_cache
from typing import Optional

import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from app.model import PropertyImageClassifier, ROOM_CLASSES, CONDITION_CLASSES, build_model

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "/service/checkpoints/model.pth")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@lru_cache(maxsize=1)
def load_model() -> PropertyImageClassifier:
    model = build_model(freeze_backbone=False)
    if os.path.exists(CHECKPOINT_PATH):
        state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
    else:
        print(f"WARNING: No checkpoint found at {CHECKPOINT_PATH}. Using random weights.")
    model.to(DEVICE)
    model.eval()
    return model


def _fetch_image(url: str) -> Image.Image:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    return img


def _predict_from_pil(img: Image.Image) -> dict:
    tensor = _TRANSFORM(img).unsqueeze(0).to(DEVICE)

    model = load_model()
    with torch.no_grad():
        room_logits, condition_logits = model(tensor)

    room_probs = F.softmax(room_logits, dim=1)[0]
    condition_probs = F.softmax(condition_logits, dim=1)[0]

    room_conf, room_idx = torch.max(room_probs, dim=0)
    condition_idx = torch.argmax(condition_probs, dim=0)

    confidence = float(room_conf.item())
    room_type = ROOM_CLASSES[int(room_idx.item())]
    condition_score: Optional[int] = (
        CONDITION_CLASSES[int(condition_idx.item())]
        if confidence >= CONFIDENCE_THRESHOLD
        else None
    )

    if confidence < CONFIDENCE_THRESHOLD:
        room_type = "uncertain"

    return {
        "room_type": room_type,
        "condition_score": condition_score,
        "confidence": round(confidence, 4),
    }


def predict(image_url: str) -> dict:
    img = _fetch_image(image_url)
    return _predict_from_pil(img)


def predict_from_bytes(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _predict_from_pil(img)
