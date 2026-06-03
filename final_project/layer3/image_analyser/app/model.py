"""
Dual-head property image classifier built on a frozen ResNet-50 backbone.

Head 1 — room_type:  8-class classification
    classes: balcony, bathroom, bedroom, building_exterior,
             garden, kitchen_dining, living_room, not_real_estate

Head 2 — condition:  5-class classification (maps to score 1–5)
    classes: 1_poor, 2_fair, 3_average, 4_good, 5_excellent
"""

import torch
import torch.nn as nn
from torchvision import models

ROOM_CLASSES = [
    "balcony", "bathroom", "bedroom", "building_exterior",
    "garden", "kitchen_dining", "living_room", "not_real_estate",
]
CONDITION_CLASSES = [1, 2, 3, 4, 5]

NUM_ROOM_CLASSES = len(ROOM_CLASSES)
NUM_CONDITION_CLASSES = len(CONDITION_CLASSES)


class PropertyImageClassifier(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # Keep everything except the final FC layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        feature_dim = backbone.fc.in_features  # 2048 for ResNet-50

        # Room type head
        self.room_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, NUM_ROOM_CLASSES),
        )

        # Condition score head
        self.condition_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, NUM_CONDITION_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        room_logits = self.room_head(features)
        condition_logits = self.condition_head(features)
        return room_logits, condition_logits


def build_model(freeze_backbone: bool = True) -> PropertyImageClassifier:
    return PropertyImageClassifier(freeze_backbone=freeze_backbone)
