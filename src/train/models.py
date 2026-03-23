"""
DenseNet-121 based Multi-label Classification Model for Chest X-ray CAD.

This module defines the DenseNet121CAD model architecture, which uses a pretrained
DenseNet-121 backbone with a modified classifier head for 14-disease multi-label
classification on the NIH ChestX-ray14 dataset.
"""

import torch
import torch.nn as nn
from torchvision import models


# NIH ChestX-ray14 disease labels (official order)
DISEASE_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


class DenseNet121CAD(nn.Module):
    """
    DenseNet-121 architecture modified for multi-label chest X-ray classification.

    - Backbone: DenseNet-121 pretrained on ImageNet
    - Head: Linear(1024 → 14) + Sigmoid for multi-label probabilities
    - Input: (B, 3, 224, 224) tensor
    - Output: (B, 14) tensor of probabilities ∈ [0, 1]
    """

    NUM_CLASSES = 14

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load pretrained DenseNet-121 backbone
        self.backbone = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Get the number of input features for the original classifier
        num_features = self.backbone.classifier.in_features  # 1024

        # Replace the classifier head for 14-class multi-label output
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, self.NUM_CLASSES),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Tensor of shape (B, 14) with disease probabilities.
        """
        return self.backbone(x)


if __name__ == "__main__":
    # Quick sanity check with a fake input
    model = DenseNet121CAD(pretrained=False)
    model.eval()

    fake_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(fake_input)

    print(f"Input shape : {fake_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("\nPer-disease probabilities:")
    for label, prob in zip(DISEASE_LABELS, output.squeeze().tolist()):
        print(f"  {label:<22s}: {prob:.4f}")
