"""
NIH ChestX-ray14 Data Loader (Placeholder).

This module provides a placeholder dataset class and dataloader factory
for the NIH ChestX-ray14 dataset. To be implemented with real data loading
logic once the dataset is downloaded and preprocessed.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class NIHChestXrayDataset(Dataset):
    """
    Placeholder dataset for NIH ChestX-ray14.

    Expected data layout:
        data_root/
        ├── images/
        │   ├── 00000001_000.png
        │   ├── 00000001_001.png
        │   └── ...
        └── Data_Entry_2017_v2020.csv

    Args:
        data_root: Root directory containing images/ and CSV.
        split: One of 'train', 'val', 'test'.
        transform: Optional torchvision transform pipeline.
    """

    DISEASE_LABELS = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia",
    ]

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform=None,
    ):
        self.data_root = data_root
        self.split = split
        self.transform = transform

        # TODO: Parse Data_Entry_2017_v2020.csv and build image list + labels
        # For now, generate fake entries for testing
        self.image_paths = []
        self.labels = []

        print(f"[NIHChestXrayDataset] Placeholder initialized for split='{split}'")
        print(f"  → data_root: {data_root}")
        print(f"  → Found {len(self)} samples (placeholder)")

    def __len__(self) -> int:
        return max(len(self.image_paths), 100)  # Fake 100 samples for skeleton

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Replace with real image loading from CSV
        # Placeholder: return random 3×224×224 image and random multi-hot labels
        image = torch.randn(3, 224, 224)
        label = torch.zeros(14)
        label[np.random.choice(14, size=np.random.randint(0, 4), replace=False)] = 1.0
        return image, label


def create_dataloader(
    data_root: str,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    transform=None,
) -> DataLoader:
    """
    Factory function to create a DataLoader for NIH ChestX-ray14.

    Args:
        data_root: Root directory of the dataset.
        split: 'train', 'val', or 'test'.
        batch_size: Batch size for the loader.
        num_workers: Number of parallel data loading workers.
        transform: Optional image transform pipeline.

    Returns:
        A PyTorch DataLoader instance.
    """
    dataset = NIHChestXrayDataset(
        data_root=data_root,
        split=split,
        transform=transform,
    )

    shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
