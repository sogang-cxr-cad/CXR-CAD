"""
Image Preprocessing & Augmentation Transforms for Chest X-ray CAD.

Provides transform pipelines for training and inference, including:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Resizing to 224×224
- ImageNet normalization
"""

from typing import Tuple

import numpy as np
from PIL import Image, ImageOps

import torch
from torchvision import transforms


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def apply_clahe(image: Image.Image, clip_limit: float = 2.0) -> Image.Image:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a PIL image.

    Uses PIL's histogram equalization as a simplified placeholder.
    For production, use cv2.createCLAHE for true tile-based CLAHE.

    Args:
        image: Input PIL Image (any mode).
        clip_limit: CLAHE clip limit (reserved for cv2 implementation).

    Returns:
        Contrast-enhanced PIL Image in RGB mode.
    """
    # Convert to grayscale if needed for equalization
    if image.mode != "L":
        gray = image.convert("L")
    else:
        gray = image

    # Apply histogram equalization (simplified CLAHE placeholder)
    equalized = ImageOps.equalize(gray)

    # Convert back to RGB (3-channel) for model input
    return equalized.convert("RGB")


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Training-time transform pipeline with data augmentation.

    Pipeline: CLAHE → Resize → RandomHFlip → RandomRotation → ToTensor → Normalize
    """
    return transforms.Compose([
        transforms.Lambda(lambda img: apply_clahe(img)),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Inference-time transform pipeline (no augmentation).

    Pipeline: CLAHE → Resize → ToTensor → Normalize
    """
    return transforms.Compose([
        transforms.Lambda(lambda img: apply_clahe(img)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_single_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """
    Preprocess a single PIL image for model inference.

    Args:
        image: Input PIL Image.
        image_size: Target size (default 224).

    Returns:
        Tensor of shape (1, 3, 224, 224) ready for model input.
    """
    transform = get_inference_transforms(image_size)
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension
