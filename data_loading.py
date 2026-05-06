#!/usr/bin/env python3
"""
Dataset loaders for the three benchmarks used in the paper (§ 3.1):
PneumoniaMNIST, BreastMNIST, Br35H.

All loaders return ``(train_ds, val_ds, test_ds)`` triples of PyTorch datasets
at ``size × size`` resolution, single-channel (grayscale), normalized as
requested. The MedMNIST datasets are downloaded automatically; Br35H must be
placed at ``<root>/{yes,no}/*.jpg``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import medmnist
import numpy as np
import torch
from medmnist import INFO
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision import transforms

IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

MEDMNIST_MEAN: List[float] = [0.5, 0.5, 0.5]
MEDMNIST_STD: List[float] = [0.5, 0.5, 0.5]


def _resolve_normalize(normalize: str) -> Tuple[List[float], List[float]]:
    if normalize == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    if normalize == "medmnist":
        return MEDMNIST_MEAN, MEDMNIST_STD
    raise ValueError(f"Unknown normalize={normalize}")


def make_transform(normalize: str) -> transforms.Compose:
    """Single-channel transform applied to MedMNIST datasets (already sized)."""
    mean, std = _resolve_normalize(normalize)
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean[0]], std=[std[0]]),
        ]
    )


def load_medmnist(name: str, size: int, normalize: str, root: str = "./data"):
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    tfm = make_transform(normalize)
    train_ds = DataClass(split="train", transform=tfm, download=True, root=root, size=size)
    val_ds = DataClass(split="val", transform=tfm, download=True, root=root, size=size)
    test_ds = DataClass(split="test", transform=tfm, download=True, root=root, size=size)
    return train_ds, val_ds, test_ds


class Br35HDataset(Dataset):
    """Br35H samples from a flat directory layout: ``<root>/{yes,no}/*.jpg``."""

    def __init__(self, root: str, transform=None) -> None:
        self.samples: List[Tuple[str, int]] = []
        for label_name, label in [("no", 0), ("yes", 1)]:
            for p in (Path(root) / label_name).glob("*.jpg"):
                self.samples.append((str(p), label))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fp, y = self.samples[idx]
        img = Image.open(fp).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def load_br35h(size: int, normalize: str, root: str):
    """
    Load Br35H from ``<root>/{yes,no}/*.jpg`` and produce a stratified
    80/10/10 train/val/test split. The split is fixed by ``random_state=42``
    so all models train and evaluate on identical samples.
    """
    mean, std = _resolve_normalize(normalize)
    tfm = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean[0]], std=[std[0]]),
        ]
    )

    full_ds = Br35HDataset(root=root, transform=tfm)
    targets = np.array([y for _, y in full_ds.samples])
    indices = np.arange(len(full_ds))

    train_idx, temp_idx = train_test_split(
        indices, train_size=0.8, stratify=targets, random_state=42,
    )
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=0.5, stratify=targets[temp_idx], random_state=42,
    )

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(full_ds, test_idx)
    return train_ds, val_ds, test_ds


def load_dataset(name: str, size: int, normalize: str, root: str = "./data"):
    if name == "br35h":
        return load_br35h(size=size, normalize=normalize, root=root)
    return load_medmnist(name=name, size=size, normalize=normalize, root=root)
