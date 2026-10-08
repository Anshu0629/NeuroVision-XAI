"""CIFAR-10 data loading with augmentation and a reproducible train/val split."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.25),
        ])
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def get_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_size: int = 5000,
    num_workers: int = 2,
    seed: int = 42,
    subset: int | None = None,
):
    """Return (train_loader, val_loader, test_loader).

    `subset` limits every split to N images - handy for a quick smoke test.
    """
    train_full = datasets.CIFAR10(data_dir, train=True, download=True,
                                  transform=get_transforms(True))
    val_full = datasets.CIFAR10(data_dir, train=True, download=True,
                                transform=get_transforms(False))
    test_set = datasets.CIFAR10(data_dir, train=False, download=True,
                                transform=get_transforms(False))

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_full), generator=g).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    train_set = Subset(train_full, train_idx)
    val_set = Subset(val_full, val_idx)  # same images, no augmentation

    if subset:
        train_set = Subset(train_set, range(min(subset, len(train_set))))
        val_set = Subset(val_set, range(min(subset, len(val_set))))
        test_set = Subset(test_set, range(min(subset, len(test_set))))

    pin = torch.cuda.is_available()
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    return (
        DataLoader(train_set, shuffle=True, drop_last=True, **kw),
        DataLoader(val_set, shuffle=False, **kw),
        DataLoader(test_set, shuffle=False, **kw),
    )
