"""Small helpers shared by the training / evaluation / inference scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # works on servers / Colab without a display
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import CLASSES
from .model import build_model


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.n += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.n, 1)


def load_checkpoint(path: str, device: torch.device):
    """Rebuild the model stored in a checkpoint and return (model, ckpt_dict)."""
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["model_name"], num_classes=len(CLASSES))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


def plot_history(history: dict, out_path: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, history["train_loss"], label="train")
    ax[0].plot(epochs, history["val_loss"], label="val")
    ax[0].set(title="Loss", xlabel="epoch")
    ax[1].plot(epochs, history["train_acc"], label="train")
    ax[1].plot(epochs, history["val_acc"], label="val")
    ax[1].set(title="Accuracy (%)", xlabel="epoch")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, out_path: str) -> None:
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES,
           yticklabels=CLASSES, xlabel="Predicted", ylabel="True",
           title="Normalized confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=7)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
