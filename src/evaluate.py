"""Evaluate a trained checkpoint on the CIFAR-10 test set.

    python -m src.evaluate --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .data import CLASSES, get_loaders
from .utils import get_device, load_checkpoint, plot_confusion_matrix, save_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Evaluate NeuroVision-XAI")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    _, _, test_loader = get_loaders(args.data_dir, args.batch_size, num_workers=args.workers)

    preds, targets = [], []
    for x, y in test_loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        targets.append(y)
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()

    acc = float((preds == targets).mean() * 100)
    print(f"Model: {ckpt['model_name']} | Test accuracy: {acc:.2f}%\n")
    report = classification_report(targets, preds, target_names=CLASSES, digits=3, output_dict=True)
    print(classification_report(targets, preds, target_names=CLASSES, digits=3))

    cm = confusion_matrix(targets, preds)
    plot_confusion_matrix(cm, "outputs/confusion_matrix.png")
    save_json({"model": ckpt["model_name"], "test_accuracy": acc, "report": report},
              "outputs/test_metrics.json")

    off = cm.copy()
    np.fill_diagonal(off, 0)
    print("Top confusions (true -> predicted):")
    for flat in np.argsort(off, axis=None)[::-1][:5]:
        i, j = divmod(int(flat), 10)
        print(f"  {CLASSES[i]:>10} -> {CLASSES[j]:<10} {off[i, j]} images")
    print("\nSaved outputs/confusion_matrix.png and outputs/test_metrics.json")


if __name__ == "__main__":
    main()
