"""Train a CIFAR-10 classifier.

    python -m src.train                                  # ResNet-18, 30 epochs
    python -m src.train --model simplecnn --epochs 20    # baseline
    python -m src.train --fast-dev-run                   # 1-minute smoke test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from .data import CLASSES, get_loaders
from .model import build_model, count_parameters
from .utils import AverageMeter, get_device, plot_history, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train NeuroVision-XAI")
    p.add_argument("--model", default="resnet18", choices=["resnet18", "simplecnn"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1, help="peak LR of the OneCycle schedule")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--fast-dev-run", action="store_true",
                   help="1 epoch on 512 images to check that everything works")
    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scaler=None, amp=False):
    train = optimizer is not None
    model.train(train)
    loss_m, acc_m = AverageMeter(), AverageMeter()
    bar = tqdm(loader, leave=False, desc="train" if train else "eval ")
    with torch.set_grad_enabled(train):
        for x, y in bar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            acc = (out.argmax(1) == y).float().mean().item() * 100
            loss_m.update(loss.item(), x.size(0))
            acc_m.update(acc, x.size(0))
            bar.set_postfix(loss=f"{loss_m.avg:.3f}", acc=f"{acc_m.avg:.1f}")
    return loss_m.avg, acc_m.avg


def main():
    args = parse_args()
    if args.fast_dev_run:
        args.epochs, args.workers = 1, 0
    set_seed(args.seed)
    device = get_device()
    amp = device.type == "cuda" and not args.no_amp
    print(f"Device: {device} | mixed precision: {amp}")

    train_loader, val_loader, _ = get_loaders(
        args.data_dir, args.batch_size, num_workers=args.workers, seed=args.seed,
        subset=512 if args.fast_dev_run else None,
    )
    model = build_model(args.model, len(CLASSES)).to(device)
    print(f"Model: {args.model} | trainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_acc, start = 0.0, time.time()

    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_loader, criterion, device, optimizer, scheduler, scaler, amp)
        vl, va = run_epoch(model, val_loader, criterion, device, amp=amp)
        for k, v in zip(history, (tl, ta, vl, va)):
            history[k].append(v)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tl:.3f} acc {ta:.2f} | "
              f"val loss {vl:.3f} acc {va:.2f}")
        if va >= best_acc:
            best_acc = va
            torch.save({"model_name": args.model, "model_state": model.state_dict(),
                        "val_acc": va, "epoch": epoch, "classes": CLASSES},
                       out_dir / "best.pt")

    mins = (time.time() - start) / 60
    print(f"Done in {mins:.1f} min | best val acc {best_acc:.2f}% -> {out_dir / 'best.pt'}")
    save_json(history, "outputs/history.json")
    plot_history(history, "outputs/training_curves.png")


if __name__ == "__main__":
    main()
