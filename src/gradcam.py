"""Grad-CAM: visual explanations for CNN decisions.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.

Usage:
    python -m src.gradcam --checkpoint checkpoints/best.pt --num-images 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import CLASSES, MEAN, STD, get_transforms
from .model import get_target_layer
from .utils import get_device, load_checkpoint


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def remove(self) -> None:
        self._handle.remove()

    @torch.enable_grad()
    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """Return (cam[N,H,W] in [0,1], probs[N,C], class_idx[N])."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            idx = logits.argmax(dim=1)
        else:
            idx = torch.full((x.size(0),), class_idx, device=x.device, dtype=torch.long)
        score = logits.gather(1, idx[:, None]).sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        flat = cam.flatten(1)
        mn, mx = flat.min(1)[0][:, None, None], flat.max(1)[0][:, None, None]
        cam = (cam - mn) / (mx - mn + 1e-8)
        return cam.detach().cpu().numpy(), logits.softmax(1).detach().cpu(), idx.cpu()


def denormalize(x: torch.Tensor) -> np.ndarray:
    """Normalized tensor [3,H,W] -> uint8 image [H,W,3]."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = (x.cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def overlay_cam(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45, size: int = 256) -> np.ndarray:
    """Blend a jet heat-map over an RGB uint8 image and upscale for display."""
    img_big = np.asarray(Image.fromarray(img).resize((size, size), Image.BICUBIC)).astype(float)
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    heat = plt.get_cmap("jet")(np.asarray(cam_img) / 255.0)[..., :3] * 255
    return ((1 - alpha) * img_big + alpha * heat).clip(0, 255).astype(np.uint8)


def explain_pil_image(model, pil_image: Image.Image, device, class_idx: int | None = None):
    """Classify a PIL image and return (probs dict, overlay uint8)."""
    pil_image = pil_image.convert("RGB").resize((32, 32), Image.BICUBIC)
    x = get_transforms(train=False)(pil_image).unsqueeze(0).to(device)
    cam_engine = GradCAM(model, get_target_layer(model))
    try:
        cam, probs, idx = cam_engine(x, class_idx)
    finally:
        cam_engine.remove()
    overlay = overlay_cam(np.asarray(pil_image), cam[0])
    return {c: float(p) for c, p in zip(CLASSES, probs[0])}, overlay


def main() -> None:
    from torchvision import datasets

    p = argparse.ArgumentParser(description="Generate a Grad-CAM grid from CIFAR-10 test images")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--num-images", type=int, default=12)
    p.add_argument("--out", default="outputs/gradcam_grid.png")
    args = p.parse_args()

    device = get_device()
    model, _ = load_checkpoint(args.checkpoint, device)
    test_set = datasets.CIFAR10(args.data_dir, train=False, download=True,
                                transform=get_transforms(False))
    g = torch.Generator().manual_seed(0)
    picks = torch.randperm(len(test_set), generator=g)[: args.num_images].tolist()

    engine = GradCAM(model, get_target_layer(model))
    cols = 4
    rows = int(np.ceil(args.num_images / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4.2, rows * 2.2))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")

    for n, i in enumerate(picks):
        x, y = test_set[i]
        cam, probs, pred = engine(x.unsqueeze(0).to(device))
        img = denormalize(x)
        r, c = divmod(n, cols)
        axes[r, c * 2].imshow(Image.fromarray(img).resize((128, 128), Image.BICUBIC))
        axes[r, c * 2].set_title(f"true: {CLASSES[y]}", fontsize=8)
        axes[r, c * 2 + 1].imshow(overlay_cam(img, cam[0], size=128))
        ok = pred.item() == y
        axes[r, c * 2 + 1].set_title(f"pred: {CLASSES[pred.item()]} ({probs[0, pred.item()]:.0%})",
                                     fontsize=8, color="green" if ok else "red")
    engine.remove()
    fig.suptitle("Grad-CAM: where does the network look?", fontsize=12)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
