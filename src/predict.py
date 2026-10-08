"""Classify your own image (and optionally explain it with Grad-CAM).

    python -m src.predict path/to/image.jpg --checkpoint checkpoints/best.pt --gradcam
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from .gradcam import explain_pil_image
from .utils import get_device, load_checkpoint


def main():
    p = argparse.ArgumentParser(description="Predict the class of one image")
    p.add_argument("image")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--gradcam", action="store_true", help="save a Grad-CAM overlay")
    p.add_argument("--out", default="outputs/prediction_gradcam.png")
    args = p.parse_args()

    device = get_device()
    model, _ = load_checkpoint(args.checkpoint, device)
    probs, overlay = explain_pil_image(model, Image.open(args.image), device)

    print(f"\nPrediction for {args.image}:")
    for name, pr in sorted(probs.items(), key=lambda kv: -kv[1])[:3]:
        print(f"  {name:<11} {pr:6.1%}")

    if args.gradcam:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(args.out)
        print(f"\nGrad-CAM overlay saved to {args.out}")


if __name__ == "__main__":
    main()
