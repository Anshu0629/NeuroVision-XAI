"""Interactive web demo (Gradio).

    python app.py --checkpoint checkpoints/best.pt
"""

import argparse

import gradio as gr

from src.gradcam import explain_pil_image
from src.utils import get_device, load_checkpoint


def build_demo(checkpoint: str):
    device = get_device()
    model, _ = load_checkpoint(checkpoint, device)

    def run(image):
        if image is None:
            return None, None
        return explain_pil_image(model, image, device)

    return gr.Interface(
        fn=run,
        inputs=gr.Image(type="pil", label="Upload an image"),
        outputs=[gr.Label(num_top_classes=3, label="Prediction"),
                 gr.Image(label="Grad-CAM: where the model looked")],
        title="NeuroVision-XAI",
        description=("CIFAR-10 classifier (ResNet-18) with Grad-CAM explanations. "
                     "Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. "
                     "Built by Anshuman Kapoor, B.Tech CSE (AIML), COER University, Roorkee."),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    a = ap.parse_args()
    build_demo(a.checkpoint).launch(share=a.share)
