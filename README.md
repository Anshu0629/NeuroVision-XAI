<div align="center">

# 🧠 NeuroVision-XAI

### Explainable Image Classification with a ResNet-18 and Grad-CAM

*A deep learning project that doesn't just classify images — it shows you **where it looked** to decide.*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)
![Explainability](https://img.shields.io/badge/XAI-Grad--CAM-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

## 📌 Overview

**NeuroVision-XAI** trains a convolutional neural network on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset (60,000 colour images, 10 classes) and then opens the "black box" using **Grad-CAM** heat-maps that highlight the image regions most responsible for each prediction.

| Stage | What it does |
|-------|--------------|
| **Data** | Reproducible train/val/test split, augmentation (random crop, flip, random erasing) |
| **Model** | ResNet-18 adapted for 32×32 images, written **from scratch**, plus a plain CNN baseline |
| **Training** | SGD + Nesterov, OneCycle LR schedule, label smoothing, mixed precision, gradient clipping |
| **Evaluation** | Accuracy, per-class precision/recall/F1, normalized confusion matrix, top confusions |
| **Explainability** | Grad-CAM implemented manually with PyTorch hooks |
| **Deployment** | CLI prediction script and an interactive Gradio web app |
| **Quality** | Unit tests with `pytest` |

## ✨ Key Features

- 🏗️ **ResNet-18 from scratch** — residual blocks, skip connections, Kaiming init, no pretrained weights
- 🔍 **Explainable AI** — Grad-CAM overlays for any image you upload
- ⚖️ **Baseline comparison** — `SimpleCNN` vs `ResNet18` to show the value of residual learning
- ⚡ **Fast to run** — mixed precision on GPU, `--fast-dev-run` smoke test in ~1 minute
- 🌐 **Web demo** — drag-and-drop interface via Gradio
- ✅ **Tested** — shape, parameter-count and Grad-CAM tests

## 🗂️ Project Structure

```
NeuroVision-XAI/
├── src/
│   ├── data.py        # CIFAR-10 loaders, augmentation, train/val split
│   ├── model.py       # SimpleCNN + ResNet-18 (from scratch)
│   ├── train.py       # training loop (OneCycle, AMP, label smoothing)
│   ├── evaluate.py    # test metrics + confusion matrix
│   ├── gradcam.py     # Grad-CAM implementation + visualization grid
│   ├── predict.py     # classify your own image from the terminal
│   └── utils.py       # seed, device, plotting helpers
├── tests/
│   └── test_model.py  # pytest unit tests
├── app.py             # Gradio web demo
├── requirements.txt
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### 1. Clone & install

```bash
git clone https://github.com/Anshu0629/NeuroVision-XAI.git
cd NeuroVision-XAI

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Quick smoke test (about 1 minute, even on CPU)

```bash
python -m src.train --fast-dev-run
pytest -q
```

### 3. Train

```bash
python -m src.train --model resnet18 --epochs 30
python -m src.train --model simplecnn --epochs 30 --out-dir checkpoints/simplecnn   # baseline
```

> 💡 **No GPU?** Use a free GPU in [Google Colab](https://colab.research.google.com): *Runtime → Change runtime type → T4 GPU*, then run `!git clone ...`, `!pip install -r requirements.txt` and the commands above.

### 4. Evaluate

```bash
python -m src.evaluate --checkpoint checkpoints/best.pt
```

### 5. Explain with Grad-CAM

```bash
python -m src.gradcam --checkpoint checkpoints/best.pt --num-images 12
python -m src.predict my_photo.jpg --gradcam
```

### 6. Launch the web app

```bash
python app.py --checkpoint checkpoints/best.pt
```

## 🧪 Methodology

**Architecture.** The standard ImageNet ResNet-18 downsamples too aggressively for 32×32 inputs. Following common CIFAR practice, the stem is a single 3×3 stride-1 convolution with no max-pool, followed by four stages of two `BasicBlock`s (64 → 128 → 256 → 512 channels), global average pooling and a linear classifier (~11.2 M parameters).

| Setting | Value |
|---------|-------|
| Optimizer | SGD, momentum 0.9, Nesterov, weight decay 5e-4 |
| LR schedule | OneCycle, peak 0.1 |
| Loss | Cross-entropy with label smoothing 0.1 |
| Augmentation | RandomCrop(32, pad 4), HorizontalFlip, RandomErasing |
| Regularization | Gradient clipping (5.0) |
| Split | 45,000 train / 5,000 validation / 10,000 test (seed 42) |

**Grad-CAM.** For a target class *c*, the gradients of its score are global-average-pooled over the last residual block's feature maps to get channel weights α<sub>k</sub>; the heat-map is `ReLU(Σ α_k · A_k)`, upsampled to the input size and overlaid on the image.

## 📊 Results

> Fill this in with **your own** numbers after training (`outputs/test_metrics.json`).

| Model | Parameters | Test accuracy |
|-------|-----------:|--------------:|
| SimpleCNN (baseline) | ~1.1 M | _your result_ |
| **ResNet-18 (ours)** | ~11.2 M | _your result_ |

## 🔭 Future Work

- [ ] Compare with transfer learning (pretrained ResNet / EfficientNet)
- [ ] Add MixUp / CutMix augmentation
- [ ] Extend explainability with Score-CAM or Integrated Gradients
- [ ] Export to ONNX and deploy on Hugging Face Spaces
- [ ] Try CIFAR-100 or a custom dataset

## 📚 References

1. K. He, X. Zhang, S. Ren, J. Sun — *Deep Residual Learning for Image Recognition*, CVPR 2016.
2. R. R. Selvaraju et al. — *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
3. A. Krizhevsky — *Learning Multiple Layers of Features from Tiny Images* (CIFAR-10), 2009.
4. L. N. Smith, N. Topin — *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates*, 2018.

## 👨‍🎓 Author

**Anshuman Kapoor**
B.Tech Computer Science & Engineering (AI & ML) — 1st Year, 1st Semester
Roll No. 13 · College of Smart Computing, **COER University, Roorkee**

GitHub: [@Anshu0629](https://github.com/Anshu0629)

If you found this project useful, please give it a ⭐!

## 📄 License

Released under the [MIT License](LICENSE).
