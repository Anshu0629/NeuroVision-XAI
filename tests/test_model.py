import pytest
import torch

from src.gradcam import GradCAM
from src.model import build_model, count_parameters, get_target_layer


@pytest.mark.parametrize("name", ["resnet18", "simplecnn"])
def test_forward_shape(name):
    model = build_model(name).eval()
    out = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)


def test_resnet18_parameter_count():
    # CIFAR-style ResNet-18 has roughly 11.2 million parameters
    assert 11_000_000 < count_parameters(build_model("resnet18")) < 11_400_000


@pytest.mark.parametrize("name", ["resnet18", "simplecnn"])
def test_gradcam_output(name):
    model = build_model(name).eval()
    engine = GradCAM(model, get_target_layer(model))
    cam, probs, idx = engine(torch.randn(2, 3, 32, 32))
    engine.remove()
    assert cam.shape == (2, 32, 32)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6
    assert torch.allclose(probs.sum(1), torch.ones(2), atol=1e-4)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("does-not-exist")
