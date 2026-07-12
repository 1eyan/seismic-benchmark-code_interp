"""Paper-faithful tests for Chai2020UNet.

Architecture checks use the full 87M model (no backward).
Functional checks use a lightweight model (base_channels=8).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Bootstrap repo root so that ``model`` is importable.
_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "model").is_dir() and (p / "utils").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("Cannot find repo root.")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.interpolation.chai2020_unet import Chai2020UNet  # noqa: E402
from model.registry import build_model  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_layers(model: nn.Module, layer_type: type) -> int:
    return sum(1 for m in model.modules() if isinstance(m, layer_type))


def _trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Architecture tests (full 87M model, no backward)
# ---------------------------------------------------------------------------

class TestPaperArchitecture:
    """Verify the paper-faithful model structure using default constructor args."""

    @pytest.fixture(scope="class")
    def model(self) -> Chai2020UNet:
        return Chai2020UNet()

    def test_conv_count(self, model: Chai2020UNet) -> None:
        assert _count_layers(model, nn.Conv2d) == 19

    def test_hidden_conv_kernel(self, model: Chai2020UNet) -> None:
        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        assert len(convs) == 19
        # 18 hidden convolutions: all 5x5.
        for conv in convs[:-1]:
            assert conv.kernel_size == (5, 5), f"Expected 5x5, got {conv.kernel_size}"
        # Output convolution: 1x1.
        assert convs[-1].kernel_size == (1, 1)

    def test_maxpool_count(self, model: Chai2020UNet) -> None:
        assert _count_layers(model, nn.MaxPool2d) == 4

    def test_no_norm_layers(self, model: Chai2020UNet) -> None:
        assert _count_layers(model, nn.BatchNorm2d) == 0
        assert _count_layers(model, nn.InstanceNorm2d) == 0
        assert _count_layers(model, nn.GroupNorm) == 0

    def test_no_dropout_layers(self, model: Chai2020UNet) -> None:
        assert _count_layers(model, nn.Dropout2d) == 0
        assert _count_layers(model, nn.Dropout) == 0

    def test_parameter_count(self, model: Chai2020UNet) -> None:
        assert _trainable_params(model) == 87_149_953

    def test_forward_shape_112(self, model: Chai2020UNet) -> None:
        x = torch.randn(1, 1, 112, 112)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_forward_shape_100(self, model: Chai2020UNet) -> None:
        """Non-16-multiple input (100x100) must still produce same-shape output."""
        x = torch.randn(1, 1, 100, 100)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_forward_no_nan_inf(self, model: Chai2020UNet) -> None:
        x = torch.randn(1, 1, 112, 112)
        with torch.no_grad():
            y = model(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()


# ---------------------------------------------------------------------------
# Functional tests (lightweight base_channels=8)
# ---------------------------------------------------------------------------

class TestFunctional:
    """Code-logic tests using a small model to keep CI fast."""

    @pytest.fixture(scope="class")
    def small_model(self) -> Chai2020UNet:
        return Chai2020UNet(base_channels=8, num_levels=4)

    @pytest.mark.parametrize("shape", [
        (1, 1, 112, 112),
        (2, 1, 128, 128),
        (1, 1, 100, 100),
        (1, 1, 101, 103),
        (1, 3, 64, 64),
    ])
    def test_forward_shape(self, small_model: Chai2020UNet, shape: tuple) -> None:
        B, C, H, W = shape
        # Rebuild with matching in/out channels for the 3-channel case.
        if C != 1:
            m = Chai2020UNet(in_channels=C, out_channels=C, base_channels=8)
        else:
            m = small_model
        x = torch.randn(B, C, H, W)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (B, C, H, W), f"Expected {(B, C, H, W)}, got {y.shape}"

    def test_forward_backward(self, small_model: Chai2020UNet) -> None:
        x = torch.randn(2, 1, 64, 64)
        y = small_model(x)
        loss = y.mean()
        loss.backward()
        for name, param in small_model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient."
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_residual_learning_zero_head(self) -> None:
        m = Chai2020UNet(
            in_channels=2, out_channels=2, base_channels=8, residual_learning=True
        )
        # Zero out output head so residual = 0.
        nn.init.zeros_(m.head.weight)
        nn.init.zeros_(m.head.bias)
        x = torch.randn(2, 2, 64, 64)
        with torch.no_grad():
            y = m(x)
        torch.testing.assert_close(y, x)

    def test_residual_learning_channel_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="in_channels == out_channels"):
            Chai2020UNet(in_channels=2, out_channels=1, residual_learning=True)

    def test_serialization(self) -> None:
        m = Chai2020UNet(base_channels=8)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            expected = m(x)
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        buf.seek(0)
        m2 = Chai2020UNet(base_channels=8)
        m2.load_state_dict(torch.load(buf, weights_only=True))
        with torch.no_grad():
            actual = m2(x)
        torch.testing.assert_close(actual, expected)

    def test_factory(self) -> None:
        m = build_model({"type": "chai2020_unet", "params": {"base_channels": 8}})
        assert isinstance(m, Chai2020UNet)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_batch_independence(self, batch_size: int) -> None:
        m = Chai2020UNet(base_channels=8)
        x = torch.randn(batch_size, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (batch_size, 1, 64, 64)

    def test_invalid_num_levels_raises(self) -> None:
        with pytest.raises(ValueError, match="num_levels"):
            Chai2020UNet(num_levels=1)

    def test_kwargs_ignored(self) -> None:
        """Extra kwargs passed by trainers must not break forward."""
        m = Chai2020UNet(base_channels=8)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x, mask=torch.zeros(1, 1, 64, 64), positions=None, extra=42)
        assert y.shape == x.shape

    def test_alignment_pad_modes(self) -> None:
        for mode in ("reflect", "replicate", "constant"):
            m = Chai2020UNet(base_channels=8, alignment_pad_mode=mode)
            x = torch.randn(1, 1, 101, 103)
            with torch.no_grad():
                y = m(x)
            assert y.shape == x.shape
