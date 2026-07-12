"""Profile tests for Wang2019ResNet — conservative reproduction configuration.

Architecture checks use the default 64-feature 3-block model.
Functional checks use a lightweight model (features=8, num_blocks=2).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "model").is_dir() and (p / "utils").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("Cannot find repo root.")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.interpolation.wang2019_resnet import (  # noqa: E402
    Wang2019ResNet,
    _ResidualBlock,
)
from model.registry import build_model  # noqa: E402


def _count_layers(model: nn.Module, layer_type: type) -> int:
    return sum(1 for m in model.modules() if isinstance(m, layer_type))


def _trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Architecture tests (default model)
# ---------------------------------------------------------------------------

class TestArchitecture:
    """Verify conservative reproduction profile using default constructor args."""

    @pytest.fixture(scope="class")
    def model(self) -> Wang2019ResNet:
        return Wang2019ResNet()

    def test_conv_count(self, model: Wang2019ResNet) -> None:
        assert _count_layers(model, nn.Conv2d) == 8

    def test_num_blocks(self, model: Wang2019ResNet) -> None:
        assert len(model.blocks) == 3

    def test_num_conv_layers_property(self, model: Wang2019ResNet) -> None:
        assert model.num_conv_layers == 8

    def test_features(self, model: Wang2019ResNet) -> None:
        assert model.stem_conv.out_channels == 64
        for blk in model.blocks:
            assert blk.conv1.out_channels == 64
            assert blk.conv2.out_channels == 64

    def test_no_pool_layers(self, model: Wang2019ResNet) -> None:
        assert _count_layers(model, nn.MaxPool2d) == 0
        assert _count_layers(model, nn.AvgPool2d) == 0

    def test_no_upsample_layers(self, model: Wang2019ResNet) -> None:
        assert _count_layers(model, nn.Upsample) == 0

    def test_no_norm_layers_by_default(self, model: Wang2019ResNet) -> None:
        assert _count_layers(model, nn.BatchNorm2d) == 0
        assert _count_layers(model, nn.InstanceNorm2d) == 0

    def test_no_dropout_layers_by_default(self, model: Wang2019ResNet) -> None:
        assert _count_layers(model, nn.Dropout2d) == 0
        assert _count_layers(model, nn.Dropout) == 0

    def test_parameter_count(self, model: Wang2019ResNet) -> None:
        params = _trainable_params(model)
        assert params == 222785, f"Expected 222785, got {params}"

    def test_output_mode_default(self, model: Wang2019ResNet) -> None:
        assert model.output_mode == "direct"

    def test_activation_default(self, model: Wang2019ResNet) -> None:
        assert _count_layers(model, nn.ReLU) >= 1


# ---------------------------------------------------------------------------
# Residual block tests
# ---------------------------------------------------------------------------

class TestResidualBlock:
    """Residual block identity shortcut behaviour."""

    def test_output_shape(self) -> None:
        blk = _ResidualBlock(channels=64)
        x = torch.randn(2, 64, 32, 32)
        out = blk(x)
        assert out.shape == x.shape

    def test_not_identity(self) -> None:
        """Block output should not equal input at random init."""
        blk = _ResidualBlock(channels=64)
        x = torch.randn(1, 64, 16, 16)
        out = blk(x)
        assert not torch.allclose(out, x)

    def test_norm_removes_bias(self) -> None:
        blk_with_norm = _ResidualBlock(channels=64, norm="batch")
        assert blk_with_norm.conv1.bias is None
        assert blk_with_norm.conv2.bias is None

    def test_bias_present_without_norm(self) -> None:
        blk = _ResidualBlock(channels=64, norm="none")
        assert blk.conv1.bias is not None
        assert blk.conv2.bias is not None

    def test_dropout_layer_added(self) -> None:
        blk = _ResidualBlock(channels=64, dropout=0.3)
        assert blk.dropout is not None
        assert isinstance(blk.dropout, nn.Dropout2d)


# ---------------------------------------------------------------------------
# Forward / Backward (lightweight)
# ---------------------------------------------------------------------------

class TestForwardBackward:
    """Gradient flow and numerical stability."""

    @pytest.fixture(scope="class")
    def model(self) -> Wang2019ResNet:
        return Wang2019ResNet(features=8, num_blocks=2)

    def test_forward_backward(self, model: Wang2019ResNet) -> None:
        x = torch.randn(2, 1, 64, 64)
        y = model(x)
        loss = y.square().mean()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_input_not_modified(self, model: Wang2019ResNet) -> None:
        x = torch.randn(1, 1, 64, 64)
        x_clone = x.clone()
        with torch.no_grad():
            model(x)
        torch.testing.assert_close(x, x_clone)


# ---------------------------------------------------------------------------
# Shape tests (lightweight)
# ---------------------------------------------------------------------------

class TestShapes:
    """Output shape must match input shape for various dimensions."""

    @pytest.fixture(scope="class")
    def model(self) -> Wang2019ResNet:
        return Wang2019ResNet(features=8, num_blocks=2)

    @pytest.mark.parametrize("shape", [
        (1, 1, 64, 64),
        (2, 1, 128, 128),
        (1, 1, 96, 96),
        (1, 1, 100, 100),
        (1, 1, 101, 103),
        (1, 3, 64, 64),
    ])
    def test_output_shape(self, model: Wang2019ResNet, shape: tuple) -> None:
        B, C, H, W = shape
        if C != 1:
            m = Wang2019ResNet(in_channels=C, out_channels=C, features=8, num_blocks=2)
        else:
            m = model
        x = torch.randn(B, C, H, W)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape

    def test_no_nan_inf(self, model: Wang2019ResNet) -> None:
        x = torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()


# ---------------------------------------------------------------------------
# Output mode tests
# ---------------------------------------------------------------------------

class TestOutputModes:
    def test_direct_output(self) -> None:
        m = Wang2019ResNet(features=8, num_blocks=2, output_mode="direct")
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape

    def test_global_residual(self) -> None:
        m = Wang2019ResNet(features=8, num_blocks=2, output_mode="global_residual")
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape

    def test_global_residual_is_not_identity(self) -> None:
        m = Wang2019ResNet(features=8, num_blocks=2, output_mode="global_residual")
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert not torch.allclose(y, x)


# ---------------------------------------------------------------------------
# Factory and serialization
# ---------------------------------------------------------------------------

class TestFactoryAndSerialization:
    """Registry factory and state-dict roundtrip."""

    def test_factory(self) -> None:
        model = build_model({
            "type": "wang2019_resnet",
            "params": {"features": 8, "num_blocks": 2},
        })
        assert isinstance(model, Wang2019ResNet)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_serialization(self) -> None:
        torch.manual_seed(42)
        model = Wang2019ResNet(features=8, num_blocks=2)
        x = torch.randn(1, 1, 64, 64)
        model.eval()
        with torch.no_grad():
            expected = model(x)

        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)

        model2 = Wang2019ResNet(features=8, num_blocks=2)
        model2.load_state_dict(torch.load(buf, weights_only=True))
        model2.eval()
        with torch.no_grad():
            actual = model2(x)
        torch.testing.assert_close(actual, expected)


# ---------------------------------------------------------------------------
# Trainer compatibility
# ---------------------------------------------------------------------------

class TestTrainerCompatibility:
    """Model must accept mask, positions, and extra kwargs."""

    def test_kwargs_ignored(self) -> None:
        model = Wang2019ResNet(features=8, num_blocks=2)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = model(x, mask=torch.zeros(1, 1, 64, 64), positions=None, extra=42)
        assert y.shape == x.shape
