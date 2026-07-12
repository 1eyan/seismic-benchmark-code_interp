"""Extension tests for Wang2019ResNet — non-profile options and invalid arguments.

All tests use lightweight configs (features=8, num_blocks=2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "model").is_dir() and (p / "utils").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("Cannot find repo root.")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.interpolation.wang2019_resnet import Wang2019ResNet  # noqa: E402


_LIGHT = dict(features=8, num_blocks=2)


# ---------------------------------------------------------------------------
# Output mode variants
# ---------------------------------------------------------------------------

class TestOutputModes:
    def test_direct(self) -> None:
        m = Wang2019ResNet(**_LIGHT, output_mode="direct")
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape

    def test_global_residual(self) -> None:
        m = Wang2019ResNet(**_LIGHT, output_mode="global_residual")
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape

    def test_global_residual_channel_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="in_channels == out_channels"):
            Wang2019ResNet(**_LIGHT, in_channels=1, out_channels=2,
                          output_mode="global_residual")


# ---------------------------------------------------------------------------
# Normalization variants
# ---------------------------------------------------------------------------

class TestNormVariants:
    @pytest.mark.parametrize("norm", ["batch", "instance", "group", "none"])
    def test_build_and_forward(self, norm: str) -> None:
        m = Wang2019ResNet(**_LIGHT, norm=norm)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape

    def test_invalid_norm_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown norm"):
            Wang2019ResNet(**_LIGHT, norm="layer")


# ---------------------------------------------------------------------------
# Activation variants
# ---------------------------------------------------------------------------

class TestActivationVariants:
    @pytest.mark.parametrize("act", ["relu", "leaky_relu", "prelu", "elu", "gelu", "none"])
    def test_build_and_forward(self, act: str) -> None:
        m = Wang2019ResNet(**_LIGHT, activation=act)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape

    def test_invalid_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown activation"):
            Wang2019ResNet(**_LIGHT, activation="swish")


# ---------------------------------------------------------------------------
# Kernel size variants
# ---------------------------------------------------------------------------

class TestKernelSizes:
    @pytest.mark.parametrize("ks", [3, 5, 7])
    def test_build_and_forward(self, ks: int) -> None:
        m = Wang2019ResNet(**_LIGHT, kernel_size=ks)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Dropout variants
# ---------------------------------------------------------------------------

class TestDropout:
    def test_dropout_in_train(self) -> None:
        m = Wang2019ResNet(**_LIGHT, dropout=0.5)
        m.train()
        x = torch.randn(1, 1, 64, 64)
        y1 = m(x)
        y2 = m(x)
        assert not torch.equal(y1, y2)

    def test_dropout_noop_in_eval(self) -> None:
        m = Wang2019ResNet(**_LIGHT, dropout=0.5)
        m.eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y1 = m(x)
            y2 = m(x)
        torch.testing.assert_close(y1, y2)


# ---------------------------------------------------------------------------
# Channel variants
# ---------------------------------------------------------------------------

class TestChannelVariants:
    def test_two_channel_input(self) -> None:
        m = Wang2019ResNet(**_LIGHT, in_channels=2)
        x = torch.randn(1, 2, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, 1, 64, 64)

    def test_two_channel_output(self) -> None:
        m = Wang2019ResNet(**_LIGHT, out_channels=2)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, 2, 64, 64)


# ---------------------------------------------------------------------------
# Feature count variants
# ---------------------------------------------------------------------------

class TestFeatureVariants:
    @pytest.mark.parametrize("features", [8, 16, 32, 128])
    def test_build_and_forward(self, features: int) -> None:
        m = Wang2019ResNet(features=features, num_blocks=2)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Block count variants
# ---------------------------------------------------------------------------

class TestBlockCountVariants:
    @pytest.mark.parametrize("num_blocks", [1, 2, 4, 5])
    def test_build_and_forward(self, num_blocks: int) -> None:
        m = Wang2019ResNet(features=8, num_blocks=num_blocks)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape

    def test_num_conv_layers_consistent(self) -> None:
        for nb in [1, 2, 3, 4]:
            m = Wang2019ResNet(features=8, num_blocks=nb)
            assert m.num_conv_layers == 2 * nb + 2


# ---------------------------------------------------------------------------
# Batch independence
# ---------------------------------------------------------------------------

class TestBatchIndependence:
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_output_shape(self, batch_size: int) -> None:
        m = Wang2019ResNet(**_LIGHT)
        x = torch.randn(batch_size, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (batch_size, 1, 64, 64)


# ---------------------------------------------------------------------------
# Invalid arguments
# ---------------------------------------------------------------------------

class TestInvalidArguments:
    def test_kernel_size_even(self) -> None:
        with pytest.raises(ValueError, match="kernel_size must be"):
            Wang2019ResNet(**_LIGHT, kernel_size=4)

    def test_kernel_size_zero(self) -> None:
        with pytest.raises(ValueError, match="kernel_size must be"):
            Wang2019ResNet(**_LIGHT, kernel_size=0)

    def test_kernel_size_negative(self) -> None:
        with pytest.raises(ValueError, match="kernel_size must be"):
            Wang2019ResNet(**_LIGHT, kernel_size=-3)

    def test_features_zero(self) -> None:
        with pytest.raises(ValueError, match="features must be"):
            Wang2019ResNet(features=0, num_blocks=2)

    def test_features_negative(self) -> None:
        with pytest.raises(ValueError, match="features must be"):
            Wang2019ResNet(features=-8, num_blocks=2)

    def test_num_blocks_zero(self) -> None:
        with pytest.raises(ValueError, match="num_blocks must be"):
            Wang2019ResNet(features=8, num_blocks=0)

    def test_num_blocks_negative(self) -> None:
        with pytest.raises(ValueError, match="num_blocks must be"):
            Wang2019ResNet(features=8, num_blocks=-1)

    def test_dropout_negative(self) -> None:
        with pytest.raises(ValueError, match="dropout must be"):
            Wang2019ResNet(**_LIGHT, dropout=-0.1)

    def test_dropout_at_least_one(self) -> None:
        with pytest.raises(ValueError, match="dropout must be"):
            Wang2019ResNet(**_LIGHT, dropout=1.0)

    def test_unsupported_output_mode(self) -> None:
        with pytest.raises(ValueError, match="output_mode"):
            Wang2019ResNet(**_LIGHT, output_mode="residual")

    def test_unsupported_norm(self) -> None:
        with pytest.raises(ValueError, match="Unknown norm"):
            Wang2019ResNet(**_LIGHT, norm="spectral")

    def test_unsupported_activation(self) -> None:
        with pytest.raises(ValueError, match="Unknown activation"):
            Wang2019ResNet(**_LIGHT, activation="swish")
