"""Extension tests for Yuan2022BTN — non-profile options and invalid arguments.

All tests use a lightweight model (base_channels=8, num_levels=3).
"""

from __future__ import annotations

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

from model.interpolation.yuan2022_btn import (  # noqa: E402
    Yuan2022BTN,
    BlindTraceConv2d,
)


_LIGHT = dict(base_channels=8, num_levels=3)


# ---------------------------------------------------------------------------
# Kernel size variants
# ---------------------------------------------------------------------------

class TestKernelSizes:
    @pytest.mark.parametrize("ks", [3, 5, 7])
    def test_build_and_forward(self, ks: int) -> None:
        m = Yuan2022BTN(**_LIGHT, kernel_size=ks)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape

    @pytest.mark.parametrize("ks", [3, 5])
    def test_ks_passes_blind_trace_test(self, ks: int) -> None:
        torch.manual_seed(42)
        m = Yuan2022BTN(**_LIGHT, kernel_size=ks)
        m.eval()
        x1 = torch.randn(1, 1, 64, 64)
        x2 = x1.clone()
        target = 32
        x2[..., :, target] += 100.0
        with torch.no_grad():
            y1 = m(x1)
            y2 = m(x2)
        torch.testing.assert_close(y1[..., :, target], y2[..., :, target])


# ---------------------------------------------------------------------------
# Upsample variants
# ---------------------------------------------------------------------------

class TestUpsampleVariants:
    @pytest.mark.parametrize("mode", ["nearest", "bilinear", "transpose"])
    def test_build_and_forward(self, mode: str) -> None:
        m = Yuan2022BTN(**_LIGHT, upsample_mode=mode)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Branch weight sharing
# ---------------------------------------------------------------------------

class TestBranchSharing:
    def test_shared_passes_blind_trace(self) -> None:
        torch.manual_seed(42)
        m = Yuan2022BTN(**_LIGHT, share_branch_weights=True)
        m.eval()
        x1 = torch.randn(1, 1, 64, 64)
        x2 = x1.clone()
        x2[..., :, 32] += 100.0
        with torch.no_grad():
            y1 = m(x1)
            y2 = m(x2)
        torch.testing.assert_close(y1[..., :, 32], y2[..., :, 32])

    def test_unshared_passes_blind_trace(self) -> None:
        torch.manual_seed(42)
        m = Yuan2022BTN(**_LIGHT, share_branch_weights=False)
        m.eval()
        x1 = torch.randn(1, 1, 64, 64)
        x2 = x1.clone()
        x2[..., :, 32] += 100.0
        with torch.no_grad():
            y1 = m(x1)
            y2 = m(x2)
        torch.testing.assert_close(y1[..., :, 32], y2[..., :, 32])


# ---------------------------------------------------------------------------
# Decoder norm variants
# ---------------------------------------------------------------------------

class TestDecoderNormVariants:
    @pytest.mark.parametrize("norm", ["none", "batch", "instance", "group"])
    def test_build_and_forward(self, norm: str) -> None:
        m = Yuan2022BTN(**_LIGHT, decoder_norm=norm)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Channel variants
# ---------------------------------------------------------------------------

class TestChannelVariants:
    def test_two_channel_input(self) -> None:
        m = Yuan2022BTN(**_LIGHT, in_channels=2)
        x = torch.randn(1, 2, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, 1, 64, 64)

    def test_two_channel_output(self) -> None:
        m = Yuan2022BTN(**_LIGHT, out_channels=2)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, 2, 64, 64)


# ---------------------------------------------------------------------------
# Invalid arguments
# ---------------------------------------------------------------------------

class TestInvalidArguments:
    def test_base_channels_zero(self) -> None:
        with pytest.raises(ValueError, match="base_channels must be"):
            Yuan2022BTN(base_channels=0)

    def test_base_channels_negative(self) -> None:
        with pytest.raises(ValueError, match="base_channels must be"):
            Yuan2022BTN(base_channels=-4)

    def test_num_levels_zero(self) -> None:
        with pytest.raises(ValueError, match="num_levels must be"):
            Yuan2022BTN(num_levels=0)

    def test_kernel_size_even(self) -> None:
        with pytest.raises(ValueError, match="kernel_size must be"):
            Yuan2022BTN(kernel_size=4)

    def test_kernel_size_zero(self) -> None:
        with pytest.raises(ValueError, match="kernel_size must be"):
            Yuan2022BTN(kernel_size=0)

    def test_kernel_size_negative(self) -> None:
        with pytest.raises(ValueError, match="kernel_size must be"):
            Yuan2022BTN(kernel_size=-3)

    def test_negative_slope_negative(self) -> None:
        with pytest.raises(ValueError, match="negative_slope must be"):
            Yuan2022BTN(negative_slope=-0.1)

    def test_unsupported_encoder_norm(self) -> None:
        with pytest.raises(ValueError, match="Unknown encoder_norm"):
            Yuan2022BTN(encoder_norm="layer")

    def test_unsupported_decoder_norm(self) -> None:
        with pytest.raises(ValueError, match="Unknown decoder_norm"):
            Yuan2022BTN(decoder_norm="layer")

    def test_unsupported_activation(self) -> None:
        with pytest.raises(ValueError, match="Unknown activation"):
            Yuan2022BTN(activation="swish")

    def test_unsupported_upsample_mode(self) -> None:
        with pytest.raises(ValueError, match="Unknown upsample_mode"):
            Yuan2022BTN(upsample_mode="pixelshuffle")

    def test_blind_conv_kernel_even(self) -> None:
        with pytest.raises(ValueError, match="kernel_size must be"):
            BlindTraceConv2d(1, 1, kernel_size=2)

    def test_blind_conv_kernel_zero(self) -> None:
        with pytest.raises(ValueError, match="kernel_size must be"):
            BlindTraceConv2d(1, 1, kernel_size=0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_not_4d_raises(self) -> None:
        m = Yuan2022BTN(**_LIGHT)
        with pytest.raises(ValueError, match="Expected input shape"):
            m._validate_input(torch.randn(1, 64))

    def test_wrong_channels_raises(self) -> None:
        m = Yuan2022BTN(**_LIGHT, in_channels=1)
        with pytest.raises(ValueError, match="Expected 1 input channel"):
            m._validate_input(torch.randn(1, 3, 64, 64))

    def test_time_too_small_raises(self) -> None:
        m = Yuan2022BTN(**_LIGHT)
        with pytest.raises(ValueError, match="Time dimension must be"):
            m._validate_input(torch.randn(1, 1, 1, 64))

    def test_trace_too_few_raises(self) -> None:
        m = Yuan2022BTN(**_LIGHT)
        with pytest.raises(ValueError, match="Trace dimension must be"):
            m._validate_input(torch.randn(1, 1, 64, 2))


# ---------------------------------------------------------------------------
# Multi-input forward
# ---------------------------------------------------------------------------

class TestMultiInput:
    def test_batch_size_greater_than_one(self) -> None:
        m = Yuan2022BTN(**_LIGHT)
        x = torch.randn(4, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (4, 1, 64, 64)

    def test_non_square_spatial(self) -> None:
        m = Yuan2022BTN(**_LIGHT)
        x = torch.randn(1, 1, 128, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape
