"""Extension tests for Chai2020UNet non-paper options.

All tests use a lightweight model (base_channels=8) for speed.
These verify that optional norm, activation, upsample, kernel_size,
dropout, and alignment_pad_mode variants build and forward-pass without error.
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

from model.interpolation.chai2020_unet import Chai2020UNet  # noqa: E402


def _forward_ok(model: Chai2020UNet, shape=(1, 1, 64, 64)) -> bool:
    x = torch.randn(*shape)
    with torch.no_grad():
        y = model(x)
    return y.shape == x.shape


class TestNormVariants:
    @pytest.mark.parametrize("norm", ["batch", "instance", "group", "none"])
    def test_build_and_forward(self, norm: str) -> None:
        m = Chai2020UNet(base_channels=8, norm=norm)
        assert _forward_ok(m)

    def test_invalid_norm_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown norm"):
            Chai2020UNet(base_channels=8, norm="layer")


class TestActivationVariants:
    @pytest.mark.parametrize("act", ["relu", "leaky_relu", "elu", "gelu", "none"])
    def test_build_and_forward(self, act: str) -> None:
        m = Chai2020UNet(base_channels=8, activation=act)
        assert _forward_ok(m)

    def test_invalid_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown activation"):
            Chai2020UNet(base_channels=8, activation="swish")


class TestUpsampleVariants:
    @pytest.mark.parametrize("mode", ["nearest", "bilinear", "transpose"])
    def test_build_and_forward(self, mode: str) -> None:
        m = Chai2020UNet(base_channels=8, upsample=mode)
        assert _forward_ok(m)

    def test_invalid_upsample_raises(self) -> None:
        with pytest.raises(ValueError, match="upsample"):
            Chai2020UNet(base_channels=8, upsample="pixelshuffle")


class TestKernelSizeVariants:
    @pytest.mark.parametrize("ks", [3, 5, 7])
    def test_build_and_forward(self, ks: int) -> None:
        m = Chai2020UNet(base_channels=8, kernel_size=ks)
        # 7x7 kernel on 64x64 input: still 4 levels of 2x down, min size = 64/16 = 4.
        assert _forward_ok(m)


class TestDropout:
    def test_dropout_builds_and_runs_in_train(self) -> None:
        m = Chai2020UNet(base_channels=8, dropout=0.5)
        m.train()
        x = torch.randn(2, 1, 64, 64)
        y = m(x)
        assert y.shape == x.shape
        # In training mode with dropout > 0, two forward passes should differ.
        y2 = m(x)
        assert not torch.equal(y, y2)

    def test_dropout_noop_in_eval(self) -> None:
        m = Chai2020UNet(base_channels=8, dropout=0.5)
        m.eval()
        x = torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            y1 = m(x)
            y2 = m(x)
        torch.testing.assert_close(y1, y2)


class TestAlignmentPadModes:
    @pytest.mark.parametrize("mode", ["reflect", "replicate", "constant"])
    def test_build_and_forward(self, mode: str) -> None:
        m = Chai2020UNet(base_channels=8, alignment_pad_mode=mode)
        # Non-multiple-of-16 input to trigger the alignment path.
        assert _forward_ok(m, shape=(1, 1, 101, 103))

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="alignment_pad_mode"):
            Chai2020UNet(base_channels=8, alignment_pad_mode="circular")


class TestMultiChannel:
    def test_three_channel_in_out(self) -> None:
        m = Chai2020UNet(in_channels=3, out_channels=3, base_channels=8)
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 3, 64, 64)

    def test_different_in_out_channels(self) -> None:
        m = Chai2020UNet(in_channels=2, out_channels=1, base_channels=8)
        x = torch.randn(2, 2, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 1, 64, 64)
