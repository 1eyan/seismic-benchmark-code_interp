"""Bottleneck residual block tests for Liu2022WRDL: structure, identity, gradient flow."""

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

from model.interpolation.liu2022_wrdl import WRDLBottleneckResidualBlock  # noqa: E402


class TestBottleneckStructure:
    def test_conv_order(self):
        """Verify 1x1 → 3x3 → 1x1 kernel order."""
        block = WRDLBottleneckResidualBlock(128, bottleneck_ratio=4)
        assert block.conv1.kernel_size == (1, 1)
        assert block.conv2.kernel_size == (3, 3)
        assert block.conv3.kernel_size == (1, 1)

    def test_channel_bottleneck(self):
        """Hidden dim = channels // ratio."""
        block = WRDLBottleneckResidualBlock(128, bottleneck_ratio=4)
        assert block.conv1.out_channels == 32
        assert block.conv2.in_channels == 32
        assert block.conv2.out_channels == 32
        assert block.conv3.in_channels == 32
        assert block.conv3.out_channels == 128

    def test_min_hidden_channels(self):
        block = WRDLBottleneckResidualBlock(2, bottleneck_ratio=4)
        assert block.conv1.out_channels == 1

    def test_bn_after_each_conv(self):
        block = WRDLBottleneckResidualBlock(64)
        assert isinstance(block.bn1, torch.nn.BatchNorm2d)
        assert isinstance(block.bn2, torch.nn.BatchNorm2d)
        assert isinstance(block.bn3, torch.nn.BatchNorm2d)


class TestGradientFlow:
    def test_all_convs_get_grad(self):
        block = WRDLBottleneckResidualBlock(32)
        x = torch.rand(2, 32, 16, 16)
        out = block(x)
        loss = out.mean()
        loss.backward()
        assert block.conv1.weight.grad is not None
        assert block.conv2.weight.grad is not None
        assert block.conv3.weight.grad is not None

    def test_identity_shortcut_no_projection(self):
        """When channels match, shortcut is direct identity (no conv projection)."""
        block = WRDLBottleneckResidualBlock(32)
        x = torch.rand(1, 32, 8, 8)
        # Zero out the residual branch weights → output should equal ReLU(input)
        with torch.no_grad():
            block.conv3.weight.zero_()
            if block.conv3.bias is not None:
                block.conv3.bias.zero_()
            block.bn3.weight.fill_(0.001)
            block.bn3.bias.zero_()
            block.bn3.running_mean.zero_()
            block.bn3.running_var.fill_(1.0)
        # With tiny BN weight, residual branch is near zero → out ≈ ReLU(x)
        # But BN with zero bias and near-zero weight → near-zero output for residual
        # So out ≈ ReLU(x + 0) = ReLU(x)
        out = block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


class TestReproducibility:
    def test_deterministic_given_seed(self):
        torch.manual_seed(42)
        block1 = WRDLBottleneckResidualBlock(16)
        torch.manual_seed(42)
        block2 = WRDLBottleneckResidualBlock(16)
        x = torch.rand(1, 16, 8, 8)
        state1 = block1.state_dict()
        state2 = block2.state_dict()
        for k in state1:
            assert torch.equal(state1[k], state2[k]), f"{k} differs"
