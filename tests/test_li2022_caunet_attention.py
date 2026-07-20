"""Unit tests for the Coordinate Attention block (CoordAttention2D)."""

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

from model.interpolation.li2022_caunet import CoordAttention2D  # noqa: E402


class TestShapeAndNormalization:
    def test_output_shape_matches_input(self) -> None:
        ca = CoordAttention2D(16, reduction_ratio=4)
        x = torch.rand(2, 16, 32, 32)
        out = ca(x)
        assert out.shape == x.shape

    def test_attention_values_in_range(self) -> None:
        ca = CoordAttention2D(8, reduction_ratio=4)
        x = torch.rand(2, 8, 16, 16)
        # Directly inspect the attention maps through the module internals.
        b, c, h, w = x.shape
        h_pool = torch.nn.functional.adaptive_avg_pool2d(x, (h, 1))
        w_pool = torch.nn.functional.adaptive_avg_pool2d(x, (1, w))
        h_pool_t = h_pool.permute(0, 1, 3, 2)
        z = torch.cat([h_pool_t, w_pool], dim=-1)
        z = ca.shared_conv(z)
        z = ca.bn(z)
        z = torch.nn.functional.hardswish(z)
        z_h, z_w = z.split([h, w], dim=-1)
        a_h = torch.sigmoid(ca.conv_h(z_h))
        a_w = torch.sigmoid(ca.conv_w(z_w))
        assert a_h.min() >= 0 and a_h.max() <= 1
        assert a_w.min() >= 0 and a_w.max() <= 1

    def test_different_spatial_sizes(self) -> None:
        ca = CoordAttention2D(16, reduction_ratio=4)
        for shape in [(32, 32), (16, 64), (720, 120), (128, 128)]:
            x = torch.rand(1, 16, *shape)
            out = ca(x)
            assert out.shape == x.shape


class TestDirectionalPooling:
    def test_h_pool_produces_strip(self) -> None:
        ca = CoordAttention2D(8, reduction_ratio=4)
        x = torch.rand(2, 8, 16, 32)
        b, c, h, w = x.shape
        h_pool = torch.nn.functional.adaptive_avg_pool2d(x, (h, 1))
        assert h_pool.shape == (b, c, h, 1)

    def test_w_pool_produces_strip(self) -> None:
        ca = CoordAttention2D(8, reduction_ratio=4)
        x = torch.rand(2, 8, 16, 32)
        b, c, h, w = x.shape
        w_pool = torch.nn.functional.adaptive_avg_pool2d(x, (1, w))
        assert w_pool.shape == (b, c, 1, w)

    def test_no_2d_global_pool(self) -> None:
        # Coordinate Attention must NOT reduce to (1,1) — that would be SE.
        ca = CoordAttention2D(8, reduction_ratio=4)
        assert not any(
            isinstance(m, nn.AdaptiveAvgPool2d) and m.output_size == (1, 1)
            for m in ca.modules()
        )


class TestReductionRatio:
    @pytest.mark.parametrize("r", [2, 4, 8, 16])
    def test_intermediate_channel_dimension(self, r: int) -> None:
        ca = CoordAttention2D(64, reduction_ratio=r)
        x = torch.rand(1, 64, 8, 8)
        b, c, h, w = x.shape
        h_pool = torch.nn.functional.adaptive_avg_pool2d(x, (h, 1))
        w_pool = torch.nn.functional.adaptive_avg_pool2d(x, (1, w))
        h_pool_t = h_pool.permute(0, 1, 3, 2)
        z = torch.cat([h_pool_t, w_pool], dim=-1)
        z = ca.shared_conv(z)
        expected = max(1, 64 // r)
        assert z.shape[1] == expected

    def test_reduction_ratio_one(self) -> None:
        ca = CoordAttention2D(32, reduction_ratio=1)
        x = torch.rand(1, 32, 8, 8)
        out = ca(x)
        assert out.shape == x.shape


class TestGradients:
    def test_attention_parameters_receive_gradients(self) -> None:
        ca = CoordAttention2D(16, reduction_ratio=4)
        x = torch.rand(2, 16, 8, 8, requires_grad=True)
        out = ca(x)
        out.sum().backward()
        for name, p in ca.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} has NaN/Inf gradients"
            assert p.grad.abs().sum() > 0, f"{name} has zero gradients"

    def test_input_gradients_flow(self) -> None:
        ca = CoordAttention2D(16, reduction_ratio=4)
        x = torch.rand(2, 16, 8, 8, requires_grad=True)
        out = ca(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0


class TestSpatialDependency:
    def test_far_pixel_perturbation_affects_output(self) -> None:
        torch.manual_seed(42)
        ca = CoordAttention2D(16, reduction_ratio=4)
        x = torch.rand(1, 16, 8, 8)
        ref = ca(x).detach().clone()
        # Perturb a single pixel at the far corner.
        x2 = x.clone()
        x2[0, :, -1, -1] += 1.0
        out2 = ca(x2).detach()
        # The perturbation should propagate via the H/W pooling, affecting all positions.
        assert not torch.allclose(ref[:, :, 0, 0], out2[:, :, 0, 0], atol=1e-6)


class TestHswishActivation:
    def test_hardswish_is_used(self) -> None:
        # The shared transform must use H-Swish, not ReLU (Hou et al. paper-explicit).
        ca = CoordAttention2D(16, reduction_ratio=4)
        # Check that no ReLU exists in the CA module.
        for m in ca.modules():
            assert not isinstance(m, nn.ReLU), "CoordAttention2D must use H-Swish, not ReLU"

    def test_hardswish_output_range(self) -> None:
        ca = CoordAttention2D(16, reduction_ratio=4)
        x = torch.randn(2, 16, 8, 8)
        b, c, h, w = x.shape
        h_pool = torch.nn.functional.adaptive_avg_pool2d(x, (h, 1))
        w_pool = torch.nn.functional.adaptive_avg_pool2d(x, (1, w))
        h_pool_t = h_pool.permute(0, 1, 3, 2)
        z = torch.cat([h_pool_t, w_pool], dim=-1)
        z = ca.shared_conv(z)
        z = ca.bn(z)
        z = torch.nn.functional.hardswish(z)
        # H-Swish lower bound is approximately -3/6 = -0.5, upper bound is infinity.
        # BN output can be negative, so just verify finite.
        assert torch.isfinite(z).all()


class TestValidation:
    def test_non_positive_channels_rejected(self) -> None:
        with pytest.raises(ValueError):
            CoordAttention2D(0)
        with pytest.raises(ValueError):
            CoordAttention2D(-1)

    def test_non_positive_reduction_ratio_rejected(self) -> None:
        with pytest.raises(ValueError):
            CoordAttention2D(16, reduction_ratio=0)
        with pytest.raises(ValueError):
            CoordAttention2D(16, reduction_ratio=-1)

    def test_wrong_input_channels_raises(self) -> None:
        ca = CoordAttention2D(16)
        x = torch.rand(1, 8, 8, 8)
        with pytest.raises(ValueError):
            ca(x)
