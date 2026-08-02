"""Unit tests for Pan2020PartialConv2d: mask-mean normalization, mask update,
degeneracy to ordinary conv, and local-vs-global differentiation."""

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

from model.interpolation.pan2020_pconv_unet import (  # noqa: E402
    Pan2020PartialConv2d,
    _same_pad_size,
)


class TestSamePadSize:
    def test_odd_kernel_stride_1(self) -> None:
        left, right = _same_pad_size(128, 7, 1)
        assert left == 3
        assert right == 3

    def test_stride_2_even_input(self) -> None:
        left, right = _same_pad_size(128, 3, 2)
        assert left == 0
        assert right == 1

    def test_stride_2_odd_input(self) -> None:
        left, right = _same_pad_size(127, 3, 2)
        assert left == 1
        assert right == 1

    def test_dilation(self) -> None:
        left, right = _same_pad_size(64, 3, 1, dilation=2)
        assert left == 2
        assert right == 2

    def test_output_size_formula(self) -> None:
        for size, k, s in [(128, 7, 2), (64, 5, 2), (32, 3, 1)]:
            p0, p1 = _same_pad_size(size, k, s)
            padded = size + p0 + p1
            expected_out = (size + s - 1) // s
            actual_out = (padded - k) // s + 1
            assert actual_out == expected_out


class TestGlobalMaskMean:
    def test_all_ones_mask_degenerates_to_ordinary_conv(self) -> None:
        pconv = Pan2020PartialConv2d(2, 3, kernel_size=3, stride=1, bias=False,
                                     normalization_mode="author_global_mask_mean")
        conv = torch.nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1, bias=False)
        conv.weight.data.copy_(pconv.weight.data)
        x = torch.rand(2, 2, 8, 8)
        mask = torch.ones(2, 2, 8, 8)
        out_pconv, _ = pconv(x, mask)
        out_conv = conv(x)
        assert torch.allclose(out_pconv, out_conv, atol=2e-6)

    def test_half_mask_doubles_output(self) -> None:
        pconv = Pan2020PartialConv2d(1, 1, kernel_size=1, stride=1, bias=False,
                                     normalization_mode="author_global_mask_mean")
        pconv.weight.data.fill_(1.0)
        x = torch.ones(1, 1, 4, 4)
        mask = torch.cat([
            torch.ones(1, 1, 2, 4),
            torch.zeros(1, 1, 2, 4),
        ], dim=2)
        out, _ = pconv(x, mask)
        expected = 1.0 / 0.5
        assert torch.allclose(out, torch.full_like(out, expected), atol=1e-5)

    def test_fixed_weights_manual_formula(self) -> None:
        pconv = Pan2020PartialConv2d(1, 1, kernel_size=3, stride=1, bias=False,
                                     normalization_mode="author_global_mask_mean")
        pconv.weight.data.fill_(1.0 / 9.0)
        x = torch.ones(1, 1, 5, 5)
        mask = torch.ones(1, 1, 5, 5)
        mask[:, :, 0, :] = 0.0
        out, mask_out = pconv(x, mask)
        assert mask_out[:, :, 0, :].eq(0).all()
        interior = out[:, :, 2, 2]
        assert torch.allclose(interior, torch.tensor(1.25), atol=1e-5)

    def test_zero_valid_fraction_raises(self) -> None:
        pconv = Pan2020PartialConv2d(1, 1, kernel_size=3, stride=1,
                                     normalization_mode="author_global_mask_mean",
                                     zero_valid_policy="error")
        x = torch.rand(1, 1, 4, 4)
        mask = torch.zeros(1, 1, 4, 4)
        with pytest.raises(RuntimeError):
            pconv(x, mask)

    def test_zero_valid_fraction_clamp(self) -> None:
        pconv = Pan2020PartialConv2d(1, 1, kernel_size=3, stride=1,
                                     normalization_mode="author_global_mask_mean",
                                     zero_valid_policy="clamp")
        x = torch.rand(1, 1, 4, 4)
        mask = torch.zeros(1, 1, 4, 4)
        out, _ = pconv(x, mask)
        assert out.shape == (1, 1, 4, 4)


class TestMaskUpdate:
    def test_stride_1_kernel_3(self) -> None:
        pconv = Pan2020PartialConv2d(1, 1, kernel_size=3, stride=1)
        x = torch.rand(1, 1, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 3:6, 3:6] = 1.0
        _, mask_out = pconv(x, mask)
        assert mask_out.sum() > mask.sum()

    def test_stride_2_mask_downsampling(self) -> None:
        pconv = Pan2020PartialConv2d(1, 1, kernel_size=7, stride=2)
        x = torch.rand(1, 1, 16, 16)
        mask = torch.ones(1, 1, 16, 16)
        _, mask_out = pconv(x, mask)
        assert mask_out.shape == (1, 1, 8, 8)

    def test_mask_output_binary(self) -> None:
        pconv = Pan2020PartialConv2d(1, 1, kernel_size=3, stride=1)
        x = torch.rand(1, 1, 16, 16)
        mask = torch.rand(1, 1, 16, 16).gt(0.5).float()
        _, mask_out = pconv(x, mask)
        unique = mask_out.unique()
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_mask_kernel_not_trainable(self) -> None:
        pconv = Pan2020PartialConv2d(1, 1, kernel_size=3)
        assert not any(p.requires_grad for n, p in pconv.named_parameters()
                       if "mask" in n)
        assert "mask_kernel" in dict(pconv.named_buffers())


class TestLocalValidRatio:
    def test_all_ones_mask_degenerates(self) -> None:
        pconv = Pan2020PartialConv2d(2, 3, kernel_size=3, stride=1, bias=False,
                                     normalization_mode="standard_local_valid_ratio")
        conv = torch.nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1, bias=False)
        conv.weight.data.copy_(pconv.weight.data)
        x = torch.rand(2, 2, 8, 8)
        mask = torch.ones(2, 2, 8, 8)
        out_pconv, _ = pconv(x, mask)
        out_conv = conv(x)
        assert torch.allclose(out_pconv, out_conv, atol=2e-6)

    def test_different_from_global_mode(self) -> None:
        pconv_g = Pan2020PartialConv2d(1, 1, kernel_size=3, stride=1, bias=False,
                                        normalization_mode="author_global_mask_mean")
        pconv_l = Pan2020PartialConv2d(1, 1, kernel_size=3, stride=1, bias=False,
                                        normalization_mode="standard_local_valid_ratio")
        pconv_l.weight.data.copy_(pconv_g.weight.data)
        x = torch.rand(1, 1, 8, 8)
        mask = torch.ones(1, 1, 8, 8)
        mask[:, :, :4, :] = 0.0
        out_g, _ = pconv_g(x, mask)
        out_l, _ = pconv_l(x, mask)
        assert not torch.allclose(out_g, out_l, atol=1e-4)


class TestShapes:
    @pytest.mark.parametrize("h,w", [(128, 128), (127, 63), (64, 64), (32, 32)])
    def test_output_size_stride_2(self, h: int, w: int) -> None:
        pconv = Pan2020PartialConv2d(1, 1, kernel_size=7, stride=2)
        x = torch.rand(1, 1, h, w)
        mask = torch.ones(1, 1, h, w)
        out, mout = pconv(x, mask)
        expected_h = (h + 1) // 2
        expected_w = (w + 1) // 2
        assert out.shape[2] == expected_h
        assert out.shape[3] == expected_w
        assert mout.shape == out.shape

    @pytest.mark.parametrize("k", [3, 5, 7])
    def test_output_size_stride_1(self, k: int) -> None:
        pconv = Pan2020PartialConv2d(1, 1, kernel_size=k, stride=1)
        x = torch.rand(1, 1, 32, 32)
        mask = torch.ones(1, 1, 32, 32)
        out, _ = pconv(x, mask)
        assert out.shape == (1, 1, 32, 32)

    def test_multichannel_input(self) -> None:
        pconv = Pan2020PartialConv2d(4, 8, kernel_size=3, stride=1)
        x = torch.rand(2, 4, 16, 16)
        mask = torch.ones(2, 4, 16, 16)
        out, mout = pconv(x, mask)
        assert out.shape == (2, 8, 16, 16)
        assert mout.shape == (2, 8, 16, 16)

    def test_channel_mismatch_raises(self) -> None:
        pconv = Pan2020PartialConv2d(2, 4, kernel_size=3)
        x = torch.rand(1, 3, 8, 8)
        mask = torch.ones(1, 3, 8, 8)
        with pytest.raises(ValueError):
            pconv(x, mask)
