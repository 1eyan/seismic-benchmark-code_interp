"""Architecture tests for Liu2022WRDL: registration, stage counts, channels, shapes."""

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

from model.interpolation import build_model  # noqa: E402
from model.interpolation.liu2022_wrdl import (  # noqa: E402
    Liu2022WRDL,
    WRDLBottleneckResidualBlock,
    WaveletExpansionIWTBlock,
    WRDLEncoderStage,
    WRDLDecoderStage,
    FixedWaveletDWT2D,
    FixedWaveletIWT2D,
)


class TestRegistration:
    def test_registered_name(self):
        model = build_model({"type": "liu2022_wrdl", "params": {}})
        assert isinstance(model, Liu2022WRDL)

    def test_default_encoder_levels(self):
        model = Liu2022WRDL()
        # 5 encoder channels defined → 4 DWT stages (last level is bottleneck)
        assert len(model.encoder_stages) == 4

    def test_default_decoder_levels(self):
        model = Liu2022WRDL()
        assert len(model.expansion_blocks) == 4
        assert len(model.decoder_stages) == 4


class TestChannelProgression:
    def test_encoder_channels_match_constructor(self):
        model = Liu2022WRDL(encoder_channels=[16, 32, 64])
        # 3 levels → 2 DWT stages
        assert len(model.encoder_stages) == 2
        # First stage: in=16, out=32
        assert model.encoder_stages[0].conv1.in_channels == 16
        assert model.encoder_stages[1].conv1.in_channels == 32

    def test_expansion_output_divisible_by_4(self):
        model = Liu2022WRDL(encoder_channels=[32, 64, 128])
        # expansion from 128 → 4*64 channels
        assert model.expansion_blocks[0].expand.out_channels == 4 * 64
        assert model.expansion_blocks[0].expand.out_channels % 4 == 0


class TestBottleneck:
    def test_bottleneck_block_structure(self):
        block = WRDLBottleneckResidualBlock(64, bottleneck_ratio=4)
        assert block.conv1.kernel_size == (1, 1)
        assert block.conv1.out_channels == 16  # 64 // 4
        assert block.conv2.kernel_size == (3, 3)
        assert block.conv2.out_channels == 16
        assert block.conv3.kernel_size == (1, 1)
        assert block.conv3.out_channels == 64

    def test_bottleneck_identity_shape(self):
        block = WRDLBottleneckResidualBlock(32)
        x = torch.rand(2, 32, 16, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_bottleneck_count(self):
        model = Liu2022WRDL(bottleneck_blocks=3)
        assert len(model.bottleneck_residuals) == 3

    def test_zero_bottleneck_blocks(self):
        model = Liu2022WRDL(bottleneck_blocks=0)
        # Sequential of 0 blocks is identity
        x = torch.rand(1, 512, 8, 8)
        out = model.bottleneck_residuals(x)
        assert out.shape == x.shape


class TestShapes:
    def test_128x128_input(self):
        model = Liu2022WRDL()
        x = torch.rand(2, 1, 128, 128)
        out = model(x)
        assert out.shape == (2, 1, 128, 128)

    def test_64x64_input(self):
        model = Liu2022WRDL(encoder_channels=[32, 64, 128])
        x = torch.rand(1, 1, 64, 64)
        out = model(x)
        assert out.shape == (1, 1, 64, 64)

    def test_odd_size_input(self):
        model = Liu2022WRDL()
        x = torch.rand(2, 1, 127, 63)
        out = model(x)
        assert out.shape == (2, 1, 127, 63)

    def test_odd_size_forward_backward(self):
        model = Liu2022WRDL(encoder_channels=[8, 16])
        x = torch.rand(2, 1, 65, 65)
        out = model(x)
        loss = out.mean()
        loss.backward()
        for name, p in model.named_parameters():
            if "filters" not in name:
                assert p.grad is not None, f"{name} has no grad"


class TestOutputActivation:
    def test_sigmoid_output(self):
        model = Liu2022WRDL(output_activation="sigmoid", encoder_channels=[8, 16])
        model.eval()
        with torch.no_grad():
            out = model(torch.rand(1, 1, 32, 32))
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_none_output_unbounded(self):
        model = Liu2022WRDL(output_activation="none", encoder_channels=[8, 16])
        model.eval()
        with torch.no_grad():
            out = model(torch.rand(1, 1, 32, 32))
        assert not (out.min() >= 0.0 and out.max() <= 1.0)

    def test_tanh_output(self):
        model = Liu2022WRDL(output_activation="tanh", encoder_channels=[8, 16])
        model.eval()
        with torch.no_grad():
            out = model(torch.rand(1, 1, 32, 32))
        assert out.min() >= -1.0 and out.max() <= 1.0


class TestWaveletParameter:
    def test_invalid_wavelet_raises(self):
        with pytest.raises(ValueError):
            Liu2022WRDL(wavelet="db2")

    def test_invalid_subband_order_raises(self):
        with pytest.raises(ValueError):
            Liu2022WRDL(subband_order="hh_hl_lh_ll")

    def test_too_few_levels_raises(self):
        with pytest.raises(ValueError):
            Liu2022WRDL(encoder_channels=[32])


class TestComponents:
    def test_encoder_stage_output_shapes(self):
        dwt = FixedWaveletDWT2D()
        stage = WRDLEncoderStage(32, 64, dwt)
        x = torch.rand(2, 32, 64, 64)
        dwt_out, skip = stage(x)
        assert dwt_out.shape == (2, 256, 32, 32)  # 4*64 channels, half size
        assert skip.shape == (2, 64, 64, 64)

    def test_expansion_iwt_block(self):
        iwt = FixedWaveletIWT2D()
        block = WaveletExpansionIWTBlock(512, 256, iwt)
        x = torch.rand(2, 512, 16, 16)
        out = block(x)
        assert out.shape == (2, 256, 32, 32)

    def test_decoder_stage(self):
        stage = WRDLDecoderStage(512, 256)
        x = torch.rand(2, 256, 32, 32)
        skip = torch.rand(2, 256, 32, 32)
        out = stage(x, skip)
        assert out.shape == (2, 256, 32, 32)
