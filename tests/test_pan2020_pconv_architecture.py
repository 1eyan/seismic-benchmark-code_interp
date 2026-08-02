"""Architecture tests for Pan2020PConvUNet: registration, stage counts,
channel/kernel progression, activations, skip connections, shapes."""

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
from model.interpolation.pan2020_pconv_unet import (  # noqa: E402
    Pan2020PConvUNet,
)


class TestRegistration:
    def test_registered_name(self) -> None:
        model = build_model({"type": "pan2020_pconv_unet", "params": {}})
        assert isinstance(model, Pan2020PConvUNet)

    def test_default_depth_is_6(self) -> None:
        model = Pan2020PConvUNet()
        assert len(model.encoder_stages) == 6
        assert len(model.decoder_stages) == 6


class TestChannelProgression:
    def test_encoder_channels(self) -> None:
        model = Pan2020PConvUNet()
        expected = [32, 64, 128, 256, 512, 512]
        for i, stage in enumerate(model.encoder_stages):
            assert stage.pconv.out_channels == expected[i]

    def test_encoder_kernels(self) -> None:
        model = Pan2020PConvUNet()
        expected = [7, 5, 5, 3, 3, 3]
        for i, stage in enumerate(model.encoder_stages):
            assert stage.pconv.kernel_size == (expected[i], expected[i])

    def test_decoder_channels(self) -> None:
        model = Pan2020PConvUNet()
        expected = [512, 256, 128, 64, 32, 1]
        for i, stage in enumerate(model.decoder_stages):
            assert stage.pconv.out_channels == expected[i]

    def test_decoder_kernels_all_3(self) -> None:
        model = Pan2020PConvUNet()
        for stage in model.decoder_stages:
            assert stage.pconv.kernel_size == (3, 3)


class TestNormAndActivation:
    def test_encoder1_no_bn(self) -> None:
        model = Pan2020PConvUNet()
        assert isinstance(model.encoder_stages[0].bn, torch.nn.Identity)

    def test_encoders_2_to_6_have_bn(self) -> None:
        model = Pan2020PConvUNet()
        for i in range(1, 6):
            assert isinstance(model.encoder_stages[i].bn, torch.nn.BatchNorm2d)

    def test_encoder_activations_are_relu(self) -> None:
        model = Pan2020PConvUNet()
        for stage in model.encoder_stages:
            assert isinstance(stage.act, torch.nn.ReLU)

    def test_decoder_activations_are_leaky_relu(self) -> None:
        model = Pan2020PConvUNet()
        for stage in model.decoder_stages:
            assert isinstance(stage.act, torch.nn.LeakyReLU)
            assert stage.act.negative_slope == 0.2

    def test_decoder12_no_bn(self) -> None:
        model = Pan2020PConvUNet()
        assert isinstance(model.decoder_stages[5].bn, torch.nn.Identity)

    def test_decoders_7_to_11_have_bn(self) -> None:
        model = Pan2020PConvUNet()
        for i in range(5):
            assert isinstance(model.decoder_stages[i].bn, torch.nn.BatchNorm2d)

    def test_output_sigmoid(self) -> None:
        model = Pan2020PConvUNet(output_activation="sigmoid")
        assert isinstance(model.output_act, torch.nn.Sigmoid)

    def test_output_none(self) -> None:
        model = Pan2020PConvUNet(output_activation="none")
        assert isinstance(model.output_act, torch.nn.Identity)


class TestShapes:
    def test_128x128_input(self) -> None:
        model = Pan2020PConvUNet()
        x = torch.rand(2, 1, 128, 128)
        mask = torch.ones(2, 1, 128, 128)
        out = model(x, mask=mask)
        assert out.shape == (2, 1, 128, 128)

    def test_odd_size_input(self) -> None:
        model = Pan2020PConvUNet()
        x = torch.rand(2, 1, 127, 63)
        mask = torch.ones(2, 1, 127, 63)
        out = model(x, mask=mask)
        assert out.shape == (2, 1, 127, 63)

    def test_packed_input_mode(self) -> None:
        model = Pan2020PConvUNet(input_mode="packed")
        x = torch.rand(2, 1, 64, 64)
        mask = torch.ones(2, 1, 64, 64)
        packed = torch.cat([x, mask], dim=1)
        out = model(packed)
        assert out.shape == (2, 1, 64, 64)

    def test_separate_and_packed_equivalent(self) -> None:
        torch.manual_seed(42)
        m1 = Pan2020PConvUNet(input_mode="separate")
        torch.manual_seed(42)
        m2 = Pan2020PConvUNet(input_mode="packed")
        x = torch.rand(2, 1, 32, 32)
        mask = torch.ones(2, 1, 32, 32)
        packed = torch.cat([x, mask], dim=1)
        out1 = m1(x, mask=mask)
        out2 = m2(packed)
        assert torch.allclose(out1, out2, atol=2e-6)

    def test_mask_kwarg_absorbed(self) -> None:
        model = Pan2020PConvUNet()
        x = torch.rand(1, 1, 32, 32)
        mask = torch.ones(1, 1, 32, 32)
        out = model(x, mask=mask, extra_unused=42)
        assert out.shape == (1, 1, 32, 32)

    def test_packed_missing_second_channel_raises(self) -> None:
        model = Pan2020PConvUNet(input_mode="packed")
        x = torch.rand(1, 1, 32, 32)
        with pytest.raises(ValueError):
            model(x)


class TestOutputRange:
    def test_sigmoid_output_in_01(self) -> None:
        model = Pan2020PConvUNet(output_activation="sigmoid")
        model.eval()
        with torch.no_grad():
            x = torch.rand(2, 1, 64, 64)
            mask = torch.ones(2, 1, 64, 64)
            out = model(x, mask=mask)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_none_output_unbounded(self) -> None:
        model = Pan2020PConvUNet(output_activation="none")
        model.eval()
        with torch.no_grad():
            x = torch.rand(2, 1, 64, 64)
            mask = torch.ones(2, 1, 64, 64)
            out = model(x, mask=mask)
        assert not (out.min() >= 0.0 and out.max() <= 1.0)


class TestPartialMask:
    def test_masked_input_propagates(self) -> None:
        model = Pan2020PConvUNet(output_activation="none")
        x = torch.rand(2, 1, 64, 64)
        mask = torch.zeros(2, 1, 64, 64)
        mask[:, :, :, :32] = 1.0
        out = model(x, mask=mask)
        assert out.shape == (2, 1, 64, 64)
        assert torch.isfinite(out).all()

    def test_all_missing_raises(self) -> None:
        model = Pan2020PConvUNet(zero_valid_policy="error")
        x = torch.rand(1, 1, 32, 32)
        mask = torch.zeros(1, 1, 32, 32)
        with pytest.raises(RuntimeError):
            model(x, mask=mask)

    def test_all_missing_clamp_does_not_raise(self) -> None:
        model = Pan2020PConvUNet(zero_valid_policy="clamp")
        x = torch.rand(1, 1, 32, 32)
        mask = torch.zeros(1, 1, 32, 32)
        out = model(x, mask=mask)
        assert out.shape == (1, 1, 32, 32)
