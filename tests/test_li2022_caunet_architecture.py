"""Architecture tests for the Li2022 CA-Unet model."""

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

from model.interpolation import build_model, MODEL_REGISTRY  # noqa: E402
from model.interpolation.li2022_caunet import (  # noqa: E402
    CoordAttention2D,
    Li2022CAUNet,
)


class TestRegistration:
    def test_registered(self) -> None:
        assert "li2022_caunet" in MODEL_REGISTRY

    def test_factory_build(self) -> None:
        model = build_model({"type": "li2022_caunet", "params": {}})
        assert isinstance(model, Li2022CAUNet)

    def test_factory_build_with_custom_params(self) -> None:
        model = build_model({
            "type": "li2022_caunet",
            "params": {"base_channels": 16, "depth": 2, "ca_reduction_ratio": 8},
        })
        assert model.depth == 2
        assert model.ca_reduction_ratio == 8


class TestTopology:
    def test_default_depth_is_three(self) -> None:
        model = Li2022CAUNet()
        assert model.depth == 3

    def test_seven_ca_blocks(self) -> None:
        model = Li2022CAUNet()
        ca_count = sum(1 for m in model.modules() if isinstance(m, CoordAttention2D))
        # 3 encoder + 1 bottleneck + 3 decoder = 7
        assert ca_count == 7

    def test_encoder_decoder_counts_match_depth(self) -> None:
        model = Li2022CAUNet(depth=4)
        assert len(model.encoders) == 4
        assert len(model.encoder_cas) == 4
        assert len(model.pools) == 4
        assert len(model.upconvs) == 4
        assert len(model.decoders) == 4
        assert len(model.decoder_cas) == 4

    def test_channel_progression(self) -> None:
        model = Li2022CAUNet(base_channels=32, depth=3)
        x = torch.rand(2, 1, 64, 64)
        h = x
        skips = []
        for enc, ca, pool in zip(model.encoders, model.encoder_cas, model.pools):
            h = enc(h)
            h = ca(h)
            skips.append(h)
            h = pool(h)
        assert skips[0].shape[1] == 32
        assert skips[1].shape[1] == 64
        assert skips[2].shape[1] == 128
        h = model.bottleneck(h)
        assert h.shape[1] == 256

    def test_maxpool2d_downsampling(self) -> None:
        model = Li2022CAUNet()
        for pool in model.pools:
            assert isinstance(pool, nn.MaxPool2d)

    def test_convtranpose2d_upsampling(self) -> None:
        model = Li2022CAUNet()
        for up in model.upconvs:
            assert isinstance(up, nn.ConvTranspose2d)

    def test_output_head_no_bn(self) -> None:
        model = Li2022CAUNet()
        assert isinstance(model.head, nn.Conv2d)

    def test_no_output_activation(self) -> None:
        model = Li2022CAUNet()
        x = torch.rand(1, 1, 32, 32)
        out = model(x)
        # Linear output — no activation, so negative values are allowed.
        assert out.shape == x.shape

    def test_no_forbidden_modules(self) -> None:
        model = Li2022CAUNet()
        forbidden = (nn.MultiheadAttention, nn.LSTM, nn.GRU, nn.RNN,
                     nn.TransformerEncoder, nn.TransformerDecoder)
        for m in model.modules():
            assert not isinstance(m, forbidden), f"Forbidden module {type(m)} found"

    def test_no_dilated_convolutions(self) -> None:
        model = Li2022CAUNet()
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                assert m.dilation == (1, 1), f"Dilated conv found: {m}"


class TestForwardShapes:
    def test_paper_synthetic_shape(self) -> None:
        model = Li2022CAUNet()
        x = torch.rand(2, 1, 128, 128)
        out = model(x)
        assert out.shape == x.shape

    def test_field_shape(self) -> None:
        model = Li2022CAUNet()
        x = torch.rand(1, 1, 720, 120)
        out = model(x)
        assert out.shape == x.shape

    def test_non_divisible_shape(self) -> None:
        model = Li2022CAUNet()
        x = torch.rand(1, 1, 101, 103)
        out = model(x)
        assert out.shape == x.shape

    def test_multichannel_io(self) -> None:
        model = Li2022CAUNet(in_channels=3, out_channels=2)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 2, 64, 64)

    def test_mask_accepted_and_ignored(self) -> None:
        model = Li2022CAUNet()
        x = torch.rand(1, 1, 32, 32)
        out_no_mask = model(x)
        out_with_mask = model(x, mask=torch.ones_like(x))
        torch.testing.assert_close(out_no_mask, out_with_mask)


class TestValidation:
    def test_depth_below_two_rejected(self) -> None:
        with pytest.raises(ValueError):
            Li2022CAUNet(depth=1)

    def test_even_kernel_rejected(self) -> None:
        with pytest.raises(ValueError):
            Li2022CAUNet(kernel_size=2)

    def test_wrong_input_channels_raises(self) -> None:
        model = Li2022CAUNet(in_channels=1)
        x = torch.rand(1, 3, 32, 32)
        with pytest.raises(ValueError):
            model(x)

    def test_wrong_ndim_raises(self) -> None:
        model = Li2022CAUNet()
        x = torch.rand(1, 1)
        with pytest.raises(ValueError):
            model(x)


class TestConsistency:
    def test_inference_mode_produces_same_output(self) -> None:
        torch.manual_seed(0)
        model = Li2022CAUNet(base_channels=8, depth=2)
        model.eval()
        x = torch.rand(1, 1, 32, 32)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_different_batch_sizes_yield_same_per_sample(self) -> None:
        torch.manual_seed(0)
        model = Li2022CAUNet(base_channels=8, depth=2)
        model.eval()
        x1 = torch.rand(1, 1, 32, 32)
        with torch.no_grad():
            out1 = model(x1)
            out4 = model(x1.repeat(4, 1, 1, 1))
        torch.testing.assert_close(out1[0], out4[0])
