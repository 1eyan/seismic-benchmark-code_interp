"""Park2022 CFunet architecture tests: registration, channel layout, Eq. 2 mask overwrite, stride-2 final conv."""

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

from model.interpolation.park2022_cfunet import (  # noqa: E402
    FourierZeroPaddingUpsample2D,
    Park2022CFUNet,
)
from model.registry import build_model  # noqa: E402


def _make_model(**overrides):
    params = dict(
        in_channels=1,
        out_channels=1,
        base_channels=22,
        num_levels=4,
        upsample_mode="fourier_zero_padding",
        upsampler_scale_factor=2,
        use_fftshift=True,
        fft_norm="backward",
        amplitude_correction="scale",
    )
    params.update(overrides)
    return Park2022CFUNet(**params)


class TestRegistration:
    def test_build_from_registry(self):
        model = build_model(
            {
                "type": "park2022_cfunet",
                "params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "base_channels": 8,
                    "num_levels": 3,
                },
            }
        )
        assert isinstance(model, Park2022CFUNet)

    def test_invalid_upsample_mode(self):
        with pytest.raises(ValueError):
            _make_model(upsample_mode="cubic")


class TestChannelLayout:
    def test_paper_channel_progression(self):
        """Coarse and refine U-Nets: 22 -> 44 -> 88 -> 176, bottleneck 352."""
        model = _make_model()
        for net in (model.coarse_net, model.refine_net):
            encoder_outs = [enc.block[0].out_channels for enc in net.encoders]
            assert encoder_outs == [22, 44, 88, 176]
            assert net.bottleneck.block[0].out_channels == 352

    def test_final_stride2_conv(self):
        """Eq. 5: the refine network ends with a 3x3 stride-2 conv."""
        model = _make_model()
        conv = model.refine_down
        assert conv.kernel_size == (3, 3)
        assert conv.stride == (2, 2)
        assert conv.padding == (1, 1)
        assert conv.out_channels == 1

    def test_fourier_upsampler_configured(self):
        model = _make_model()
        assert isinstance(model.upsampler, FourierZeroPaddingUpsample2D)
        assert model.upsampler.scale_factor == 2


class TestForwardShapes:
    @pytest.mark.parametrize("mode", ["fourier_zero_padding", "bilinear", "nearest"])
    def test_output_shape(self, mode):
        model = _make_model(upsample_mode=mode)
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert tuple(out.shape) == (2, 1, 64, 64)

    def test_intermediates_shapes(self):
        model = _make_model()
        x = torch.randn(2, 1, 64, 64)
        inter = model(x, return_intermediates=True)
        assert set(inter.keys()) == {"coarse", "upsampled", "final"}
        assert tuple(inter["coarse"].shape) == (2, 1, 64, 64)
        assert tuple(inter["upsampled"].shape) == (2, 1, 128, 128)
        assert tuple(inter["final"].shape) == (2, 1, 64, 64)

    def test_intermediates_cached_after_plain_forward(self):
        model = _make_model()
        x = torch.randn(1, 1, 64, 64)
        model(x)
        assert set(model._intermediates.keys()) == {"coarse", "upsampled", "final"}


class TestEq2MaskOverwrite:
    def test_observed_positions_copy_input(self):
        """Eq. 2: at observed positions the coarse output equals the input."""
        model = _make_model()
        x = torch.randn(1, 1, 64, 64)
        mask = torch.zeros_like(x)
        mask[0, 0, :, :32] = 1.0  # left half observed
        inter = model(x, mask=mask, return_intermediates=True)
        oc = inter["coarse"]
        torch.testing.assert_close(oc[0, 0, :, :32], x[0, 0, :, :32], rtol=1e-6, atol=1e-7)
        # Missing half comes from the coarse network (not the input).
        assert not torch.allclose(oc[0, 0, :, 32:], x[0, 0, :, 32:])

    def test_mask_inferred_from_zeros(self):
        """Without an explicit mask, zero traces are treated as missing."""
        model = _make_model()
        x = torch.randn(1, 1, 64, 64)
        x[0, 0, :, 32:] = 0.0  # right half missing
        inter = model(x, return_intermediates=True)
        oc = inter["coarse"]
        torch.testing.assert_close(oc[0, 0, :, :32], x[0, 0, :, :32], rtol=1e-6, atol=1e-7)

    def test_mask_shape_mismatch_raises(self):
        model = _make_model()
        x = torch.randn(1, 1, 64, 64)
        with pytest.raises(ValueError):
            model(x, mask=torch.zeros(1, 2, 64, 64))

    def test_mask_expands_to_input_channels(self):
        model = _make_model(in_channels=3)
        x = torch.randn(1, 3, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        mask[0, 0, :, :32] = 1.0
        out = model(x, mask=mask)
        assert tuple(out.shape) == (1, 3, 64, 64)


class TestTrainerCompatibility:
    def test_forward_signature_has_mask(self):
        import inspect

        sig = inspect.signature(Park2022CFUNet.forward)
        assert "mask" in sig.parameters

    def test_kwargs_absorbed(self):
        model = _make_model()
        x = torch.randn(1, 1, 32, 32)
        out = model(x, some_extra_kwarg=42)
        assert tuple(out.shape) == (1, 1, 32, 32)
