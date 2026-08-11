"""Fourier zero-padding upsampler math tests: constant, sine, Nyquist, amplitude, shapes, gradients."""

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

from model.interpolation.park2022_cfunet import FourierZeroPaddingUpsample2D  # noqa: E402


class TestShapes:
    @pytest.mark.parametrize("shape,scale,expected", [
        ((2, 3, 16, 20), 2, (2, 3, 32, 40)),
        ((1, 1, 32, 32), 2, (1, 1, 64, 64)),
        ((2, 1, 8, 12), 4, (2, 1, 32, 48)),
    ])
    def test_output_shape(self, shape, scale, expected):
        up = FourierZeroPaddingUpsample2D(scale_factor=scale)
        out = up(torch.rand(shape))
        assert tuple(out.shape) == expected
        assert out.dtype == torch.float32

    def test_odd_size_raises(self):
        up = FourierZeroPaddingUpsample2D(scale_factor=2)
        with pytest.raises(ValueError, match="even"):
            up(torch.rand(1, 1, 15, 16))

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            FourierZeroPaddingUpsample2D(scale_factor=1)
        with pytest.raises(ValueError):
            FourierZeroPaddingUpsample2D(fft_norm="bogus")
        with pytest.raises(ValueError):
            FourierZeroPaddingUpsample2D(amplitude_correction="bogus")


class TestAmplitudeCorrection:
    def test_constant_backward_scaled(self):
        """Aligned samples of a constant are preserved exactly (x s^2)."""
        up = FourierZeroPaddingUpsample2D(scale_factor=2, fft_norm="backward")
        x = torch.full((1, 1, 16, 16), 0.7)
        out = up(x)
        torch.testing.assert_close(out, x, rtol=1e-5, atol=1e-6)

    def test_constant_backward_uncorrected(self):
        """Without correction, backward norm scales aligned samples by 1/s^2."""
        up = FourierZeroPaddingUpsample2D(
            scale_factor=2, fft_norm="backward", amplitude_correction="none"
        )
        x = torch.full((1, 1, 16, 16), 0.7)
        out = up(x)
        torch.testing.assert_close(out, torch.full_like(out, 0.7 / 4.0), rtol=1e-5, atol=1e-6)

    def test_constant_forward_norm(self):
        """forward norm needs no correction: aligned samples are exact."""
        up = FourierZeroPaddingUpsample2D(
            scale_factor=2, fft_norm="forward", amplitude_correction="none"
        )
        x = torch.full((1, 1, 16, 16), 0.7)
        out = up(x)
        torch.testing.assert_close(out, x, rtol=1e-5, atol=1e-6)

    def test_constant_ortho_norm(self):
        """ortho norm scales aligned samples by 1/s; correction factor is s."""
        up = FourierZeroPaddingUpsample2D(
            scale_factor=2, fft_norm="ortho", amplitude_correction="scale"
        )
        x = torch.full((1, 1, 16, 16), 0.7)
        out = up(x)
        torch.testing.assert_close(out, x, rtol=1e-5, atol=1e-6)

    def test_scale_factor_4(self):
        up = FourierZeroPaddingUpsample2D(scale_factor=4)
        x = torch.full((1, 1, 8, 8), 1.3)
        out = up(x)
        torch.testing.assert_close(out, x, rtol=1e-5, atol=1e-6)


class TestBandlimitedSines:
    def _sine(self, shape, k0, l0):
        n = torch.arange(shape[2], dtype=torch.float32)[None, None, :, None]
        m = torch.arange(shape[3], dtype=torch.float32)[None, None, None, :]
        return torch.sin(2.0 * torch.pi * (k0 * n / shape[2] + l0 * m / shape[3]))

    @pytest.mark.parametrize("k0,l0", [(2, 3), (1, 1), (7, 2)])
    def test_aligned_bins_exact_everywhere(self, k0, l0):
        """A sine on aligned bins is reproduced exactly at every grid point."""
        up = FourierZeroPaddingUpsample2D(scale_factor=2)
        x = self._sine((1, 1, 16, 20), k0, l0)
        out = up(x)
        ref = self._sine((1, 1, 32, 40), k0, l0)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)

    def test_nyquist_pattern(self):
        """Nyquist (alternating) input stays alternating after upsampling."""
        up = FourierZeroPaddingUpsample2D(scale_factor=2)
        n = torch.arange(16)[None, None, :, None]
        m = torch.arange(16)[None, None, None, :]
        x = ((-1.0) ** (n + m)).to(dtype=torch.float32)
        out = up(x)
        n2 = torch.arange(32)[None, None, :, None]
        m2 = torch.arange(32)[None, None, None, :]
        ref = ((-1.0) ** (n2 + m2)).to(dtype=torch.float32)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


class TestGradientFlow:
    def test_backward_flows(self):
        up = FourierZeroPaddingUpsample2D(scale_factor=2)
        x = torch.randn(2, 1, 16, 16, requires_grad=True)
        out = up(x)
        loss = out.square().sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0
