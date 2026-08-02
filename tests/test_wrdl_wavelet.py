"""DWT/IWT tests for Liu2022WRDL: perfect reconstruction, subband direction, filter properties."""

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

from model.interpolation.liu2022_wrdl import (  # noqa: E402
    FixedWaveletDWT2D,
    FixedWaveletIWT2D,
    _make_haar_analysis_filters,
    _make_haar_synthesis_filters,
)


class TestPerfectReconstruction:
    @pytest.mark.parametrize("shape", [
        (1, 1, 32, 32),
        (2, 4, 64, 48),
        (1, 8, 128, 128),
    ])
    def test_iwt_dwt_identity(self, shape):
        dwt = FixedWaveletDWT2D()
        iwt = FixedWaveletIWT2D()
        x = torch.rand(shape)
        recon = iwt(dwt(x))
        torch.testing.assert_close(recon, x, rtol=1e-5, atol=1e-6)

    def test_multi_channel_equivalence(self):
        """Each channel is processed independently."""
        dwt = FixedWaveletDWT2D()
        iwt = FixedWaveletIWT2D()
        x = torch.rand(2, 3, 32, 32)
        coeffs = dwt(x)
        assert coeffs.shape == (2, 12, 16, 16)
        recon = iwt(coeffs)
        assert recon.shape == (2, 3, 32, 32)
        torch.testing.assert_close(recon, x, rtol=1e-5, atol=1e-6)


class TestFilterFrozen:
    def test_dwt_filters_not_trainable(self):
        dwt = FixedWaveletDWT2D()
        assert isinstance(dwt.filters, torch.Tensor)
        assert not dwt.filters.requires_grad

    def test_iwt_filters_not_trainable(self):
        iwt = FixedWaveletIWT2D()
        assert isinstance(iwt.filters, torch.Tensor)
        assert not iwt.filters.requires_grad

    def test_filters_not_parameter(self):
        dwt = FixedWaveletDWT2D()
        iwt = FixedWaveletIWT2D()
        param_names = {n for n, _ in dwt.named_parameters()}
        assert "filters" not in param_names
        param_names = {n for n, _ in iwt.named_parameters()}
        assert "filters" not in param_names

    def test_save_load_preserves_filters(self):
        dwt = FixedWaveletDWT2D()
        buf = dwt.state_dict()
        dwt2 = FixedWaveletDWT2D()
        dwt2.load_state_dict(buf)
        assert torch.equal(dwt.filters, dwt2.filters)


class TestDWTShape:
    def test_standard_shape(self):
        dwt = FixedWaveletDWT2D()
        x = torch.rand(1, 1, 64, 64)
        out = dwt(x)
        assert out.shape == (1, 4, 32, 32)

    def test_multi_channel_shape(self):
        dwt = FixedWaveletDWT2D()
        x = torch.rand(2, 3, 48, 96)
        out = dwt(x)
        assert out.shape == (2, 12, 24, 48)

    def test_odd_spatial_raises(self):
        dwt = FixedWaveletDWT2D()
        x = torch.rand(1, 1, 63, 64)
        with pytest.raises(ValueError):
            dwt(x)


class TestIWTShape:
    def test_standard_shape(self):
        iwt = FixedWaveletIWT2D()
        coeffs = torch.rand(1, 4, 32, 32)
        out = iwt(coeffs)
        assert out.shape == (1, 1, 64, 64)

    def test_non_divisible_4_raises(self):
        iwt = FixedWaveletIWT2D()
        coeffs = torch.rand(1, 6, 16, 16)
        with pytest.raises(ValueError):
            iwt(coeffs)


class TestSubbandDirection:
    """Verify subband directional sensitivity for Haar wavelet."""

    def test_constant_input_ll_dominant(self):
        dwt = FixedWaveletDWT2D()
        x = torch.ones(1, 1, 16, 16)
        coeffs = dwt(x)  # (1, 4, 8, 8)
        # Constant input: only LL should be non-zero
        ll = coeffs[:, 0:1]
        lh = coeffs[:, 1:2]
        hl = coeffs[:, 2:3]
        hh = coeffs[:, 3:4]
        # All energy in LL
        assert ll.abs().sum() > 0
        assert lh.abs().sum() == pytest.approx(0.0, abs=1e-5)
        assert hl.abs().sum() == pytest.approx(0.0, abs=1e-5)
        assert hh.abs().sum() == pytest.approx(0.0, abs=1e-5)

    def test_vertical_edge_lh_response(self):
        """Vertical edge (varies in time dim) → energy in LH."""
        dwt = FixedWaveletDWT2D()
        x = torch.zeros(1, 1, 16, 16)
        x[:, :, :8, :] = 1.0  # top half = 1, bottom half = 0 → vertical edge
        coeffs = dwt(x)
        lh = coeffs[:, 1:2]  # LH: low in trace, high in time
        assert lh.abs().sum() > 0.5

    def test_horizontal_edge_hl_response(self):
        """Horizontal edge (varies in trace dim) → energy in HL."""
        dwt = FixedWaveletDWT2D()
        x = torch.zeros(1, 1, 16, 16)
        x[:, :, :, :8] = 1.0  # left half = 1, right half = 0 → horizontal edge
        coeffs = dwt(x)
        hl = coeffs[:, 2:3]  # HL: high in trace, low in time
        assert hl.abs().sum() > 0.5


class TestFilterValues:
    def test_haar_normalization(self):
        """Filters should have L2 norm = 1 per 1D kernel for orthonormal Haar."""
        filters = _make_haar_analysis_filters()
        # Each 2D filter is outer(k1, k2) where k1,k2 have norm 1.
        # The Frobenius norm of outer(k1, k2) is ||k1|| * ||k2|| = 1 * 1 = 1.
        for i in range(4):
            f = filters[i, 0]  # (2, 2)
            assert torch.norm(f) == pytest.approx(1.0, abs=1e-6)

    def test_analysis_synthesis_same_for_haar(self):
        a = _make_haar_analysis_filters()
        s = _make_haar_synthesis_filters()
        assert torch.equal(a, s)
