"""Extension tests for Yoon2021DBiLSTM — input-mode variants, edge cases, and
invalid-argument checks.

All tests use a lightweight profile (hidden_sizes=(8, 16, 16)) for speed.
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

from model.interpolation.yoon2021_dbilstm import (  # noqa: E402
    Yoon2021DBiLSTM,
    YoonDBiLSTMCore,
)


_LIGHT = (8, 16, 16)


# ---------------------------------------------------------------------------
# Input-mode shape variants
# ---------------------------------------------------------------------------

class TestInputModeShapes:
    """Each valid input mode must produce the expected output shape."""

    def test_pair_btf_deep(self) -> None:
        m = Yoon2021DBiLSTM(input_mode="pair_btf", hidden_sizes=_LIGHT)
        x = torch.randn(3, 127, 2)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (3, 127, 1)

    def test_pair_bct_deep(self) -> None:
        m = Yoon2021DBiLSTM(input_mode="pair_bct", hidden_sizes=_LIGHT)
        x = torch.randn(3, 2, 127)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (3, 1, 127)

    def test_pair_bct1_deep(self) -> None:
        m = Yoon2021DBiLSTM(input_mode="pair_bct1", hidden_sizes=_LIGHT)
        x = torch.randn(3, 2, 127, 1)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (3, 1, 127, 1)

    def test_regular_midpoint_4d_deep(self) -> None:
        m = Yoon2021DBiLSTM(input_mode="regular_midpoint_4d", hidden_sizes=_LIGHT)
        x = torch.randn(2, 1, 64, 15)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape

    def test_kwargs_swallowed(self) -> None:
        m = Yoon2021DBiLSTM(input_mode="pair_btf", hidden_sizes=_LIGHT)
        x = torch.randn(1, 64, 2)
        with torch.no_grad():
            y = m(x, mask=torch.zeros(1, 64), positions=None)
        assert y.shape == (1, 64, 1)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary inputs that must not crash."""

    def test_minimum_traces_regular_4d(self) -> None:
        """X=3 means exactly one target position at offset=1, stride=2."""
        m = Yoon2021DBiLSTM(
            input_mode="regular_midpoint_4d",
            hidden_sizes=_LIGHT,
            target_stride=2,
            target_offset=1,
        )
        x = torch.randn(1, 1, 64, 3)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape

    def test_no_target_positions_returns_input(self) -> None:
        """When stride is too large for the gather, output == input."""
        m = Yoon2021DBiLSTM(
            input_mode="regular_midpoint_4d",
            hidden_sizes=_LIGHT,
            target_stride=100,
            target_offset=1,
        )
        x = torch.randn(1, 1, 64, 15)
        with torch.no_grad():
            y = m(x)
        torch.testing.assert_close(y, x)

    def test_single_batch_pair_btf(self) -> None:
        core = YoonDBiLSTMCore(hidden_sizes=_LIGHT)
        x = torch.randn(1, 128, 2)
        with torch.no_grad():
            y = core(x)
        assert y.shape == (1, 128, 1)

    def test_single_timestep(self) -> None:
        core = YoonDBiLSTMCore(hidden_sizes=_LIGHT)
        x = torch.randn(2, 1, 2)
        with torch.no_grad():
            y = core(x)
        assert y.shape == (2, 1, 1)

    def test_chunk_size_greater_than_targets(self) -> None:
        """spatial_chunk_size larger than n_targets still works."""
        m = Yoon2021DBiLSTM(
            input_mode="regular_midpoint_4d",
            hidden_sizes=_LIGHT,
            spatial_chunk_size=999,
        )
        x = torch.randn(1, 1, 64, 7)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape

    def test_chunk_size_one(self) -> None:
        """Degenerate chunk size of 1 exercises the chunk loop carefully."""
        m = Yoon2021DBiLSTM(
            input_mode="regular_midpoint_4d",
            hidden_sizes=_LIGHT,
            spatial_chunk_size=1,
        )
        x = torch.randn(1, 1, 64, 7)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Invalid argument tests
# ---------------------------------------------------------------------------

class TestInvalidArguments:
    """Every invalid argument must raise a clear ValueError."""

    def test_unsupported_input_mode(self) -> None:
        with pytest.raises(ValueError, match="Unsupported input_mode"):
            Yoon2021DBiLSTM(input_mode="bad_mode", hidden_sizes=_LIGHT)

    def test_empty_hidden_sizes(self) -> None:
        with pytest.raises(ValueError, match="hidden_sizes must not be empty"):
            YoonDBiLSTMCore(hidden_sizes=())

    def test_negative_hidden_size(self) -> None:
        with pytest.raises(ValueError, match="All hidden sizes must be positive"):
            YoonDBiLSTMCore(hidden_sizes=(64, 0, 128))

    def test_dropout_negative(self) -> None:
        with pytest.raises(ValueError, match="dropout must be in"):
            YoonDBiLSTMCore(hidden_sizes=_LIGHT, dropout=-0.1)

    def test_dropout_at_least_one(self) -> None:
        with pytest.raises(ValueError, match="dropout must be in"):
            YoonDBiLSTMCore(hidden_sizes=_LIGHT, dropout=1.0)

    def test_target_stride_zero(self) -> None:
        with pytest.raises(ValueError, match="target_stride must be positive"):
            Yoon2021DBiLSTM(input_mode="regular_midpoint_4d", hidden_sizes=_LIGHT,
                            target_stride=0)

    def test_target_stride_negative(self) -> None:
        with pytest.raises(ValueError, match="target_stride must be positive"):
            Yoon2021DBiLSTM(input_mode="regular_midpoint_4d", hidden_sizes=_LIGHT,
                            target_stride=-2)

    def test_target_offset_negative(self) -> None:
        with pytest.raises(ValueError, match="target_offset must be non-negative"):
            Yoon2021DBiLSTM(input_mode="regular_midpoint_4d", hidden_sizes=_LIGHT,
                            target_offset=-1)

    def test_spatial_chunk_size_zero(self) -> None:
        with pytest.raises(ValueError, match="spatial_chunk_size must be positive"):
            Yoon2021DBiLSTM(input_mode="regular_midpoint_4d", hidden_sizes=_LIGHT,
                            spatial_chunk_size=0)

    def test_spatial_chunk_size_negative(self) -> None:
        with pytest.raises(ValueError, match="spatial_chunk_size must be positive"):
            Yoon2021DBiLSTM(input_mode="regular_midpoint_4d", hidden_sizes=_LIGHT,
                            spatial_chunk_size=-1)


# ---------------------------------------------------------------------------
# Input validation error messages
# ---------------------------------------------------------------------------

class TestInputValidation:
    """_validate_input must raise descriptive ValueError per mode."""

    def test_pair_btf_wrong_last_dim(self) -> None:
        m = Yoon2021DBiLSTM(input_mode="pair_btf", hidden_sizes=_LIGHT)
        with pytest.raises(ValueError, match="expects input shape"):
            m._validate_input(torch.randn(1, 64, 3))

    def test_pair_bct_wrong_channels(self) -> None:
        m = Yoon2021DBiLSTM(input_mode="pair_bct", hidden_sizes=_LIGHT)
        with pytest.raises(ValueError, match="expects input shape"):
            m._validate_input(torch.randn(1, 3, 64))

    def test_pair_bct1_wrong_channels(self) -> None:
        m = Yoon2021DBiLSTM(input_mode="pair_bct1", hidden_sizes=_LIGHT)
        with pytest.raises(ValueError, match="expects input shape"):
            m._validate_input(torch.randn(1, 3, 64, 1))

    def test_regular_midpoint_4d_wrong_channels(self) -> None:
        m = Yoon2021DBiLSTM(input_mode="regular_midpoint_4d", hidden_sizes=_LIGHT)
        with pytest.raises(ValueError, match="expects input shape"):
            m._validate_input(torch.randn(1, 3, 64, 15))

    def test_regular_midpoint_4d_too_few_traces(self) -> None:
        m = Yoon2021DBiLSTM(input_mode="regular_midpoint_4d", hidden_sizes=_LIGHT)
        with pytest.raises(ValueError, match="at least 3 traces"):
            m._validate_input(torch.randn(1, 1, 64, 2))


# ---------------------------------------------------------------------------
# Configurable hyper-parameters
# ---------------------------------------------------------------------------

class TestConfigurableParams:
    """Non-profile dropout and stride/offset values."""

    def test_dropout_zero(self) -> None:
        core = YoonDBiLSTMCore(hidden_sizes=_LIGHT, dropout=0.0)
        x = torch.randn(2, 64, 2)
        with torch.no_grad():
            y = core(x)
        assert y.shape == (2, 64, 1)

    def test_different_stride_offset(self) -> None:
        m = Yoon2021DBiLSTM(
            input_mode="regular_midpoint_4d",
            hidden_sizes=_LIGHT,
            target_stride=3,
            target_offset=2,
        )
        x = torch.randn(1, 1, 128, 20)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape

    def test_single_layer(self) -> None:
        core = YoonDBiLSTMCore(hidden_sizes=(16,))
        assert len(core.layers) == 1
        x = torch.randn(2, 32, 2)
        with torch.no_grad():
            y = core(x)
        assert y.shape == (2, 32, 1)

    def test_no_chunking(self) -> None:
        m = Yoon2021DBiLSTM(
            input_mode="regular_midpoint_4d",
            hidden_sizes=_LIGHT,
            spatial_chunk_size=None,
        )
        x = torch.randn(2, 1, 64, 31)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape
