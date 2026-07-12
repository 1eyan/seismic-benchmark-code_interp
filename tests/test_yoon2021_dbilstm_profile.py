"""Profile tests for Yoon2021DBiLSTM — conservative reproduction configuration.

Architecture checks use the default 3-layer model.
Functional checks use the default hidden_sizes=(64, 128, 128).
"""

from __future__ import annotations

import io
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

from model.interpolation.yoon2021_dbilstm import (  # noqa: E402
    Yoon2021DBiLSTM,
    YoonDBiLSTMCore,
    _BiLSTMSkipLayer,
)
from model.registry import build_model  # noqa: E402


def _trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Architecture tests (default model)
# ---------------------------------------------------------------------------

class TestArchitecture:
    """Verify conservative reproduction profile using default constructor args."""

    @pytest.fixture(scope="class")
    def model(self) -> Yoon2021DBiLSTM:
        return Yoon2021DBiLSTM()

    def test_three_bilstm_layers(self, model: Yoon2021DBiLSTM) -> None:
        assert len(model.core.layers) == 3

    def test_hidden_sizes(self, model: Yoon2021DBiLSTM) -> None:
        assert model.core.hidden_sizes == (64, 128, 128)

    def test_input_features(self, model: Yoon2021DBiLSTM) -> None:
        assert model.core.input_features == 2

    def test_output_features(self, model: Yoon2021DBiLSTM) -> None:
        assert model.core.output_features == 1

    def test_lstm_is_bidirectional(self, model: Yoon2021DBiLSTM) -> None:
        for layer in model.core.layers:
            assert layer.lstm.bidirectional is True

    def test_lstm_batch_first(self, model: Yoon2021DBiLSTM) -> None:
        for layer in model.core.layers:
            assert layer.lstm.batch_first is True

    def test_skip_projections_exist(self, model: Yoon2021DBiLSTM) -> None:
        for i, layer in enumerate(model.core.layers):
            assert layer.use_skip is True
            assert layer.skip_projection is not None

    def test_dropout_rate(self, model: Yoon2021DBiLSTM) -> None:
        assert model.core.dropout_rate == 0.2

    def test_output_head_is_linear(self, model: Yoon2021DBiLSTM) -> None:
        assert isinstance(model.core.output_head, nn.Linear)
        assert model.core.output_head.in_features == 256
        assert model.core.output_head.out_features == 1

    def test_no_cnn_modules(self, model: Yoon2021DBiLSTM) -> None:
        for bad in (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
            count = sum(1 for _m in model.modules() if isinstance(_m, bad))
            assert count == 0, f"Found {bad.__name__}"

    def test_parameter_count(self, model: Yoon2021DBiLSTM) -> None:
        params = _trainable_params(model)
        assert params == 727553, f"Expected 727553, got {params}"

    def test_default_input_mode(self, model: Yoon2021DBiLSTM) -> None:
        assert model.input_mode == "regular_midpoint_4d"


# ---------------------------------------------------------------------------
# Core network shape tests
# ---------------------------------------------------------------------------

class TestCoreShapes:
    """Native core [B, T, 2] -> [B, T, 1] shape contract."""

    @pytest.fixture(scope="class")
    def core(self) -> YoonDBiLSTMCore:
        return YoonDBiLSTMCore()

    @pytest.mark.parametrize("shape", [
        (1, 256, 2),
        (4, 128, 2),
        (2, 512, 2),
        (1, 1, 2),
    ])
    def test_output_shape(self, core: YoonDBiLSTMCore, shape: tuple) -> None:
        x = torch.randn(*shape)
        with torch.no_grad():
            y = core(x)
        assert y.shape == (shape[0], shape[1], 1)

    def test_output_values_are_finite(self, core: YoonDBiLSTMCore) -> None:
        x = torch.randn(2, 64, 2)
        with torch.no_grad():
            y = core(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_deterministic_in_eval(self, core: YoonDBiLSTMCore) -> None:
        core.eval()
        x = torch.randn(1, 64, 2)
        with torch.no_grad():
            y1 = core(x)
            y2 = core(x)
        torch.testing.assert_close(y1, y2)


# ---------------------------------------------------------------------------
# Skip connection unit tests
# ---------------------------------------------------------------------------

class TestSkipLayer:
    """Projected additive skip behaviour."""

    def test_output_shape(self) -> None:
        layer = _BiLSTMSkipLayer(input_size=2, hidden_size=64)
        x = torch.randn(2, 128, 2)
        out = layer(x)
        assert out.shape == (2, 128, 128)  # 2 * 64

    def test_skip_projection_is_linear_when_mismatched(self) -> None:
        layer = _BiLSTMSkipLayer(input_size=2, hidden_size=64)
        assert isinstance(layer.skip_projection, nn.Linear)
        assert layer.skip_projection.in_features == 2
        assert layer.skip_projection.out_features == 128

    def test_skip_projection_is_identity_when_matched(self) -> None:
        # Layer 3: input 256, hidden 128 -> output 256 (matches input)
        layer = _BiLSTMSkipLayer(input_size=256, hidden_size=128)
        assert isinstance(layer.skip_projection, nn.Identity)

    def test_skip_projection_no_bias(self) -> None:
        layer = _BiLSTMSkipLayer(input_size=2, hidden_size=64)
        assert layer.skip_projection.bias is None

    def test_no_skip_mode(self) -> None:
        layer = _BiLSTMSkipLayer(input_size=2, hidden_size=64, use_skip=False)
        assert layer.skip_projection is None
        x = torch.randn(2, 128, 2)
        out = layer(x)
        assert out.shape == (2, 128, 128)

    def test_skip_changes_output(self) -> None:
        torch.manual_seed(42)
        layer_with = _BiLSTMSkipLayer(input_size=2, hidden_size=64, use_skip=True)
        torch.manual_seed(42)
        layer_without = _BiLSTMSkipLayer(input_size=2, hidden_size=64, use_skip=False)
        x = torch.randn(1, 32, 2)
        with torch.no_grad():
            y_with = layer_with(x)
            y_without = layer_without(x)
        assert not torch.allclose(y_with, y_without)


# ---------------------------------------------------------------------------
# No-skip core variant
# ---------------------------------------------------------------------------

class TestCoreNoSkip:
    """Core without skip connections still produces correct shapes."""

    def test_no_skip_shape(self) -> None:
        core = YoonDBiLSTMCore(use_skip=False)
        x = torch.randn(2, 64, 2)
        with torch.no_grad():
            y = core(x)
        assert y.shape == (2, 64, 1)


# ---------------------------------------------------------------------------
# 4D gather adapter tests
# ---------------------------------------------------------------------------

class TestRegularMidpoint4D:
    """Full-gather adapter: [B, 1, T, X] -> [B, 1, T, X]."""

    @pytest.fixture(scope="class")
    def model(self) -> Yoon2021DBiLSTM:
        return Yoon2021DBiLSTM(input_mode="regular_midpoint_4d",
                               target_stride=2, target_offset=1)

    def test_output_shape(self, model: Yoon2021DBiLSTM) -> None:
        x = torch.randn(1, 1, 64, 9)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_known_traces_preserved(self, model: Yoon2021DBiLSTM) -> None:
        """Non-target traces (even indices with offset=1, stride=2) are unchanged."""
        x = torch.randn(1, 1, 64, 9)
        with torch.no_grad():
            y = model(x)
        # Known trace indices: 0, 2, 4, 6, 8 (targets are 1, 3, 5, 7)
        for j in [0, 2, 4, 6, 8]:
            torch.testing.assert_close(y[:, :, :, j], x[:, :, :, j])

    def test_predictions_differ_from_input(self, model: Yoon2021DBiLSTM) -> None:
        """Target traces should be modified (not copied from input)."""
        x = torch.randn(1, 1, 64, 9)
        with torch.no_grad():
            y = model(x)
        # Target traces at indices 1, 3, 5, 7 should differ
        any_differ = False
        for j in [1, 3, 5, 7]:
            if not torch.allclose(y[:, :, :, j], x[:, :, :, j]):
                any_differ = True
                break
        assert any_differ, "All target traces were unchanged"

    def test_batch_independence(self, model: Yoon2021DBiLSTM) -> None:
        x = torch.randn(4, 1, 32, 9)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_no_target_positions_returns_input(self) -> None:
        """When there are no valid target positions, output equals input."""
        m = Yoon2021DBiLSTM(input_mode="regular_midpoint_4d",
                            target_stride=100, target_offset=0)
        x = torch.randn(1, 1, 32, 5)
        with torch.no_grad():
            y = m(x)
        torch.testing.assert_close(y, x)


# ---------------------------------------------------------------------------
# Chunking consistency
# ---------------------------------------------------------------------------

class TestChunking:
    """Spatial chunking must produce identical results to no chunking."""

    def test_chunking_matches_no_chunking(self) -> None:
        torch.manual_seed(42)
        m_nochunk = Yoon2021DBiLSTM(
            input_mode="regular_midpoint_4d",
            target_stride=2, target_offset=1,
            spatial_chunk_size=None,
        )
        torch.manual_seed(42)
        m_chunked = Yoon2021DBiLSTM(
            input_mode="regular_midpoint_4d",
            target_stride=2, target_offset=1,
            spatial_chunk_size=2,
        )
        x = torch.randn(1, 1, 32, 15)  # 7 target traces, chunk_size=2 -> 4 chunks
        m_nochunk.eval()
        m_chunked.eval()
        with torch.no_grad():
            y_nochunk = m_nochunk(x)
            y_chunked = m_chunked(x)
        torch.testing.assert_close(y_chunked, y_nochunk)


# ---------------------------------------------------------------------------
# Input mode shape tests
# ---------------------------------------------------------------------------

class TestInputModes:
    """Each input mode preserves its shape contract."""

    def test_pair_btf(self) -> None:
        m = Yoon2021DBiLSTM(input_mode="pair_btf")
        x = torch.randn(2, 128, 2)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 128, 1)

    def test_pair_bct(self) -> None:
        m = Yoon2021DBiLSTM(input_mode="pair_bct")
        x = torch.randn(2, 2, 128)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 1, 128)

    def test_pair_bct1(self) -> None:
        m = Yoon2021DBiLSTM(input_mode="pair_bct1")
        x = torch.randn(2, 2, 128, 1)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 1, 128, 1)


# ---------------------------------------------------------------------------
# Forward / Backward
# ---------------------------------------------------------------------------

class TestForwardBackward:
    """Gradient flow and numerical stability."""

    def test_forward_backward(self) -> None:
        model = Yoon2021DBiLSTM(input_mode="regular_midpoint_4d")
        x = torch.randn(2, 1, 64, 9)
        y = model(x)
        loss = y.square().mean()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_input_not_modified(self) -> None:
        model = Yoon2021DBiLSTM(input_mode="regular_midpoint_4d")
        x = torch.randn(1, 1, 64, 9)
        x_clone = x.clone()
        with torch.no_grad():
            model(x)
        torch.testing.assert_close(x, x_clone)


# ---------------------------------------------------------------------------
# Factory and serialization
# ---------------------------------------------------------------------------

class TestFactoryAndSerialization:
    """Registry factory and state-dict roundtrip."""

    def test_factory(self) -> None:
        model = build_model({
            "type": "yoon2021_dbilstm",
            "params": {"input_mode": "regular_midpoint_4d"},
        })
        assert isinstance(model, Yoon2021DBiLSTM)
        x = torch.randn(1, 1, 64, 9)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_serialization(self) -> None:
        torch.manual_seed(42)
        model = Yoon2021DBiLSTM(input_mode="regular_midpoint_4d")
        x = torch.randn(1, 1, 64, 9)
        model.eval()
        with torch.no_grad():
            expected = model(x)

        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)

        model2 = Yoon2021DBiLSTM(input_mode="regular_midpoint_4d")
        model2.load_state_dict(torch.load(buf, weights_only=True))
        model2.eval()
        with torch.no_grad():
            actual = model2(x)
        torch.testing.assert_close(actual, expected)


# ---------------------------------------------------------------------------
# Trainer compatibility
# ---------------------------------------------------------------------------

class TestTrainerCompatibility:
    """Model must accept mask, positions, and extra kwargs."""

    def test_kwargs_ignored(self) -> None:
        model = Yoon2021DBiLSTM(input_mode="regular_midpoint_4d")
        x = torch.randn(1, 1, 64, 9)
        with torch.no_grad():
            y = model(x, mask=torch.zeros(1, 1, 64, 9), positions=None, extra=42)
        assert y.shape == x.shape
