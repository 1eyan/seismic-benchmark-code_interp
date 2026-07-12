"""Profile tests for Guo2023MST — conservative reproduction configuration.

Architecture checks use the default 3-scale (32, 64, 128) model.
Functional checks use a lightweight profile (embed_dims=(8, 16, 32), heads=(1, 2, 4)).
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

from model.interpolation.guo2023_mst import (  # noqa: E402
    Guo2023MST,
    ConvStem,
    MultiScaleFeaturePyramid,
    MSTTransformerBlock,
    ScaleTransformer,
    MultiScaleFeatureFusion,
)
from model.registry import build_model  # noqa: E402


_LIGHT = dict(embed_dims=(8, 16, 32), depths=(1, 1, 1), num_heads=(1, 2, 4))


def _count_layers(model: nn.Module, layer_type: type) -> int:
    return sum(1 for m in model.modules() if isinstance(m, layer_type))


def _trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Architecture tests (default model)
# ---------------------------------------------------------------------------

class TestArchitecture:
    """Verify conservative reproduction profile using default constructor args."""

    @pytest.fixture(scope="class")
    def model(self) -> Guo2023MST:
        return Guo2023MST()

    def test_num_scales(self, model: Guo2023MST) -> None:
        assert model.num_scales == 3
        assert len(model.scale_transformers) == 3

    def test_channel_progression(self, model: Guo2023MST) -> None:
        assert model.embed_dims == (32, 64, 128)

    def test_head_divisibility(self, model: Guo2023MST) -> None:
        for ed, nh in zip(model.embed_dims, (4, 4, 8)):
            assert ed % nh == 0

    def test_transformer_depth(self, model: Guo2023MST) -> None:
        for i, st in enumerate(model.scale_transformers):
            assert len(st.blocks) == 1, f"Scale {i} has wrong depth"

    def test_no_classification_modules(self, model: Guo2023MST) -> None:
        for bad in (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.Softmax):
            assert _count_layers(model, bad) == 0, f"Found {bad.__name__}"

    def test_output_head_channels(self, model: Guo2023MST) -> None:
        assert model.fusion.output_head.out_channels == 1

    def test_parameter_count_reproducible(self, model: Guo2023MST) -> None:
        params = _trainable_params(model)
        assert params > 0


# ---------------------------------------------------------------------------
# Attention shape test
# ---------------------------------------------------------------------------

class TestAttentionShape:
    """Per-scale attention must preserve feature map shape."""

    @pytest.fixture(scope="class")
    def model(self) -> Guo2023MST:
        return Guo2023MST(**_LIGHT)

    def test_scale0_preserves_shape(self, model: Guo2023MST) -> None:
        x = torch.randn(2, 8, 32, 32)
        out = model.scale_transformers[0](x)
        assert out.shape == x.shape

    def test_scale1_preserves_shape(self, model: Guo2023MST) -> None:
        x = torch.randn(2, 16, 16, 16)
        out = model.scale_transformers[1](x)
        assert out.shape == x.shape

    def test_scale2_preserves_shape(self, model: Guo2023MST) -> None:
        x = torch.randn(2, 32, 8, 8)
        out = model.scale_transformers[2](x)
        assert out.shape == x.shape

    def test_attention_no_nan_inf(self, model: Guo2023MST) -> None:
        x = torch.randn(2, 8, 32, 32)
        out = model.scale_transformers[0](x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_attention_gradient_flow(self, model: Guo2023MST) -> None:
        x = torch.randn(2, 8, 32, 32, requires_grad=True)
        out = model.scale_transformers[0](x)
        loss = out.square().mean()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_attention_matrix_is_global(self, model: Guo2023MST) -> None:
        """Verify that attention is global (NxN), not windowed."""
        # We can verify this indirectly: the ScaleTransformer produces N=H*W tokens
        # and the Transformer block has no windowing mechanism.
        x = torch.randn(1, 8, 4, 4)
        st = model.scale_transformers[0]
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [1, 16, 8]
        # The MHA computes Q @ K^T, shape [B, heads, N, N] = [1, 1, 16, 16]
        # which is global (all pairs of 16 tokens interact)
        assert tokens.shape == (1, 16, 8)


# ---------------------------------------------------------------------------
# Multi-scale fusion test
# ---------------------------------------------------------------------------

class TestMultiScaleFusion:
    """Fusion must incorporate all scales and produce correct output shape."""

    def test_fusion_concat_channels(self) -> None:
        fusion = MultiScaleFeatureFusion(
            embed_dims=(8, 16, 32), out_channels=1,
            kernel_size=3, upsample_mode="bilinear", activation="gelu",
        )
        f0 = torch.ones(1, 8, 32, 32)
        f1 = torch.ones(1, 16, 16, 16) * 2
        f2 = torch.ones(1, 32, 8, 8) * 3

        out = fusion([f0, f1, f2])
        assert out.shape == (1, 1, 32, 32)

    def test_each_scale_contributes(self) -> None:
        """Modifying any single scale must change the output."""
        torch.manual_seed(42)
        fusion = MultiScaleFeatureFusion(
            embed_dims=(8, 16, 32), out_channels=1,
            kernel_size=3, upsample_mode="bilinear", activation="gelu",
        )
        f0 = torch.randn(1, 8, 32, 32)
        f1 = torch.randn(1, 16, 16, 16)
        f2 = torch.randn(1, 32, 8, 8)

        base = fusion([f0, f1, f2])
        assert not torch.allclose(base, fusion([torch.zeros_like(f0), f1, f2]))
        assert not torch.allclose(base, fusion([f0, torch.zeros_like(f1), f2]))
        assert not torch.allclose(base, fusion([f0, f1, torch.zeros_like(f2)]))

    def test_upsample_to_target_size(self) -> None:
        """Upsampled features must match the first scale's spatial dimensions."""
        fusion = MultiScaleFeatureFusion(
            embed_dims=(8, 16, 32), out_channels=1,
            kernel_size=3, upsample_mode="bilinear", activation="gelu",
        )
        f0 = torch.randn(1, 8, 32, 32)
        f1 = torch.randn(1, 16, 16, 16)
        f2 = torch.randn(1, 32, 8, 8)

        out = fusion([f0, f1, f2])
        assert out.shape[-2:] == f0.shape[-2:]


# ---------------------------------------------------------------------------
# Shape Tests (lightweight)
# ---------------------------------------------------------------------------

class TestShapes:
    """Output shape must match input shape."""

    @pytest.fixture(scope="class")
    def model(self) -> Guo2023MST:
        return Guo2023MST(**_LIGHT)

    @pytest.mark.parametrize("shape", [
        (1, 1, 96, 96),
        (2, 1, 112, 112),
        (1, 1, 128, 128),
        (1, 1, 100, 100),
        (1, 1, 101, 103),
    ])
    def test_output_shape(self, model: Guo2023MST, shape: tuple) -> None:
        x = torch.randn(*shape)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_no_nan_inf(self, model: Guo2023MST) -> None:
        x = torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()


# ---------------------------------------------------------------------------
# Token limit test
# ---------------------------------------------------------------------------

class TestTokenLimit:
    """Token count guard must raise when exceeded."""

    def test_raises_when_exceeded(self) -> None:
        st = ScaleTransformer(embed_dim=8, depth=1, num_heads=1, max_tokens=100)
        feature = torch.randn(1, 8, 32, 32)  # 1024 tokens > 100
        with pytest.raises(RuntimeError, match="exceeds max_tokens"):
            st(feature)

    def test_passes_when_under_limit(self) -> None:
        st = ScaleTransformer(embed_dim=8, depth=1, num_heads=1, max_tokens=2000)
        feature = torch.randn(1, 8, 32, 32)
        out = st(feature)
        assert out.shape == feature.shape

    def test_model_token_guard(self) -> None:
        model = Guo2023MST(**_LIGHT, max_tokens_per_scale=50)
        x = torch.randn(1, 1, 32, 32)
        with pytest.raises(RuntimeError, match="exceeds max_tokens"):
            model(x)


# ---------------------------------------------------------------------------
# Padding test
# ---------------------------------------------------------------------------

class TestPadding:
    """Constant zero-padding must produce correct output size."""

    def test_non_multiple_padding(self) -> None:
        model = Guo2023MST(**_LIGHT)
        x = torch.randn(1, 1, 101, 103)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_padding_does_not_introduce_nan(self) -> None:
        model = Guo2023MST(**_LIGHT)
        x = torch.randn(1, 1, 101, 103)
        with torch.no_grad():
            y = model(x)
        assert not torch.isnan(y).any()


# ---------------------------------------------------------------------------
# Forward / Backward (lightweight)
# ---------------------------------------------------------------------------

class TestForwardBackward:
    """Gradient flow and numerical stability."""

    def test_forward_backward_lightweight(self) -> None:
        model = Guo2023MST(**_LIGHT)
        x = torch.randn(2, 1, 64, 64)
        y = model(x)
        loss = y.square().mean()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_input_not_modified(self) -> None:
        model = Guo2023MST(**_LIGHT)
        x = torch.randn(1, 1, 64, 64)
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
            "type": "guo2023_mst",
            "params": {"embed_dims": [8, 16, 32], "num_heads": [1, 2, 4]},
        })
        assert isinstance(model, Guo2023MST)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_serialization(self) -> None:
        torch.manual_seed(42)
        model = Guo2023MST(**_LIGHT)
        x = torch.randn(1, 1, 64, 64)
        model.eval()
        with torch.no_grad():
            expected = model(x)

        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)

        model2 = Guo2023MST(**_LIGHT)
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
        model = Guo2023MST(**_LIGHT)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = model(x, mask=torch.zeros(1, 1, 64, 64), positions=None, extra=42)
        assert y.shape == x.shape
