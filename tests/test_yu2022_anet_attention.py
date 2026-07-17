"""Unit tests for ANetNonLocalAttention2D — exactness, globality, chunking."""

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

from model.interpolation.yu2022_anet import ANetNonLocalAttention2D  # noqa: E402


@pytest.fixture()
def attention() -> ANetNonLocalAttention2D:
    torch.manual_seed(0)
    return ANetNonLocalAttention2D(channels=8)


class TestShapeAndNormalization:
    def test_output_shape_matches_input(self, attention: ANetNonLocalAttention2D) -> None:
        x = torch.randn(2, 8, 6, 5)
        assert attention(x).shape == x.shape

    def test_attention_matrix_shape(self, attention: ANetNonLocalAttention2D) -> None:
        x = torch.randn(2, 8, 6, 5)
        attn = attention.compute_attention(x)
        assert attn.shape == (2, 30, 30)

    def test_rows_sum_to_one(self, attention: ANetNonLocalAttention2D) -> None:
        x = torch.randn(2, 8, 6, 5)
        attn = attention.compute_attention(x)
        torch.testing.assert_close(attn.sum(dim=-1), torch.ones(2, 30))

    def test_no_nan_inf(self, attention: ANetNonLocalAttention2D) -> None:
        x = torch.randn(2, 8, 6, 5) * 100.0
        out = attention(x)
        assert torch.isfinite(out).all()


class TestResidualIdentity:
    def test_zeroed_output_projection_is_identity(self) -> None:
        torch.manual_seed(0)
        attention = ANetNonLocalAttention2D(channels=8)
        with torch.no_grad():
            attention.out_proj.weight.zero_()
            attention.out_proj.bias.zero_()
        x = torch.randn(2, 8, 6, 5)
        torch.testing.assert_close(attention(x), x)


class TestGlobalDependency:
    def test_far_pixel_influences_output(self) -> None:
        torch.manual_seed(1)
        attention = ANetNonLocalAttention2D(channels=4)
        x1 = torch.randn(1, 4, 8, 8)
        x2 = x1.clone()
        x2[0, :, 0, 0] += 5.0  # perturb the far corner only
        out1 = attention(x1)
        out2 = attention(x2)
        # Output at the opposite corner must change: attention is global.
        assert not torch.allclose(out1[0, :, 7, 7], out2[0, :, 7, 7])


class TestQueryChunking:
    def test_chunked_equals_unchunked(self) -> None:
        torch.manual_seed(2)
        attention = ANetNonLocalAttention2D(channels=8)
        x = torch.randn(2, 8, 6, 5)
        full = attention(x)
        attention.query_chunk_size = 4  # 30 queries -> chunks of 4
        chunked = attention(x)
        torch.testing.assert_close(full, chunked, rtol=1e-5, atol=1e-6)

    def test_chunk_larger_than_n(self) -> None:
        torch.manual_seed(2)
        attention = ANetNonLocalAttention2D(channels=8, query_chunk_size=10_000)
        x = torch.randn(1, 8, 4, 4)
        assert attention(x).shape == x.shape

    def test_chunked_gradients_flow(self) -> None:
        torch.manual_seed(3)
        attention = ANetNonLocalAttention2D(channels=4, query_chunk_size=3)
        x = torch.randn(1, 4, 5, 5, requires_grad=True)
        attention(x).sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        for p in attention.parameters():
            assert p.grad is not None


class TestConfiguration:
    def test_reduced_qk_channels(self) -> None:
        attention = ANetNonLocalAttention2D(channels=8, attention_channels=2)
        assert attention.query.out_channels == 2
        assert attention.key.out_channels == 2
        assert attention.value.out_channels == 8
        x = torch.randn(1, 8, 4, 4)
        assert attention(x).shape == x.shape

    def test_sqrt_channel_scaling_runs(self) -> None:
        attention = ANetNonLocalAttention2D(channels=8, scaling="sqrt_channel")
        x = torch.randn(1, 8, 4, 4)
        assert attention(x).shape == x.shape

    def test_invalid_scaling_rejected(self) -> None:
        with pytest.raises(ValueError):
            ANetNonLocalAttention2D(channels=8, scaling="transformer")

    def test_invalid_chunk_rejected(self) -> None:
        with pytest.raises(ValueError):
            ANetNonLocalAttention2D(channels=8, query_chunk_size=0)

    def test_only_pointwise_convs(self) -> None:
        attention = ANetNonLocalAttention2D(channels=8)
        for m in attention.modules():
            if isinstance(m, nn.Conv2d):
                assert m.kernel_size == (1, 1)
