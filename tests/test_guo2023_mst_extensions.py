"""Extension tests for Guo2023MST — non-profile options and invalid arguments.

All tests use lightweight configs (embed_dims=(8, 16, 32), heads=(1, 2, 4)).
"""

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

from model.interpolation.guo2023_mst import Guo2023MST  # noqa: E402


_LIGHT = dict(embed_dims=(8, 16, 32), depths=(1, 1, 1), num_heads=(1, 2, 4))


# ---------------------------------------------------------------------------
# Output mode variants
# ---------------------------------------------------------------------------

class TestOutputModes:
    def test_direct_output(self) -> None:
        m = Guo2023MST(**_LIGHT, output_mode="direct")
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape

    def test_global_residual(self) -> None:
        m = Guo2023MST(**_LIGHT, output_mode="global_residual")
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape

    def test_global_residual_is_not_identity_at_init(self) -> None:
        m = Guo2023MST(**_LIGHT, output_mode="global_residual")
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        # With random init, residual output should differ from input
        assert not torch.allclose(y, x)


# ---------------------------------------------------------------------------
# Positional encoding variants
# ---------------------------------------------------------------------------

class TestPositionalEncoding:
    @pytest.mark.parametrize("pe", ["none", "sincos_2d"])
    def test_build_and_forward(self, pe: str) -> None:
        m = Guo2023MST(**_LIGHT, positional_encoding=pe)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Kernel size variants
# ---------------------------------------------------------------------------

class TestKernelSizes:
    @pytest.mark.parametrize("ks", [3, 5, 7])
    def test_build_and_forward(self, ks: int) -> None:
        m = Guo2023MST(**_LIGHT, kernel_size=ks)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Downsample / upsample variants
# ---------------------------------------------------------------------------

class TestDownsampleVariants:
    def test_stride_conv(self) -> None:
        m = Guo2023MST(**_LIGHT, downsample_mode="stride_conv")
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape

    def test_maxpool(self) -> None:
        m = Guo2023MST(**_LIGHT, downsample_mode="maxpool")
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape


class TestUpsampleVariants:
    @pytest.mark.parametrize("mode", ["bilinear", "nearest", "transpose"])
    def test_build_and_forward(self, mode: str) -> None:
        m = Guo2023MST(**_LIGHT, upsample_mode=mode)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Depth variants
# ---------------------------------------------------------------------------

class TestDepthVariants:
    def test_deeper_transformers(self) -> None:
        m = Guo2023MST(embed_dims=(8, 16, 32), depths=(2, 2, 2),
                       num_heads=(1, 2, 4))
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Channel variants
# ---------------------------------------------------------------------------

class TestChannelVariants:
    def test_two_channel_input(self) -> None:
        m = Guo2023MST(**_LIGHT, in_channels=2)
        x = torch.randn(1, 2, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, 1, 64, 64)

    def test_two_channel_output(self) -> None:
        m = Guo2023MST(**_LIGHT, out_channels=2)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, 2, 64, 64)


# ---------------------------------------------------------------------------
# Dropout and drop path
# ---------------------------------------------------------------------------

class TestDropout:
    def test_dropout_in_train(self) -> None:
        m = Guo2023MST(**_LIGHT, dropout=0.5)
        m.train()
        x = torch.randn(1, 1, 64, 64)
        y1 = m(x)
        y2 = m(x)
        assert not torch.equal(y1, y2)

    def test_dropout_noop_in_eval(self) -> None:
        m = Guo2023MST(**_LIGHT, dropout=0.5)
        m.eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y1 = m(x)
            y2 = m(x)
        torch.testing.assert_close(y1, y2)


# ---------------------------------------------------------------------------
# Global residual validation
# ---------------------------------------------------------------------------

class TestGlobalResidualValidation:
    def test_channel_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="global_residual requires"):
            Guo2023MST(**_LIGHT, in_channels=1, out_channels=2,
                       output_mode="global_residual")


# ---------------------------------------------------------------------------
# Invalid arguments
# ---------------------------------------------------------------------------

class TestInvalidArguments:
    def test_num_scales_zero(self) -> None:
        with pytest.raises(ValueError, match="num_scales must be"):
            Guo2023MST(num_scales=0, embed_dims=(32,), depths=(1,), num_heads=(4,))

    def test_embed_dims_mismatch_num_scales(self) -> None:
        with pytest.raises(ValueError, match="len\\(embed_dims\\)"):
            Guo2023MST(num_scales=3, embed_dims=(32, 64), depths=(1, 1, 1),
                       num_heads=(4, 4, 8))

    def test_depths_mismatch_num_scales(self) -> None:
        with pytest.raises(ValueError, match="len\\(depths\\)"):
            Guo2023MST(num_scales=3, embed_dims=(32, 64, 128), depths=(1, 1),
                       num_heads=(4, 4, 8))

    def test_num_heads_mismatch_num_scales(self) -> None:
        with pytest.raises(ValueError, match="len\\(num_heads\\)"):
            Guo2023MST(num_scales=3, embed_dims=(32, 64, 128), depths=(1, 1, 1),
                       num_heads=(4, 4))

    def test_embed_dim_not_divisible(self) -> None:
        with pytest.raises(ValueError, match="must be divisible"):
            Guo2023MST(num_scales=1, embed_dims=(32,), depths=(1,),
                       num_heads=(3,))

    def test_embed_dim_zero(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            Guo2023MST(num_scales=1, embed_dims=(0,), depths=(1,), num_heads=(2,))

    def test_num_heads_zero(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            Guo2023MST(num_scales=1, embed_dims=(32,), depths=(1,), num_heads=(0,))

    def test_depth_zero(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            Guo2023MST(num_scales=1, embed_dims=(32,), depths=(0,), num_heads=(4,))

    def test_mlp_ratio_zero(self) -> None:
        with pytest.raises(ValueError, match="mlp_ratio must be"):
            Guo2023MST(mlp_ratio=0.0)

    def test_kernel_size_even(self) -> None:
        with pytest.raises(ValueError, match="kernel_size must be"):
            Guo2023MST(kernel_size=4)

    def test_dropout_negative(self) -> None:
        with pytest.raises(ValueError, match="dropout must be"):
            Guo2023MST(dropout=-0.1)

    def test_dropout_at_least_one(self) -> None:
        with pytest.raises(ValueError, match="dropout must be"):
            Guo2023MST(dropout=1.0)

    def test_attention_dropout_negative(self) -> None:
        with pytest.raises(ValueError, match="attention_dropout must be"):
            Guo2023MST(attention_dropout=-0.1)

    def test_drop_path_negative(self) -> None:
        with pytest.raises(ValueError, match="drop_path must be"):
            Guo2023MST(drop_path=-0.1)

    def test_unsupported_downsample_mode(self) -> None:
        with pytest.raises(ValueError, match="Unknown downsample_mode"):
            Guo2023MST(downsample_mode="avg_pool")

    def test_unsupported_upsample_mode(self) -> None:
        with pytest.raises(ValueError, match="Unknown upsample_mode"):
            Guo2023MST(upsample_mode="pixelshuffle")

    def test_unsupported_positional_encoding(self) -> None:
        with pytest.raises(ValueError, match="Unknown positional_encoding"):
            Guo2023MST(positional_encoding="learned")

    def test_unsupported_output_mode(self) -> None:
        with pytest.raises(ValueError, match="Unknown output_mode"):
            Guo2023MST(output_mode="residual")

    def test_unsupported_norm_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown norm_type"):
            Guo2023MST(norm_type="spectral")

    def test_unsupported_activation(self) -> None:
        with pytest.raises(ValueError, match="Unknown activation"):
            Guo2023MST(activation="swish")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_not_4d_raises(self) -> None:
        m = Guo2023MST(**_LIGHT)
        with pytest.raises(ValueError, match="Expected input shape"):
            m._validate_input(torch.randn(1, 64))

    def test_wrong_channels_raises(self) -> None:
        m = Guo2023MST(**_LIGHT, in_channels=1)
        with pytest.raises(ValueError, match="Expected 1 input channel"):
            m._validate_input(torch.randn(1, 3, 64, 64))

    def test_height_too_small_raises(self) -> None:
        m = Guo2023MST(**_LIGHT)
        with pytest.raises(ValueError, match="height must be"):
            m._validate_input(torch.randn(1, 1, 1, 64))

    def test_width_too_small_raises(self) -> None:
        m = Guo2023MST(**_LIGHT)
        with pytest.raises(ValueError, match="width must be"):
            m._validate_input(torch.randn(1, 1, 64, 1))
