"""Architecture tests for Yu2022ANet — paper topology and structural invariants.

Structure checks use the default paper-profile model (64 base channels,
6 residual blocks); functional shape checks use lightweight variants.
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

from model.interpolation.yu2022_anet import (  # noqa: E402
    ANetNonLocalAttention2D,
    ANetResidualBlock,
    ANetUpsampleStage,
    Yu2022ANet,
)
from model.registry import MODEL_REGISTRY, build_model  # noqa: E402


def _count_modules(model: nn.Module, layer_type: type) -> int:
    return sum(1 for m in model.modules() if isinstance(m, layer_type))


@pytest.fixture(scope="module")
def default_model() -> Yu2022ANet:
    torch.manual_seed(0)
    return Yu2022ANet()


def _small_model(**overrides) -> Yu2022ANet:
    params = dict(base_channels=8, num_residual_blocks=2)
    params.update(overrides)
    torch.manual_seed(0)
    return Yu2022ANet(**params)


class TestPaperTopology:
    def test_registered(self) -> None:
        assert "yu2022_anet" in MODEL_REGISTRY

    def test_factory_build(self) -> None:
        model = build_model({"type": "yu2022_anet", "params": {"base_channels": 8,
                                                               "num_residual_blocks": 2}})
        assert isinstance(model, Yu2022ANet)

    def test_two_stride2_downsampling_stages(self, default_model: Yu2022ANet) -> None:
        assert len(default_model.down_stages) == 2
        for stage in default_model.down_stages:
            conv = stage[0]
            assert isinstance(conv, nn.Conv2d)
            assert conv.stride == (2, 2)

    def test_first_conv_64_channels(self, default_model: Yu2022ANet) -> None:
        assert default_model.down_stages[0][0].out_channels == 64

    def test_channel_doubling(self, default_model: Yu2022ANet) -> None:
        assert default_model.down_stages[1][0].in_channels == 64
        assert default_model.down_stages[1][0].out_channels == 128

    def test_six_residual_blocks(self, default_model: Yu2022ANet) -> None:
        assert len(default_model.residual_blocks) == 6
        assert all(isinstance(b, ANetResidualBlock) for b in default_model.residual_blocks)

    def test_residual_block_two_convs(self, default_model: Yu2022ANet) -> None:
        block = default_model.residual_blocks[0]
        assert _count_modules(block, nn.Conv2d) == 2
        assert block.conv1.in_channels == 128
        assert block.conv2.out_channels == 128

    def test_single_attention_module(self, default_model: Yu2022ANet) -> None:
        assert _count_modules(default_model, ANetNonLocalAttention2D) == 1
        assert isinstance(default_model.attention, ANetNonLocalAttention2D)

    def test_attention_full_channels_by_default(self, default_model: Yu2022ANet) -> None:
        attn = default_model.attention
        assert attn.qk_channels == 128
        assert attn.query.kernel_size == (1, 1)
        assert attn.key.kernel_size == (1, 1)
        assert attn.value.kernel_size == (1, 1)
        assert attn.out_proj.kernel_size == (1, 1)

    def test_two_upsampling_stages(self, default_model: Yu2022ANet) -> None:
        assert len(default_model.upsample_stages) == 2
        assert all(isinstance(s, ANetUpsampleStage) for s in default_model.upsample_stages)

    def test_default_decoder_channels(self, default_model: Yu2022ANet) -> None:
        assert default_model.upsample_stages[0].conv.out_channels == 64
        assert default_model.upsample_stages[1].conv.out_channels == 32

    def test_nearest_conv_upsampling_default(self, default_model: Yu2022ANet) -> None:
        for stage in default_model.upsample_stages:
            assert isinstance(stage.upsample, nn.Upsample)
        assert _count_modules(default_model, nn.ConvTranspose2d) == 0

    def test_transposed_variant(self) -> None:
        model = _small_model(upsample_mode="transposed")
        assert _count_modules(model, nn.ConvTranspose2d) == 2

    def test_final_layer(self, default_model: Yu2022ANet) -> None:
        assert isinstance(default_model.final, nn.Conv2d)
        assert default_model.final.out_channels == 1

    def test_module_order(self, default_model: Yu2022ANet) -> None:
        names = [name for name, _ in default_model.named_children()]
        assert names == ["down_stages", "residual_blocks", "attention",
                         "upsample_stages", "final"]

    def test_batchnorm_everywhere_except_upsample_and_final(
        self, default_model: Yu2022ANet
    ) -> None:
        # 2 down + 12 residual + 2 reconstruction convs carry BN; final has none.
        assert _count_modules(default_model, nn.BatchNorm2d) == 16
        for stage in default_model.upsample_stages:
            assert not isinstance(stage.upsample, nn.BatchNorm2d)

    def test_no_forbidden_modules(self, default_model: Yu2022ANet) -> None:
        forbidden = (
            nn.MultiheadAttention,
            nn.TransformerEncoder,
            nn.TransformerEncoderLayer,
            nn.MaxPool2d,
            nn.AvgPool2d,
            nn.LSTM,
        )
        assert _count_modules(default_model, forbidden) == 0

    def test_no_dilated_convs(self, default_model: Yu2022ANet) -> None:
        for m in default_model.modules():
            if isinstance(m, nn.Conv2d):
                assert m.dilation == (1, 1)

    def test_no_output_activation(self, default_model: Yu2022ANet) -> None:
        x = torch.randn(1, 1, 32, 32) * 10.0
        with torch.no_grad():
            default_model.eval()
            out = default_model(x)
        # A linear output head must be able to produce negative values.
        assert (out < 0).any() or (out > 1).any() or out.abs().max() < 1e-3


class TestShapes:
    @pytest.mark.parametrize("shape", [(1, 1, 128, 128), (2, 1, 128, 128)])
    def test_paper_patch_shapes(self, shape) -> None:
        model = _small_model().eval()
        with torch.no_grad():
            out = model(torch.randn(*shape))
        assert out.shape == shape

    def test_field_patch_shape_with_chunking(self) -> None:
        model = _small_model(attention_query_chunk_size=512).eval()
        with torch.no_grad():
            out = model(torch.randn(1, 1, 720, 120))
        assert out.shape == (1, 1, 720, 120)

    def test_arbitrary_size_zero_padding(self) -> None:
        model = _small_model().eval()
        with torch.no_grad():
            out = model(torch.randn(1, 1, 101, 103))
        assert out.shape == (1, 1, 101, 103)

    def test_multichannel(self) -> None:
        model = _small_model(in_channels=2, out_channels=2).eval()
        with torch.no_grad():
            out = model(torch.randn(1, 2, 32, 32))
        assert out.shape == (1, 2, 32, 32)


class TestValidation:
    def test_even_kernel_rejected(self) -> None:
        with pytest.raises(ValueError):
            Yu2022ANet(kernel_size=4)

    def test_nonpositive_blocks_rejected(self) -> None:
        with pytest.raises(ValueError):
            Yu2022ANet(num_residual_blocks=0)

    def test_bad_upsample_mode_rejected(self) -> None:
        with pytest.raises(ValueError):
            Yu2022ANet(upsample_mode="pixel_shuffle")

    def test_bad_attention_scaling_rejected(self) -> None:
        with pytest.raises(ValueError):
            Yu2022ANet(attention_scaling="softmax_temp")

    def test_bad_decoder_channels_rejected(self) -> None:
        with pytest.raises(ValueError):
            Yu2022ANet(decoder_channels=(64,))
        with pytest.raises(ValueError):
            Yu2022ANet(decoder_channels=(64, 0))

    def test_wrong_input_channels_rejected(self) -> None:
        model = _small_model()
        with pytest.raises(ValueError):
            model(torch.randn(1, 3, 32, 32))

    def test_wrong_ndim_rejected(self) -> None:
        model = _small_model()
        with pytest.raises(ValueError):
            model(torch.randn(1, 32, 32))
