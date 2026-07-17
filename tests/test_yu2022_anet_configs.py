"""Config-loading tests for the three ANet YAML profiles."""

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

from model.interpolation import build_model  # noqa: E402
from model.interpolation.yu2022_anet import Yu2022ANet  # noqa: E402
from utils.losses import ANetSSIML1Loss, MSELoss, build_loss  # noqa: E402
from utils.train_utils import build_optimizer, build_scheduler, load_config  # noqa: E402

_CONFIG_DIR = _REPO_ROOT / "configs" / "interpolation"
_SEG_C3 = _CONFIG_DIR / "yu2022_anet_seg_c3_paper.yaml"
_FIELD = _CONFIG_DIR / "yu2022_anet_field_paper.yaml"
_MSE = _CONFIG_DIR / "yu2022_anet_mse_ablation.yaml"


@pytest.fixture(scope="module")
def seg_c3_cfg() -> dict:
    return load_config(_SEG_C3)


@pytest.fixture(scope="module")
def field_cfg() -> dict:
    return load_config(_FIELD)


@pytest.fixture(scope="module")
def mse_cfg() -> dict:
    return load_config(_MSE)


class TestConfigsLoadAndBuild:
    @pytest.mark.parametrize("path", [_SEG_C3, _FIELD, _MSE])
    def test_config_exists(self, path: Path) -> None:
        assert path.exists()

    def test_models_build(self, seg_c3_cfg, field_cfg, mse_cfg) -> None:
        for cfg in (seg_c3_cfg, field_cfg, mse_cfg):
            model = build_model(cfg["model"])
            assert isinstance(model, Yu2022ANet)
            assert len(model.residual_blocks) == 6
            assert len(model.down_stages) == 2

    def test_paper_loss_builds(self, seg_c3_cfg) -> None:
        loss_fn = build_loss(seg_c3_cfg["loss"])
        assert isinstance(loss_fn, ANetSSIML1Loss)
        assert loss_fn.lambda_l1 == 1.0

    def test_mse_ablation_uses_mse(self, mse_cfg) -> None:
        loss_fn = build_loss(mse_cfg["loss"])
        assert isinstance(loss_fn, MSELoss)

    def test_optimizer_paper_settings(self, seg_c3_cfg) -> None:
        model = Yu2022ANet(base_channels=8, num_residual_blocks=2)
        optimizer = build_optimizer(model, seg_c3_cfg["optim"])
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)

    def test_no_scheduler(self, seg_c3_cfg) -> None:
        model = Yu2022ANet(base_channels=8, num_residual_blocks=2)
        optimizer = build_optimizer(model, seg_c3_cfg["optim"])
        scheduler = build_scheduler(optimizer, seg_c3_cfg["scheduler"], 20)
        assert scheduler is None


class TestPaperHyperparameters:
    def test_synthetic_batch_size_32(self, seg_c3_cfg) -> None:
        assert seg_c3_cfg["data"]["loader"]["batch_size"] == 32

    def test_field_batch_size_8(self, field_cfg) -> None:
        assert field_cfg["data"]["loader"]["batch_size"] == 8

    def test_epochs_20(self, seg_c3_cfg, field_cfg, mse_cfg) -> None:
        for cfg in (seg_c3_cfg, field_cfg, mse_cfg):
            assert cfg["train"]["epochs"] == 20

    def test_patch_sizes(self, seg_c3_cfg, field_cfg) -> None:
        assert seg_c3_cfg["preprocess"]["patch_time"] == 128
        assert seg_c3_cfg["preprocess"]["patch_trace"] == 128
        assert field_cfg["preprocess"]["patch_time"] == 720
        assert field_cfg["preprocess"]["patch_trace"] == 120

    def test_mask_ratio_range(self, seg_c3_cfg, field_cfg, mse_cfg) -> None:
        for cfg in (seg_c3_cfg, field_cfg, mse_cfg):
            assert cfg["preprocess"]["mask_mode"] == "continuous"
            assert cfg["preprocess"]["mask_ratio_range"] == [0.1, 0.3]

    def test_minmax_normalization(self, seg_c3_cfg, field_cfg, mse_cfg) -> None:
        for cfg in (seg_c3_cfg, field_cfg, mse_cfg):
            assert cfg["preprocess"]["normalize_mode"] == "minmax"

    def test_no_mask_input_channel(self, seg_c3_cfg) -> None:
        assert seg_c3_cfg["model"]["params"]["in_channels"] == 1

    def test_ablation_model_matches_paper_model(self, seg_c3_cfg, mse_cfg) -> None:
        assert mse_cfg["model"]["params"] == seg_c3_cfg["model"]["params"]
        assert mse_cfg["optim"] == seg_c3_cfg["optim"]
        assert mse_cfg["train"]["epochs"] == seg_c3_cfg["train"]["epochs"]
        assert (
            mse_cfg["data"]["loader"]["batch_size"]
            == seg_c3_cfg["data"]["loader"]["batch_size"]
        )

    def test_raw_output_evaluation(self, seg_c3_cfg, field_cfg, mse_cfg) -> None:
        for cfg in (seg_c3_cfg, field_cfg, mse_cfg):
            assert cfg["inference"]["overwrite_observed"] is False
