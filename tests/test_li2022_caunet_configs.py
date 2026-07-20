"""Config-loading tests for the Li2022 CA-Unet YAML profiles."""

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
from model.interpolation.li2022_caunet import Li2022CAUNet  # noqa: E402
from utils.losses import ANetSSIML1Loss, build_loss  # noqa: E402
from utils.train_utils import build_optimizer, build_scheduler, load_config  # noqa: E402

_CONFIG_DIR = _REPO_ROOT / "configs" / "interpolation"
_SEG_C3 = _CONFIG_DIR / "li2022_caunet_seg_c3_paper.yaml"
_FIELD = _CONFIG_DIR / "li2022_caunet_field_paper.yaml"


@pytest.fixture(scope="module")
def seg_c3_cfg() -> dict:
    return load_config(_SEG_C3)


@pytest.fixture(scope="module")
def field_cfg() -> dict:
    return load_config(_FIELD)


class TestConfigsLoadAndBuild:
    @pytest.mark.parametrize("path", [_SEG_C3, _FIELD])
    def test_config_exists(self, path: Path) -> None:
        assert path.exists()

    def test_models_build(self, seg_c3_cfg, field_cfg) -> None:
        for cfg in (seg_c3_cfg, field_cfg):
            model = build_model(cfg["model"])
            assert isinstance(model, Li2022CAUNet)
            assert model.depth == 3

    def test_loss_builds_via_alias(self, seg_c3_cfg) -> None:
        loss_fn = build_loss(seg_c3_cfg["loss"])
        assert isinstance(loss_fn, ANetSSIML1Loss)
        assert loss_fn.lambda_l1 == 1.0

    def test_optimizer_adam(self, seg_c3_cfg) -> None:
        model = Li2022CAUNet(base_channels=8, depth=2)
        optimizer = build_optimizer(model, seg_c3_cfg["optim"])
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)

    def test_no_scheduler(self, seg_c3_cfg) -> None:
        model = Li2022CAUNet(base_channels=8, depth=2)
        optimizer = build_optimizer(model, seg_c3_cfg["optim"])
        scheduler = build_scheduler(optimizer, seg_c3_cfg["scheduler"], 20)
        assert scheduler is None


class TestPaperHyperparameters:
    def test_synthetic_batch_size_32(self, seg_c3_cfg) -> None:
        assert seg_c3_cfg["data"]["loader"]["batch_size"] == 32

    def test_field_batch_size_8(self, field_cfg) -> None:
        assert field_cfg["data"]["loader"]["batch_size"] == 8

    def test_epochs_20(self, seg_c3_cfg, field_cfg) -> None:
        assert seg_c3_cfg["train"]["epochs"] == 20
        assert field_cfg["train"]["epochs"] == 20

    def test_patch_sizes(self, seg_c3_cfg, field_cfg) -> None:
        assert seg_c3_cfg["preprocess"]["patch_time"] == 128
        assert seg_c3_cfg["preprocess"]["patch_trace"] == 128
        assert field_cfg["preprocess"]["patch_time"] == 720
        assert field_cfg["preprocess"]["patch_trace"] == 120

    def test_mask_ratio_range(self, seg_c3_cfg, field_cfg) -> None:
        for cfg in (seg_c3_cfg, field_cfg):
            assert cfg["preprocess"]["mask_mode"] == "continuous"
            assert cfg["preprocess"]["mask_ratio_range"] == [0.1, 0.3]

    def test_minmax_normalization(self, seg_c3_cfg, field_cfg) -> None:
        for cfg in (seg_c3_cfg, field_cfg):
            assert cfg["preprocess"]["normalize_mode"] == "minmax"

    def test_no_mask_input_channel(self, seg_c3_cfg) -> None:
        assert seg_c3_cfg["model"]["params"]["in_channels"] == 1

    def test_ca_reduction_ratio_in_config(self, seg_c3_cfg) -> None:
        assert seg_c3_cfg["model"]["params"]["ca_reduction_ratio"] == 16

    def test_depth_in_config(self, seg_c3_cfg) -> None:
        assert seg_c3_cfg["model"]["params"]["depth"] == 3

    def test_raw_output_evaluation(self, seg_c3_cfg, field_cfg) -> None:
        for cfg in (seg_c3_cfg, field_cfg):
            assert cfg["inference"]["overwrite_observed"] is False

    def test_loss_uses_ssim_l1_alias(self, seg_c3_cfg) -> None:
        assert seg_c3_cfg["loss"]["type"] == "ssim_l1"

    def test_field_loss_uses_ssim_l1_alias(self, field_cfg) -> None:
        assert field_cfg["loss"]["type"] == "ssim_l1"


class TestPaperAlignmentBlock:
    def test_synthetic_has_paper_alignment(self, seg_c3_cfg) -> None:
        pa = seg_c3_cfg["paper_alignment"]
        assert "architecture" in pa
        assert "loss" in pa
        assert "training" in pa

    def test_architecture_audit_has_required_keys(self, seg_c3_cfg) -> None:
        arch = seg_c3_cfg["paper_alignment"]["architecture"]
        for key, entry in arch.items():
            assert "classification" in entry, f"missing classification in {key}"
            assert "value" in entry, f"missing value in {key}"
            assert "evidence" in entry, f"missing evidence in {key}"
            assert entry["classification"] in (
                "paper-explicit", "paper-figure", "paper-inherited",
                "paper-inferred", "reproduction-assumption",
                "repository-adaptation", "unresolved",
            )

    def test_filed_has_paper_alignment(self, field_cfg) -> None:
        pa = field_cfg["paper_alignment"]
        assert "architecture" in pa
        assert "loss" in pa
        assert "training" in pa
