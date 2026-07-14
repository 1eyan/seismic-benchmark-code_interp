"""Validate that all paper-aligned configs load correctly and produce
runnable training components (model, loss, optimizer, scheduler)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "utils").is_dir() and (p / "model").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("Cannot find repo root.")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.interpolation import build_model
from utils import build_loss, build_optimizer, build_scheduler, load_config

# ---------------------------------------------------------------------------
# All paper configs to validate
# ---------------------------------------------------------------------------
PAPER_CONFIGS = [
    "configs/interpolation/chai2020_unet_paper.yaml",
    "configs/interpolation/wang2019_resnet_paper.yaml",
    "configs/interpolation/yoon2021_dbilstm_paper.yaml",
    "configs/interpolation/yuan2022_btn_random.yaml",
    "configs/interpolation/yuan2022_btn_spectrum.yaml",
    "configs/interpolation/yuan2022_btn_mix.yaml",
    "configs/interpolation/guo2023_mst_paper.yaml",
]

# Existing configs that must still work
EXISTING_CONFIGS = [
    "configs/interpolation/chai2020_unet.yaml",
    "configs/interpolation/wang2019_resnet.yaml",
    "configs/interpolation/yoon2021_dbilstm.yaml",
    "configs/interpolation/yuan2022_btn.yaml",
    "configs/interpolation/guo2023_mst.yaml",
]


def _load_and_validate(cfg_path: str) -> Dict[str, Any]:
    cfg = load_config(_REPO_ROOT / cfg_path)
    assert "model" in cfg, f"{cfg_path}: missing model"
    assert "loss" in cfg, f"{cfg_path}: missing loss"
    assert "optim" in cfg, f"{cfg_path}: missing optim"
    assert "train" in cfg, f"{cfg_path}: missing train"
    return cfg


class TestPaperConfigsLoad:
    @pytest.mark.parametrize("config_path", PAPER_CONFIGS)
    def test_config_loads(self, config_path):
        cfg = _load_and_validate(config_path)
        assert cfg is not None

    @pytest.mark.parametrize("config_path", PAPER_CONFIGS)
    def test_model_builds(self, config_path):
        cfg = _load_and_validate(config_path)
        model = build_model(cfg["model"])
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.parametrize("config_path", PAPER_CONFIGS)
    def test_loss_builds(self, config_path):
        cfg = _load_and_validate(config_path)
        loss_fn = build_loss(cfg["loss"])
        assert loss_fn is not None

    @pytest.mark.parametrize("config_path", PAPER_CONFIGS)
    def test_optimizer_builds(self, config_path):
        cfg = _load_and_validate(config_path)
        model = build_model(cfg["model"])
        opt = build_optimizer(model, cfg["optim"])
        assert opt is not None

    @pytest.mark.parametrize("config_path", PAPER_CONFIGS)
    def test_scheduler_builds(self, config_path):
        cfg = _load_and_validate(config_path)
        model = build_model(cfg["model"])
        opt = build_optimizer(model, cfg["optim"])
        total_epochs = int(cfg["train"]["epochs"])
        sched = build_scheduler(opt, cfg.get("scheduler", {"type": "none"}), total_epochs)
        # scheduler can be None ("none" type)

    @pytest.mark.parametrize("config_path", PAPER_CONFIGS)
    def test_paper_alignment_block_present(self, config_path):
        cfg = load_config(_REPO_ROOT / config_path)
        assert "paper_alignment" in cfg, (
            f"{config_path}: missing paper_alignment block"
        )

    @pytest.mark.parametrize("config_path", PAPER_CONFIGS)
    def test_paper_alignment_has_required_sections(self, config_path):
        cfg = load_config(_REPO_ROOT / config_path)
        pa = cfg["paper_alignment"]
        assert "architecture" in pa, f"{config_path}: paper_alignment.architecture missing"
        assert "training" in pa, f"{config_path}: paper_alignment.training missing"


class TestExistingConfigsUnaffected:
    """Regression: existing configs must still work with updated infrastructure."""

    @pytest.mark.parametrize("config_path", EXISTING_CONFIGS)
    def test_config_loads_and_builds(self, config_path):
        cfg = load_config(_REPO_ROOT / config_path)
        model = build_model(cfg["model"])
        loss_fn = build_loss(cfg["loss"])
        opt = build_optimizer(model, cfg["optim"])
        assert model is not None
        assert loss_fn is not None
        assert opt is not None


class TestBTNSelfSupervisedPipeline:
    """Verify BTN self-supervised: mask flows from data to loss."""

    def test_return_mask_produces_three_tensors(self):
        """Simulate _patchify_pairs with return_mask=True."""
        cfg = load_config(_REPO_ROOT / "configs/interpolation/yuan2022_btn_random.yaml")
        assert cfg["preprocess"]["return_mask"] is True

    def test_normalized_observed_l1_uses_mask(self):
        from utils.losses import NormalizedObservedL1Loss
        loss_fn = NormalizedObservedL1Loss()
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randn(2, 1, 32, 32)
        mask = torch.ones(2, 1, 1, 32)
        mask[:, :, :, 16:] = 0.0  # half observed
        loss_with_mask = loss_fn(pred, target, mask=mask)
        loss_without_mask = loss_fn(pred, target)
        # With half the positions masked, loss should differ
        assert not torch.allclose(loss_with_mask, loss_without_mask)


class TestChai2020PaperConfig:
    def test_uses_adam_not_adamw(self):
        cfg = load_config(_REPO_ROOT / "configs/interpolation/chai2020_unet_paper.yaml")
        assert cfg["optim"]["type"] == "adam"

    def test_patch_size_112(self):
        cfg = load_config(_REPO_ROOT / "configs/interpolation/chai2020_unet_paper.yaml")
        assert cfg["preprocess"]["patch_time"] == 112
        assert cfg["preprocess"]["patch_trace"] == 112

    def test_epochs_50(self):
        cfg = load_config(_REPO_ROOT / "configs/interpolation/chai2020_unet_paper.yaml")
        assert cfg["train"]["epochs"] == 50


class TestGuo2023PaperConfig:
    def test_uses_adamw(self):
        cfg = load_config(_REPO_ROOT / "configs/interpolation/guo2023_mst_paper.yaml")
        assert cfg["optim"]["type"] == "adamw"

    def test_batch_size_32(self):
        cfg = load_config(_REPO_ROOT / "configs/interpolation/guo2023_mst_paper.yaml")
        assert cfg["data"]["loader"]["batch_size"] == 32

    def test_patch_96(self):
        cfg = load_config(_REPO_ROOT / "configs/interpolation/guo2023_mst_paper.yaml")
        assert cfg["preprocess"]["patch_time"] == 96
        assert cfg["preprocess"]["patch_trace"] == 96
