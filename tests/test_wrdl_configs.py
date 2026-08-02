"""Config tests for Liu2022WRDL: all 6 configs load, model/loss/optimizer build from config."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

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
from utils.losses import build_loss  # noqa: E402
from utils.optimizers import build_optimizer  # noqa: E402

_CONFIG_DIR = _REPO_ROOT / "configs" / "interpolation"

WRDL_CONFIGS = [
    "liu2022_wrdl_conservative.yaml",
    "liu2022_wrdl_smoke.yaml",
    "liu2022_wrdl_pooling_ablation.yaml",
    "liu2022_wrdl_no_residual_ablation.yaml",
    "liu2022_wrdl_ssim_ablation.yaml",
    "liu2022_wrdl_huber_ablation.yaml",
]


class TestConfigsExist:
    @pytest.mark.parametrize("name", WRDL_CONFIGS)
    def test_config_file_exists(self, name):
        path = _CONFIG_DIR / name
        assert path.exists(), f"{name} not found at {path}"


class TestConfigsLoad:
    @pytest.mark.parametrize("name", WRDL_CONFIGS)
    def test_config_is_valid_yaml(self, name):
        path = _CONFIG_DIR / name
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert cfg is not None
        assert "model" in cfg
        assert "loss" in cfg


class TestBuildFromConfigs:
    @pytest.mark.parametrize("name", [
        n for n in WRDL_CONFIGS if "pooling" not in n  # pooling ablation uses UNet
    ])
    def test_model_builds_from_config(self, name):
        path = _CONFIG_DIR / name
        with open(path) as f:
            cfg = yaml.safe_load(f)
        model = build_model(cfg["model"])
        assert model is not None

    @pytest.mark.parametrize("name", WRDL_CONFIGS)
    def test_loss_builds_from_config(self, name):
        path = _CONFIG_DIR / name
        with open(path) as f:
            cfg = yaml.safe_load(f)
        loss = build_loss(cfg["loss"])
        assert loss is not None


class TestConservativeConfig:
    def test_paper_alignment_block(self):
        path = _CONFIG_DIR / "liu2022_wrdl_conservative.yaml"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        pa = cfg.get("paper_alignment")
        assert pa is not None, "conservative config must have paper_alignment block"
        assert pa["model_name"] == "paper-explicit (WRDL)"
        assert pa["backbone"] == "paper-explicit (U-Net)"
        assert pa["dwt_replaces_pooling"] == "paper-explicit"
        assert pa["ssim_huber_loss"] == "paper-explicit"

    def test_encoder_channels(self):
        path = _CONFIG_DIR / "liu2022_wrdl_conservative.yaml"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        channels = cfg["model"]["params"]["encoder_channels"]
        assert channels == [32, 64, 128, 256, 512]

    def test_loss_type(self):
        path = _CONFIG_DIR / "liu2022_wrdl_conservative.yaml"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["loss"]["type"] == "wrdl_ssim_huber"


class TestSmokeConfig:
    def test_smoke_is_2_epochs(self):
        path = _CONFIG_DIR / "liu2022_wrdl_smoke.yaml"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["train"]["epochs"] == 2

    def test_smoke_is_tiny_model(self):
        path = _CONFIG_DIR / "liu2022_wrdl_smoke.yaml"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["model"]["params"]["encoder_channels"] == [8, 16]


class TestAblationConfigs:
    def test_no_residual_has_zero_blocks(self):
        path = _CONFIG_DIR / "liu2022_wrdl_no_residual_ablation.yaml"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["model"]["params"]["bottleneck_blocks"] == 0

    def test_ssim_ablation_huber_weight_zero(self):
        path = _CONFIG_DIR / "liu2022_wrdl_ssim_ablation.yaml"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["loss"]["params"]["huber_weight"] == 0.0

    def test_huber_ablation_ssim_weight_zero(self):
        path = _CONFIG_DIR / "liu2022_wrdl_huber_ablation.yaml"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["loss"]["params"]["ssim_weight"] == 0.0

    def test_pooling_ablation_uses_unet(self):
        path = _CONFIG_DIR / "liu2022_wrdl_pooling_ablation.yaml"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["model"]["type"] == "unet"
