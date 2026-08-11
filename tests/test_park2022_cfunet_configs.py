"""Park2022 CFunet config tests: every config loads, builds, and carries the expected paper values."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "model").is_dir() and (p / "utils").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("Cannot find repo root.")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.interpolation.park2022_cfunet import Park2022CFUNet  # noqa: E402
from model.registry import build_model  # noqa: E402
from utils.losses import CFunetMSEFourierLoss  # noqa: E402
from utils.train_utils import load_config  # noqa: E402

_CFG_DIR = _REPO_ROOT / "configs" / "interpolation"

_CONFIGS = [
    "park2022_cfunet_paper.yaml",
    "park2022_cfunet_field_paper.yaml",
    "park2022_cfunet_eval75.yaml",
    "park2022_cfunet_mse.yaml",
    "park2022_cfunet_upsampling_ablation.yaml",
    "park2022_cfunet_smoke.yaml",
    "park2022_baseline_unet.yaml",
]


@pytest.mark.parametrize("name", _CONFIGS)
def test_config_loads_and_builds(name):
    cfg = load_config(_CFG_DIR / name)
    assert "paper_alignment" in cfg or name.endswith("smoke.yaml")
    model = build_model(cfg["model"])
    if name == "park2022_baseline_unet.yaml":
        assert cfg["model"]["type"] == "unet"
        assert cfg["loss"]["type"] == "mse"
    else:
        assert isinstance(model, Park2022CFunet)
        assert cfg["loss"]["type"] == "cfunet_mse_fourier"
        loss = CFunetMSEFourierLoss(**cfg["loss"]["params"])
        assert hasattr(loss, "attach_model")


class TestPaperConfigValues:
    def _paper(self):
        return load_config(_CFG_DIR / "park2022_cfunet_paper.yaml")

    def test_paper_explicit_hyperparameters(self):
        cfg = self._paper()
        assert cfg["preprocess"]["mask_mode"] == "cfunet_random"
        assert cfg["preprocess"]["mask_ratio_range"] == [0.5, 0.875]
        assert cfg["data"]["loader"]["batch_size"] == 32
        assert cfg["train"]["epochs"] == 20
        assert cfg["optim"]["type"] == "adam"
        assert cfg["optim"]["params"]["lr"] == 5.0e-5
        assert cfg["scheduler"]["type"] == "none"

    def test_paper_model_configuration(self):
        cfg = self._paper()
        m = cfg["model"]["params"]
        assert m["base_channels"] == 22
        assert m["num_levels"] == 4
        assert m["upsample_mode"] == "fourier_zero_padding"
        assert m["upsampler_scale_factor"] == 2

    def test_paper_loss_alpha(self):
        cfg = self._paper()
        assert cfg["loss"]["params"]["alpha"] == 1.0

    def test_patch_size(self):
        cfg = self._paper()
        assert cfg["preprocess"]["patch_time"] == 128
        assert cfg["preprocess"]["patch_trace"] == 128

    def test_paper_alignment_classification_present(self):
        pa = self._paper()["paper_alignment"]
        for key in (
            "fourier_upsampling",
            "final_stride2_conv",
            "mse_loss_coarse_final",
            "fourier_l1_loss",
            "mask_ratio_train",
            "mask_ratio_validation",
        ):
            assert key in pa
            assert pa[key].startswith("paper-explicit")


class TestFieldConfigValues:
    def _field(self):
        return load_config(_CFG_DIR / "park2022_cfunet_field_paper.yaml")

    def test_field_hyperparameters(self):
        cfg = self._field()
        assert cfg["loss"]["params"]["alpha"] == 0.1
        assert cfg["optim"]["params"]["lr"] == 1.0e-5
        assert cfg["preprocess"]["patch_time"] == 128
        assert cfg["preprocess"]["patch_trace"] == 120


class TestAblationConfigValues:
    def test_mse_ablation_alpha_zero(self):
        cfg = load_config(_CFG_DIR / "park2022_cfunet_mse.yaml")
        assert cfg["loss"]["params"]["alpha"] == 0.0

    def test_upsampling_ablation_bilinear(self):
        cfg = load_config(_CFG_DIR / "park2022_cfunet_upsampling_ablation.yaml")
        assert cfg["model"]["params"]["upsample_mode"] == "bilinear"

    def test_eval75_pins_ratio(self):
        cfg = load_config(_CFG_DIR / "park2022_cfunet_eval75.yaml")
        assert cfg["preprocess"]["mask_ratio_range"] == [0.75, 0.75]
