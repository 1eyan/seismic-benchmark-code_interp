"""Config-loading tests for the Pan2020 PConv U-Net YAML profiles."""

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
from model.interpolation.pan2020_pconv_unet import Pan2020PConvUNet  # noqa: E402
from utils.losses import Pan2020PConvLoss, build_loss  # noqa: E402
from utils.train_utils import build_optimizer, build_scheduler, load_config  # noqa: E402

_CONFIG_DIR = _REPO_ROOT / "configs" / "interpolation"
_AUTHOR = _CONFIG_DIR / "pan2020_pconv_author_code.yaml"
_FINETUNE = _CONFIG_DIR / "pan2020_pconv_finetune.yaml"
_SMOKE = _CONFIG_DIR / "pan2020_pconv_smoke.yaml"
_ABLATION = _CONFIG_DIR / "pan2020_standard_local_pconv_ablation.yaml"


@pytest.fixture(scope="module")
def author_cfg() -> dict:
    return load_config(_AUTHOR)


@pytest.fixture(scope="module")
def finetune_cfg() -> dict:
    return load_config(_FINETUNE)


@pytest.fixture(scope="module")
def smoke_cfg() -> dict:
    return load_config(_SMOKE)


@pytest.fixture(scope="module")
def ablation_cfg() -> dict:
    return load_config(_ABLATION)


class TestConfigsExist:
    @pytest.mark.parametrize("path", [_AUTHOR, _FINETUNE, _SMOKE, _ABLATION])
    def test_config_exists(self, path: Path) -> None:
        assert path.exists()


class TestConfigsLoadAndBuild:
    def test_model_builds(self, author_cfg) -> None:
        model = build_model(author_cfg["model"])
        assert isinstance(model, Pan2020PConvUNet)
        assert len(model.encoder_stages) == 6

    def test_loss_builds(self, author_cfg) -> None:
        loss_fn = build_loss(author_cfg["loss"])
        assert isinstance(loss_fn, Pan2020PConvLoss)
        assert loss_fn.hole_weight == 6.0
        assert loss_fn.tv_weight == 0.1

    def test_optimizer_adam(self, author_cfg) -> None:
        model = Pan2020PConvUNet(encoder_channels=[8, 16, 32, 64, 64, 64],
                                 decoder_channels=[64, 32, 16, 8, 4, 1])
        optimizer = build_optimizer(model, author_cfg["optim"])
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-4)

    def test_no_scheduler(self, author_cfg) -> None:
        model = Pan2020PConvUNet(encoder_channels=[8, 16, 32, 64, 64, 64],
                                 decoder_channels=[64, 32, 16, 8, 4, 1])
        optimizer = build_optimizer(model, author_cfg["optim"])
        scheduler = build_scheduler(optimizer, author_cfg["scheduler"], 100)
        assert scheduler is None

    def test_smoke_model_builds(self, smoke_cfg) -> None:
        model = build_model(smoke_cfg["model"])
        assert isinstance(model, Pan2020PConvUNet)

    def test_ablation_local_mode(self, ablation_cfg) -> None:
        assert ablation_cfg["model"]["params"]["normalization_mode"] == \
            "standard_local_valid_ratio"


class TestPaperHyperparameters:
    def test_batch_size_4(self, author_cfg) -> None:
        assert author_cfg["data"]["loader"]["batch_size"] == 4

    def test_lr_2e_4(self, author_cfg) -> None:
        assert author_cfg["optim"]["params"]["lr"] == pytest.approx(2.0e-4)

    def test_patch_size_128(self, author_cfg) -> None:
        assert author_cfg["preprocess"]["patch_time"] == 128
        assert author_cfg["preprocess"]["patch_trace"] == 128

    def test_encoder_kernels(self, author_cfg) -> None:
        assert author_cfg["model"]["params"]["encoder_kernels"] == [7, 5, 5, 3, 3, 3]

    def test_encoder_channels(self, author_cfg) -> None:
        assert author_cfg["model"]["params"]["encoder_channels"] == \
            [32, 64, 128, 256, 512, 512]

    def test_global_normalization_mode(self, author_cfg) -> None:
        assert author_cfg["model"]["params"]["normalization_mode"] == \
            "author_global_mask_mean"

    def test_minmax_input_normalization(self, author_cfg) -> None:
        assert author_cfg["preprocess"]["normalize_mode"] == "minmax"

    def test_pan2020_random_mask_mode(self, author_cfg) -> None:
        assert author_cfg["preprocess"]["mask_mode"] == "pan2020_random"

    def test_separate_input_mode(self, author_cfg) -> None:
        assert author_cfg["model"]["params"]["input_mode"] == "separate"

    def test_inference_overwrite_observed_false(self, author_cfg) -> None:
        assert author_cfg["inference"]["overwrite_observed"] is False

    def test_finetune_lr(self, finetune_cfg) -> None:
        assert finetune_cfg["optim"]["params"]["lr"] == pytest.approx(5.0e-5)

    def test_finetune_resume(self, finetune_cfg) -> None:
        assert finetune_cfg["train"]["resume"] is not None

    def test_smoke_epochs_2(self, smoke_cfg) -> None:
        assert smoke_cfg["train"]["epochs"] == 2

    def test_smoke_max_shots_1(self, smoke_cfg) -> None:
        assert smoke_cfg["preprocess"]["max_shots"] == 1


class TestPaperAlignmentBlock:
    def test_author_has_paper_alignment(self, author_cfg) -> None:
        pa = author_cfg["paper_alignment"]
        assert "architecture" in pa
        assert "loss" in pa
        assert "training" in pa

    def test_architecture_audit_has_required_keys(self, author_cfg) -> None:
        arch = author_cfg["paper_alignment"]["architecture"]
        for key, entry in arch.items():
            assert "classification" in entry, f"missing classification in {key}"
            assert "value" in entry, f"missing value in {key}"
            assert "evidence" in entry, f"missing evidence in {key}"
            assert entry["classification"] in (
                "paper-explicit", "paper-figure", "paper-inherited",
                "paper-inferred", "reproduction-assumption",
                "repository-adaptation", "unresolved", "author-code",
            )

    def test_ablation_has_paper_alignment(self, ablation_cfg) -> None:
        pa = ablation_cfg["paper_alignment"]
        assert "architecture" in pa
        assert "loss" in pa
