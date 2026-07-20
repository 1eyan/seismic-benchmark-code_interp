"""Loss alias and integration tests for the CA-Unet SSIM+L1 loss."""

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

from utils.losses import ANetSSIML1Loss, LOSS_REGISTRY, build_loss  # noqa: E402


class TestAlias:
    def test_ssim_l1_is_anet_ssim_l1(self) -> None:
        assert "ssim_l1" in LOSS_REGISTRY
        assert "anet_ssim_l1" in LOSS_REGISTRY
        assert LOSS_REGISTRY["ssim_l1"] is LOSS_REGISTRY["anet_ssim_l1"]

    def test_factory_build_via_ssim_l1(self) -> None:
        loss_fn = build_loss({"type": "ssim_l1", "params": {"lambda_l1": 1.0}})
        assert isinstance(loss_fn, ANetSSIML1Loss)
        assert loss_fn.lambda_l1 == 1.0

    def test_factory_build_via_anet_ssim_l1_still_works(self) -> None:
        loss_fn = build_loss({"type": "anet_ssim_l1", "params": {"lambda_l1": 2.0}})
        assert isinstance(loss_fn, ANetSSIML1Loss)
        assert loss_fn.lambda_l1 == 2.0

    def test_alias_and_original_produce_identical_results(self) -> None:
        torch.manual_seed(0)
        pred = torch.rand(2, 1, 8, 8)
        target = torch.rand(2, 1, 8, 8)
        loss1 = build_loss({"type": "ssim_l1", "params": {}})
        loss2 = build_loss({"type": "anet_ssim_l1", "params": {}})
        torch.testing.assert_close(loss1(pred, target), loss2(pred, target))


class TestExactValues:
    def test_identical_inputs_loss_is_minus_one(self) -> None:
        loss_fn = build_loss({"type": "ssim_l1", "params": {}})
        x = torch.rand(2, 1, 8, 8)
        assert loss_fn(x, x.clone()).item() == pytest.approx(-1.0)

    def test_perfect_reconstruction_is_minus_one(self) -> None:
        loss_fn = build_loss({"type": "ssim_l1", "params": {}})
        x = torch.ones(2, 1, 8, 8)
        assert loss_fn(x, x.clone()).item() == pytest.approx(-1.0)

    def test_loss_increases_with_error(self) -> None:
        torch.manual_seed(1)
        target = torch.rand(1, 1, 8, 8)
        pred = target.clone()
        pred += 0.5
        loss_fn = build_loss({"type": "ssim_l1", "params": {}})
        assert loss_fn(pred, target).item() > -1.0


class TestComponents:
    def test_components_via_alias(self) -> None:
        loss_fn = build_loss({"type": "ssim_l1", "params": {}})
        x = torch.rand(2, 1, 8, 8)
        comps = loss_fn.components(x, x.clone())
        torch.testing.assert_close(comps["ssim"], torch.tensor(1.0))
        torch.testing.assert_close(comps["loss_l1"], torch.tensor(0.0))
        torch.testing.assert_close(comps["loss_total"], torch.tensor(-1.0))
