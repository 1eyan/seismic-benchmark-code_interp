"""Exact-value tests for the ANet hybrid loss ``-SSIM + lambda_l1 * L1``."""

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

from utils.losses import LOSS_REGISTRY, ANetSSIML1Loss, build_loss  # noqa: E402

C1 = 1.0e-4
C2 = 9.0e-4


def _manual_hybrid(pred: torch.Tensor, target: torch.Tensor, lambda_l1: float = 1.0) -> dict:
    """Reference implementation with per-sample global-patch statistics."""
    x = pred.reshape(pred.shape[0], -1).double()
    y = target.reshape(target.shape[0], -1).double()
    mu_x, mu_y = x.mean(dim=1), y.mean(dim=1)
    var_x = ((x - mu_x[:, None]) ** 2).mean(dim=1)
    var_y = ((y - mu_y[:, None]) ** 2).mean(dim=1)
    cov = ((x - mu_x[:, None]) * (y - mu_y[:, None])).mean(dim=1)
    ssim = ((2 * mu_x * mu_y + C1) * (2 * cov + C2)) / (
        (mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2)
    )
    l1 = (x - y).abs().mean(dim=1)
    return {
        "ssim": ssim.mean().item(),
        "l1": l1.mean().item(),
        "total": (-ssim.mean() + lambda_l1 * l1.mean()).item(),
    }


class TestExactValues:
    def test_identical_inputs(self) -> None:
        loss_fn = ANetSSIML1Loss()
        x = torch.rand(3, 1, 16, 16)
        comps = loss_fn.components(x, x.clone())
        torch.testing.assert_close(comps["ssim"], torch.tensor(1.0))
        torch.testing.assert_close(comps["loss_l1"], torch.tensor(0.0))
        torch.testing.assert_close(comps["loss_total"], torch.tensor(-1.0))

    def test_perfect_reconstruction_is_minus_one_not_zero(self) -> None:
        # Paper formula: loss = -SSIM + L1 = -1 at perfect reconstruction.
        loss_fn = ANetSSIML1Loss()
        x = torch.rand(2, 1, 8, 8)
        assert loss_fn(x, x.clone()).item() == pytest.approx(-1.0)

    def test_matches_manual_computation(self) -> None:
        torch.manual_seed(0)
        pred = torch.rand(4, 1, 12, 12)
        target = torch.rand(4, 1, 12, 12)
        loss_fn = ANetSSIML1Loss()
        comps = loss_fn.components(pred, target)
        manual = _manual_hybrid(pred, target)
        assert comps["ssim"].item() == pytest.approx(manual["ssim"], abs=1e-5)
        assert comps["loss_l1"].item() == pytest.approx(manual["l1"], abs=1e-6)
        assert comps["loss_total"].item() == pytest.approx(manual["total"], abs=1e-5)

    def test_constant_offset_vs_one_minus_ssim(self) -> None:
        torch.manual_seed(1)
        pred = torch.rand(2, 1, 8, 8)
        target = torch.rand(2, 1, 8, 8)
        comps = ANetSSIML1Loss().components(pred, target)
        shifted = (1.0 - comps["ssim"]) + comps["loss_l1"]
        assert comps["loss_total"].item() == pytest.approx(shifted.item() - 1.0, abs=1e-6)

    def test_zeros_and_ones_are_finite(self) -> None:
        loss_fn = ANetSSIML1Loss()
        zeros = torch.zeros(2, 1, 8, 8)
        ones = torch.ones(2, 1, 8, 8)
        assert torch.isfinite(loss_fn(zeros, ones))
        assert torch.isfinite(loss_fn(zeros, zeros.clone()))
        assert torch.isfinite(loss_fn(ones, ones.clone()))

    def test_all_zero_inputs_give_ssim_one(self) -> None:
        zeros = torch.zeros(1, 1, 8, 8)
        comps = ANetSSIML1Loss().components(zeros, zeros.clone())
        torch.testing.assert_close(comps["ssim"], torch.tensor(1.0))


class TestLambdaSemantics:
    def test_lambda_weights_l1_term_only(self) -> None:
        torch.manual_seed(2)
        pred = torch.rand(2, 1, 8, 8)
        target = torch.rand(2, 1, 8, 8)
        c1 = ANetSSIML1Loss(lambda_l1=1.0).components(pred, target)
        c2 = ANetSSIML1Loss(lambda_l1=2.0).components(pred, target)
        torch.testing.assert_close(c1["loss_ssim"], c2["loss_ssim"])
        torch.testing.assert_close(c1["loss_l1"], c2["loss_l1"])
        diff = (c2["loss_total"] - c1["loss_total"]).item()
        assert diff == pytest.approx(c1["loss_l1"].item(), abs=1e-6)


class TestFullPatchLoss:
    def test_observed_region_error_changes_loss(self) -> None:
        torch.manual_seed(3)
        target = torch.rand(1, 1, 8, 8)
        pred = target.clone()
        pred[..., :, :4] += 0.3  # "observed" half
        loss_fn = ANetSSIML1Loss()
        assert loss_fn(pred, target).item() != pytest.approx(-1.0)

    def test_missing_region_error_changes_loss(self) -> None:
        torch.manual_seed(4)
        target = torch.rand(1, 1, 8, 8)
        pred = target.clone()
        pred[..., :, 4:] += 0.3  # "missing" half
        loss_fn = ANetSSIML1Loss()
        assert loss_fn(pred, target).item() != pytest.approx(-1.0)

    def test_no_mask_argument_required(self) -> None:
        pred = torch.rand(1, 1, 8, 8)
        target = torch.rand(1, 1, 8, 8)
        # extras (e.g. mask from the trainer) must be accepted and ignored.
        value = ANetSSIML1Loss()(pred, target, mask=torch.ones_like(pred))
        assert torch.isfinite(value)


class TestGradients:
    def test_backward_produces_finite_gradients(self) -> None:
        torch.manual_seed(5)
        pred = torch.rand(2, 1, 8, 8, requires_grad=True)
        target = torch.rand(2, 1, 8, 8)
        ANetSSIML1Loss()(pred, target).backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
        assert pred.grad.abs().sum() > 0


class TestRegistryAndValidation:
    def test_registered(self) -> None:
        assert "anet_ssim_l1" in LOSS_REGISTRY

    def test_factory_build(self) -> None:
        loss_fn = build_loss({"type": "anet_ssim_l1", "params": {"lambda_l1": 1.0}})
        assert isinstance(loss_fn, ANetSSIML1Loss)
        assert loss_fn.lambda_l1 == 1.0

    def test_invalid_ssim_mode_rejected(self) -> None:
        with pytest.raises(ValueError):
            ANetSSIML1Loss(ssim_mode="windowed")

    def test_invalid_constants_rejected(self) -> None:
        with pytest.raises(ValueError):
            ANetSSIML1Loss(c1=0.0)
        with pytest.raises(ValueError):
            ANetSSIML1Loss(c2=-1.0)

    def test_missing_target_rejected(self) -> None:
        with pytest.raises(ValueError):
            ANetSSIML1Loss()(torch.rand(1, 1, 4, 4))
