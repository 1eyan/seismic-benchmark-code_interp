"""Loss tests for WRDLSSIMHuberLoss: Huber piecewise, SSIM correctness, hybrid weights."""

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

from utils.losses import WRDLSSIMHuberLoss, build_loss  # noqa: E402


class TestHuber:
    def test_small_error_quadratic(self):
        loss_fn = WRDLSSIMHuberLoss(ssim_weight=0.0, huber_weight=1.0, huber_delta=1.0)
        pred = torch.tensor([0.5])
        target = torch.tensor([0.0])
        # |e| = 0.5 < delta=1.0 → quadratic: 0.5 * 0.5^2 = 0.125
        expected = torch.tensor(0.125)
        out = loss_fn(pred.view(1, 1, 1, 1), target.view(1, 1, 1, 1))
        assert torch.allclose(out, expected, atol=1e-6)

    def test_large_error_linear(self):
        loss_fn = WRDLSSIMHuberLoss(ssim_weight=0.0, huber_weight=1.0, huber_delta=1.0)
        pred = torch.tensor([2.0])
        target = torch.tensor([0.0])
        # |e| = 2.0 > delta=1.0 → linear: delta*(|e| - 0.5*delta) = 1.0*(2.0-0.5)=1.5
        expected = torch.tensor(1.5)
        out = loss_fn(pred.view(1, 1, 1, 1), target.view(1, 1, 1, 1))
        assert torch.allclose(out, expected, atol=1e-6)

    def test_at_delta_transition(self):
        loss_fn = WRDLSSIMHuberLoss(ssim_weight=0.0, huber_weight=1.0, huber_delta=1.0)
        pred = torch.tensor([1.0])
        target = torch.tensor([0.0])
        # |e| = 1.0 = delta → quadratic: 0.5 * 1^2 = 0.5
        # And linear: 1.0*(1.0 - 0.5) = 0.5
        # Both give 0.5 at the transition
        expected = torch.tensor(0.5)
        out = loss_fn(pred.view(1, 1, 1, 1), target.view(1, 1, 1, 1))
        assert torch.allclose(out, expected, atol=1e-6)


class TestSSIM:
    def test_perfect_match(self):
        loss_fn = WRDLSSIMHuberLoss(ssim_weight=1.0, huber_weight=0.0)
        x = torch.ones(1, 1, 32, 32)
        # SSIM = 1, loss_ssim = 0
        out = loss_fn(x, x)
        assert out.item() == pytest.approx(0.0, abs=1e-4)

    def test_constant_input_no_nan(self):
        loss_fn = WRDLSSIMHuberLoss(ssim_weight=1.0, huber_weight=0.0)
        x = torch.full((1, 1, 32, 32), 0.5)
        y = torch.full((1, 1, 32, 32), -0.5)
        out = loss_fn(x, y)
        assert torch.isfinite(out).all()

    def test_ssim_value_between_0_and_1(self):
        loss_fn = WRDLSSIMHuberLoss(ssim_weight=1.0, huber_weight=0.0)
        x = torch.rand(2, 1, 32, 32)
        y = torch.rand(2, 1, 32, 32)
        comps = loss_fn.components(x, y)
        assert 0.0 <= comps["ssim"].item() <= 1.0


class TestHybridWeights:
    def test_ssim_weight_only_affects_ssim_term(self):
        loss1 = WRDLSSIMHuberLoss(ssim_weight=1.0, huber_weight=1.0)
        loss2 = WRDLSSIMHuberLoss(ssim_weight=2.0, huber_weight=1.0)
        x = torch.rand(2, 1, 32, 32)
        y = torch.rand(2, 1, 32, 32)
        c1 = loss1.components(x, y)
        c2 = loss2.components(x, y)
        # SSIM term should double
        assert c2["loss_ssim"].item() == pytest.approx(2.0 * c1["loss_ssim"].item())
        # Huber term unchanged
        assert c2["loss_huber"].item() == pytest.approx(c1["loss_huber"].item())

    def test_huber_weight_only_affects_huber_term(self):
        loss1 = WRDLSSIMHuberLoss(ssim_weight=1.0, huber_weight=1.0)
        loss2 = WRDLSSIMHuberLoss(ssim_weight=1.0, huber_weight=3.0)
        x = torch.rand(2, 1, 32, 32)
        y = torch.rand(2, 1, 32, 32)
        c1 = loss1.components(x, y)
        c2 = loss2.components(x, y)
        assert c2["loss_huber"].item() == pytest.approx(3.0 * c1["loss_huber"].item())
        assert c2["loss_ssim"].item() == pytest.approx(c1["loss_ssim"].item())

    def test_component_sum_matches_total(self):
        loss_fn = WRDLSSIMHuberLoss(ssim_weight=2.0, huber_weight=3.0)
        x = torch.rand(2, 1, 32, 32)
        y = torch.rand(2, 1, 32, 32)
        c = loss_fn.components(x, y)
        expected = 2.0 * c["loss_ssim"] + 3.0 * c["loss_huber"]
        # total already has weights applied
        assert c["loss_total"].item() == pytest.approx(expected.item())


class TestFactory:
    def test_build_from_config(self):
        loss = build_loss({
            "type": "wrdl_ssim_huber",
            "params": {"ssim_weight": 1.0, "huber_weight": 1.0, "huber_delta": 1.0},
        })
        assert isinstance(loss, WRDLSSIMHuberLoss)
        assert loss.huber_delta == 1.0

    def test_registered_name(self):
        loss = build_loss({"type": "wrdl_ssim_huber", "params": {}})
        assert isinstance(loss, WRDLSSIMHuberLoss)


class TestValidation:
    def test_negative_ssim_weight_raises(self):
        with pytest.raises(ValueError):
            WRDLSSIMHuberLoss(ssim_weight=-1.0)

    def test_negative_huber_weight_raises(self):
        with pytest.raises(ValueError):
            WRDLSSIMHuberLoss(huber_weight=-1.0)

    def test_zero_delta_raises(self):
        with pytest.raises(ValueError):
            WRDLSSIMHuberLoss(huber_delta=0.0)

    def test_even_window_raises(self):
        with pytest.raises(ValueError):
            WRDLSSIMHuberLoss(window_size=10)

    def test_missing_target_raises(self):
        loss = WRDLSSIMHuberLoss()
        with pytest.raises(ValueError):
            loss(torch.rand(1, 1, 32, 32))
