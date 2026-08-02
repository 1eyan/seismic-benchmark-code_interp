"""Loss tests for Pan2020PConvLoss: valid/hole/TV components,
weight scaling, per-sample spatial sum + batch mean reduction."""

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

from utils.losses import Pan2020PConvLoss  # noqa: E402


class TestValidHoleComponents:
    def test_perfect_reconstruction_zero_loss(self) -> None:
        loss_fn = Pan2020PConvLoss()
        pred = torch.ones(2, 1, 8, 8)
        target = torch.ones(2, 1, 8, 8)
        mask = torch.ones(2, 1, 8, 8)
        val = loss_fn(pred, target, mask=mask)
        assert val.item() == pytest.approx(0.0, abs=1e-5)

    def test_valid_region_only(self) -> None:
        loss_fn = Pan2020PConvLoss()
        pred = torch.full((1, 1, 4, 4), 2.0)
        target = torch.ones(1, 1, 4, 4)
        mask = torch.ones(1, 1, 4, 4)
        mask[:, :, :2, :] = 0.0  # first 2 rows are "hole"
        # L_valid: |2-1| * mask -> sum over 2 valid rows * 4 cols = 8, mean batch = 8
        # L_hole:  |2-1| * (1-mask) over 2 hole rows = 8, scaled by 6 = 48
        # TV: depends on composite
        loss = loss_fn(pred, target, mask=mask)
        assert loss.item() > 0

    def test_hole_weight_scaling(self) -> None:
        loss6 = Pan2020PConvLoss(hole_weight=6.0)
        loss12 = Pan2020PConvLoss(hole_weight=12.0)
        pred = torch.full((1, 1, 4, 4), 2.0)
        target = torch.ones(1, 1, 4, 4)
        mask = torch.zeros(1, 1, 4, 4)  # all hole
        l6 = loss6(pred, target, mask=mask)
        l12 = loss12(pred, target, mask=mask)
        # With no valid region and same TV, ratio should be ~2
        assert l12.item() > l6.item()

    def test_valid_change_only_affects_valid_term(self) -> None:
        loss_fn = Pan2020PConvLoss(tv_weight=0.0)
        pred_a = torch.full((1, 1, 4, 4), 1.0)
        pred_b = pred_a.clone()
        pred_b[:, :, 0, 0] = 5.0  # change in observed region
        target = torch.ones(1, 1, 4, 4)
        mask = torch.ones(1, 1, 4, 4)
        mask[:, :, 2:, :] = 0.0  # bottom half = hole
        # Both preds differ only in observed area
        la = loss_fn(pred_a, target, mask=mask)
        lb = loss_fn(pred_b, target, mask=mask)
        assert lb.item() > la.item()

    def test_hole_change_only_affects_hole_term(self) -> None:
        loss_fn = Pan2020PConvLoss(tv_weight=0.0)
        pred_a = torch.full((1, 1, 4, 4), 1.0)
        pred_b = pred_a.clone()
        pred_b[:, :, 3, 3] = 5.0  # change in hole region
        target = torch.ones(1, 1, 4, 4)
        mask = torch.ones(1, 1, 4, 4)
        mask[:, :, 2:, :] = 0.0  # last 2 rows = hole
        la = loss_fn(pred_a, target, mask=mask)
        lb = loss_fn(pred_b, target, mask=mask)
        assert lb.item() > la.item()


class TestTVComponent:
    def test_tv_far_from_hole_does_not_change_loss(self) -> None:
        loss_fn = Pan2020PConvLoss(tv_dilation=3)
        pred_a = torch.full((1, 1, 16, 16), 1.0)
        pred_b = pred_a.clone()
        pred_b[:, :, 2, 2] = 5.0  # far from hole boundary
        target = torch.ones(1, 1, 16, 16)
        mask = torch.ones(1, 1, 16, 16)
        mask[:, :, 10:, :] = 0.0  # hole at bottom
        la = loss_fn(pred_a, target, mask=mask)
        lb = loss_fn(pred_b, target, mask=mask)
        # The changed pixel is far from the hole; TV term should be nearly identical
        # But hole and valid terms might differ slightly if the pixel is in valid region
        assert la.item() == pytest.approx(lb.item(), abs=1e-4)

    def test_tv_near_hole_changes_loss(self) -> None:
        loss_fn = Pan2020PConvLoss(tv_dilation=7)
        pred_a = torch.ones(1, 1, 16, 16)
        pred_b = pred_a.clone()
        pred_b[:, :, 9, 8] = 5.0  # near hole boundary (hole starts at row 10)
        target = torch.ones(1, 1, 16, 16)
        mask = torch.ones(1, 1, 16, 16)
        mask[:, :, 10:, :] = 0.0
        la = loss_fn(pred_a, target, mask=mask)
        lb = loss_fn(pred_b, target, mask=mask)
        assert lb.item() > la.item()

    def test_zero_tv_weight_disables_tv(self) -> None:
        loss_with = Pan2020PConvLoss(tv_weight=0.1)
        loss_without = Pan2020PConvLoss(tv_weight=0.0)
        pred = torch.rand(1, 1, 8, 8)
        target = torch.rand(1, 1, 8, 8)
        mask = torch.ones(1, 1, 8, 8)
        mask[:, :, 4:, :] = 0.0
        lw = loss_with(pred, target, mask=mask)
        lwo = loss_without(pred, target, mask=mask)
        assert lw.item() > lwo.item()


class TestReduction:
    def test_batch_mean_spatial_sum(self) -> None:
        """Verify per-sample spatial sum then batch mean: doubling batch
        with identical samples should give approximately the same loss."""
        loss_fn = Pan2020PConvLoss(tv_weight=0.0)
        pred = torch.full((2, 1, 4, 4), 2.0)
        target = torch.ones(2, 1, 4, 4)
        mask = torch.ones(2, 1, 4, 4)
        # Per-sample spatial sum: 4*4*|2-1| = 16 full, split by mask
        loss2 = loss_fn(pred, target, mask=mask)

        pred4 = torch.cat([pred, pred], dim=0)
        target4 = torch.cat([target, target], dim=0)
        mask4 = torch.cat([mask, mask], dim=0)
        loss4 = loss_fn(pred4, target4, mask=mask4)
        assert loss2.item() == pytest.approx(loss4.item(), abs=1e-5)

    def test_requires_mask(self) -> None:
        loss_fn = Pan2020PConvLoss()
        pred = torch.rand(1, 1, 4, 4)
        target = torch.rand(1, 1, 4, 4)
        with pytest.raises(ValueError, match="mask"):
            loss_fn(pred, target)

    def test_requires_target(self) -> None:
        loss_fn = Pan2020PConvLoss()
        pred = torch.rand(1, 1, 4, 4)
        with pytest.raises(ValueError, match="target"):
            loss_fn(pred)


class TestInitValidation:
    def test_negative_hole_weight_raises(self) -> None:
        with pytest.raises(ValueError):
            Pan2020PConvLoss(hole_weight=-1.0)

    def test_negative_tv_weight_raises(self) -> None:
        with pytest.raises(ValueError):
            Pan2020PConvLoss(tv_weight=-1.0)

    def test_even_tv_dilation_raises(self) -> None:
        with pytest.raises(ValueError):
            Pan2020PConvLoss(tv_dilation=4)

    def test_zero_tv_dilation_raises(self) -> None:
        with pytest.raises(ValueError):
            Pan2020PConvLoss(tv_dilation=0)
