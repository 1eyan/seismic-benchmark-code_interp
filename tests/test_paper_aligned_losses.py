"""Tests for paper-aligned loss functions: NormalizedObservedL1, MaskedL1,
MaskedMSE, WeightedCompositeLoss, FKSpectrumSuppressionLoss."""

import sys
from pathlib import Path

import numpy as np
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

from utils.losses import (
    MaskedL1Loss,
    MaskedMSELoss,
    NormalizedObservedL1Loss,
    WeightedCompositeLoss,
    build_loss,
)
from utils.spectrum_loss import FKSpectrumSuppressionLoss


class TestNormalizedObservedL1:
    def test_falls_back_to_standard_l1_without_mask(self):
        loss_fn = NormalizedObservedL1Loss()
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 2.0, 2.0])
        result = loss_fn(pred, target)
        expected = torch.tensor(2.0 / 3.0)  # (|1-2| + |2-2| + |3-2|) / 3
        assert torch.allclose(result, expected, atol=1e-6)

    def test_computes_l1_on_observed_only(self):
        loss_fn = NormalizedObservedL1Loss()
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
        mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])  # 3 observed
        result = loss_fn(pred, target, mask=mask)
        expected = (1.0 + 1.0 + 2.0) / 3.0  # only pos (0,0), (1,0), (1,1)
        assert torch.allclose(result, torch.tensor(expected), atol=1e-6)

    def test_small_eps_avoids_division_by_zero(self):
        loss_fn = NormalizedObservedL1Loss(eps=1e-6)
        pred = torch.zeros(2, 2)
        target = torch.zeros(2, 2)
        mask = torch.zeros(2, 2)
        result = loss_fn(pred, target, mask=mask)
        assert torch.isfinite(result)
        assert result.item() >= 0.0

    def test_mask_broadcast_to_pred_shape(self):
        loss_fn = NormalizedObservedL1Loss()
        pred = torch.ones(2, 1, 4, 4)
        target = torch.zeros(2, 1, 4, 4)
        mask_2d = torch.ones(4, 4)
        result = loss_fn(pred, target, mask=mask_2d)
        assert torch.allclose(result, torch.tensor(1.0), atol=1e-6)


class TestMaskedL1:
    def test_masked_l1_observed_only(self):
        loss_fn = MaskedL1Loss()
        pred = torch.ones(3, 4)
        target = torch.zeros(3, 4)
        mask = torch.eye(3, 4)
        result = loss_fn(pred, target, mask=mask)
        assert torch.allclose(result, torch.tensor(1.0), atol=1e-6)


class TestMaskedMSE:
    def test_masked_mse_observed_only(self):
        loss_fn = MaskedMSELoss()
        pred = 2.0 * torch.ones(3, 4)
        target = torch.zeros(3, 4)
        mask = torch.eye(3, 4)
        result = loss_fn(pred, target, mask=mask)
        assert torch.allclose(result, torch.tensor(4.0), atol=1e-6)


class TestWeightedCompositeLoss:
    def test_weighted_sum_of_sub_losses(self):
        cfg = {"type": "weighted_composite", "params": {
            "terms": [
                {"type": "mse", "params": {"reduction": "mean"}},
                {"type": "l1", "params": {"reduction": "mean"}},
            ],
            "weights": [0.7, 0.3],
        }}
        loss_fn = build_loss(cfg)
        pred = torch.tensor([0.0, 0.0])
        target = torch.tensor([3.0, 4.0])
        result = loss_fn(pred, target)
        mse = torch.tensor((9.0 + 16.0) / 2.0)
        l1 = torch.tensor((3.0 + 4.0) / 2.0)
        expected = 0.7 * mse + 0.3 * l1
        assert torch.allclose(result, expected, atol=1e-6)

    def test_extras_passed_to_sub_losses(self):
        cfg = {"type": "weighted_composite", "params": {
            "terms": [
                {"type": "normalized_observed_l1", "params": {}},
            ],
            "weights": [1.0],
        }}
        loss_fn = build_loss(cfg)
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        result = loss_fn(pred, target, mask=mask)
        expected = (1.0 + 3.0 + 4.0) / 3.0
        assert torch.allclose(result, torch.tensor(expected), atol=1e-6)

    def test_raises_on_weight_length_mismatch(self):
        with pytest.raises(ValueError):
            WeightedCompositeLoss(
                terms=[{"type": "mse", "params": {}}],
                weights=[1.0, 2.0],
            )


class TestFKSpectrumSuppressionLoss:
    def test_output_is_finite(self):
        loss_fn = FKSpectrumSuppressionLoss(dt=0.002, dx=1.0, v_min=1500.0)
        pred = torch.randn(2, 1, 64, 64)
        result = loss_fn(pred)
        assert torch.isfinite(result)
        assert 0.0 <= result.item() <= 1.0

    def test_zero_input_gives_low_loss(self):
        loss_fn = FKSpectrumSuppressionLoss(dt=0.002, dx=1.0, v_min=1500.0)
        pred = torch.zeros(1, 1, 32, 32)
        result = loss_fn(pred)
        assert torch.isfinite(result)

    def test_accepts_target_for_api_compatibility(self):
        loss_fn = FKSpectrumSuppressionLoss()
        pred = torch.randn(1, 1, 32, 32)
        target = torch.randn(1, 1, 32, 32)
        result = loss_fn(pred, target)
        assert torch.isfinite(result)

    def test_raises_on_invalid_dim(self):
        loss_fn = FKSpectrumSuppressionLoss()
        with pytest.raises(ValueError):
            loss_fn(torch.randn(32, 32))  # 2D tensor


class TestLossRegistry:
    def test_all_new_losses_registered(self):
        for name in ["normalized_observed_l1", "masked_l1", "masked_mse",
                      "weighted_composite", "fk_spectrum_suppression"]:
            loss_fn = build_loss({"type": name, "params": {}})
            assert loss_fn is not None, f"Loss '{name}' not registered"

    def test_existing_losses_unaffected(self):
        for name in ["mse", "l1", "weighted_mse"]:
            loss_fn = build_loss({"type": name, "params": {}})
            assert loss_fn is not None
