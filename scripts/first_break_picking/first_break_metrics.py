"""Metrics for first-break binary mask segmentation and pick extraction."""

from __future__ import annotations

from collections import OrderedDict
import re
from typing import Any, Dict, List, Optional

import torch

BAD_FIRST_BREAK_PICK_INDEX = -1


class _BinaryMetric:
    higher_is_better = True

    def __init__(self, threshold: float = 0.5, eps: float = 1.0e-6) -> None:
        self.threshold = float(threshold)
        self.eps = float(eps)

    def _masks(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = target.detach() >= 0
        pred_mask = (torch.sigmoid(pred.detach()) >= self.threshold) & valid
        target_mask = (target.detach() >= 0.5) & valid
        return pred_mask, target_mask, valid

    def _pred_picks(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_mask, _, _ = self._masks(pred, target)
        pred_flat = pred_mask[:, 0].reshape(-1, pred_mask.size(-1))
        pred_has = pred_flat.any(dim=1)
        pred_pick = pred_flat.float().argmax(dim=1).to(dtype=torch.long)
        pred_pick = torch.where(
            pred_has,
            pred_pick,
            torch.full_like(pred_pick, BAD_FIRST_BREAK_PICK_INDEX),
        )
        return pred_pick, pred_has

    def _pick_data(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_pick: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if target_pick is None:
            raise ValueError("First-break pick metrics require target_pick.")
        pred_pick, pred_has = self._pred_picks(pred, target)
        target_pick = target_pick.detach().to(device=pred_pick.device).reshape(-1).to(dtype=torch.long)
        if target_pick.numel() != pred_pick.numel():
            raise ValueError(
                "target_pick shape is incompatible with prediction traces: "
                f"{tuple(target_pick.shape)} vs {tuple(pred_pick.shape)}."
            )
        target_valid = target_pick >= 0
        return pred_pick.float(), pred_has, target_pick.float(), target_valid


class DiceMetric(_BinaryMetric):
    """Mean binary Dice score, ignoring target pixels < 0."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_pick: Optional[torch.Tensor] = None,
    ) -> float:
        pred_mask, target_mask, valid = self._masks(pred, target)
        pred_f = pred_mask.float().reshape(pred_mask.size(0), -1)
        target_f = target_mask.float().reshape(target_mask.size(0), -1)
        valid_f = valid.float().reshape(valid.size(0), -1)
        sample_has_valid = valid_f.sum(dim=1) > 0
        if not bool(sample_has_valid.any()):
            return float("nan")
        inter = (pred_f * target_f).sum(dim=1)
        denom = pred_f.sum(dim=1) + target_f.sum(dim=1)
        score = (2.0 * inter + self.eps) / (denom + self.eps)
        return float(score[sample_has_valid].mean().item())


class F1Metric(DiceMetric):
    """Binary F1 score, equivalent to Dice for one foreground class."""


class IoUMetric(_BinaryMetric):
    """Mean binary intersection-over-union, ignoring target pixels < 0."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_pick: Optional[torch.Tensor] = None,
    ) -> float:
        pred_mask, target_mask, valid = self._masks(pred, target)
        pred_f = pred_mask.reshape(pred_mask.size(0), -1)
        target_f = target_mask.reshape(target_mask.size(0), -1)
        valid_f = valid.reshape(valid.size(0), -1)
        sample_has_valid = valid_f.sum(dim=1) > 0
        if not bool(sample_has_valid.any()):
            return float("nan")
        inter = (pred_f & target_f).float().sum(dim=1)
        union = (pred_f | target_f).float().sum(dim=1)
        score = (inter + self.eps) / (union + self.eps)
        return float(score[sample_has_valid].mean().item())


class MeanAbsoluteError(_BinaryMetric):
    """Mean absolute first-break sample error over traces with a target pick."""

    higher_is_better = False

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_pick: Optional[torch.Tensor] = None,
    ) -> float:
        pred_pick, _, target_pick_f, valid = self._pick_data(pred, target, target_pick)
        if not bool(valid.any()):
            return float("nan")
        err = pred_pick[valid] - target_pick_f[valid]
        return float(err.abs().mean().item())


class RootMeanSquaredError(_BinaryMetric):
    """Root mean squared first-break sample error over traces with a target pick."""

    higher_is_better = False

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_pick: Optional[torch.Tensor] = None,
    ) -> float:
        pred_pick, _, target_pick_f, valid = self._pick_data(pred, target, target_pick)
        if not bool(valid.any()):
            return float("nan")
        err = pred_pick[valid] - target_pick_f[valid]
        return float(torch.sqrt((err ** 2).mean()).item())


class MeanBiasError(_BinaryMetric):
    """Mean signed first-break sample error, computed as prediction minus target."""

    higher_is_better = False

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_pick: Optional[torch.Tensor] = None,
    ) -> float:
        pred_pick, _, target_pick_f, valid = self._pick_data(pred, target, target_pick)
        if not bool(valid.any()):
            return float("nan")
        err = pred_pick[valid] - target_pick_f[valid]
        return float(err.mean().item())


class GatherCoverage(_BinaryMetric):
    """Fraction of target-pick traces for which the model produced any pick."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_pick: Optional[torch.Tensor] = None,
    ) -> float:
        _, pred_has, _, valid = self._pick_data(pred, target, target_pick)
        if not bool(valid.any()):
            return float("nan")
        return float(pred_has[valid].float().mean().item())


class PickWithin(_BinaryMetric):
    """Fraction of target-pick traces picked within a sample tolerance."""

    def __init__(
        self,
        threshold: float = 0.5,
        tolerance: int = 5,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__(threshold=threshold, eps=eps)
        self.tolerance = int(tolerance)

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_pick: Optional[torch.Tensor] = None,
    ) -> float:
        pred_pick, pred_has, target_pick_f, valid = self._pick_data(pred, target, target_pick)
        if not bool(valid.any()):
            return float("nan")
        err = (pred_pick[valid] - target_pick_f[valid]).abs()
        hit = pred_has[valid] & (err < self.tolerance)
        return float(hit.float().mean().item())


_METRICS = {
    "dice": DiceMetric,
    "f1": F1Metric,
    "iou": IoUMetric,
    "pick_mae": MeanAbsoluteError,
    "MeanAbsoluteError": MeanAbsoluteError,
    "RootMeanSquaredError": RootMeanSquaredError,
    "MeanBiasError": MeanBiasError,
    "GatherCoverage": GatherCoverage,
    "pick_within": PickWithin,
}


def build_first_break_metrics(cfg_list: List[Dict[str, Any]]) -> "OrderedDict[str, Any]":
    """Instantiate metrics from a list of ``{name, params}`` entries."""
    metrics: "OrderedDict[str, Any]" = OrderedDict()
    for item in cfg_list:
        name = str(item["name"])
        metric_key = name
        params = dict(item.get("params") or {})
        if name.startswith("pick_within_"):
            metric_key = "pick_within"
            params.setdefault("tolerance", int(name.rsplit("_", 1)[-1]))
        hit_rate_match = re.fullmatch(r"HitRate(\d+)px", name)
        if hit_rate_match is not None:
            metric_key = "pick_within"
            params.setdefault("tolerance", int(hit_rate_match.group(1)))
        if metric_key not in _METRICS:
            raise ValueError(f"Unknown first-break metric {name!r}.")
        metrics[name] = _METRICS[metric_key](**params)
    return metrics
