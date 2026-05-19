"""Losses for first-break binary mask segmentation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """Weighted sum of masked BCE-with-logits and soft Dice loss."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("bce_weight and dice_weight must be non-negative.")
        if bce_weight == 0 and dice_weight == 0:
            raise ValueError("At least one of bce_weight or dice_weight must be > 0.")
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target >= 0
        if not bool(valid.any()):
            return pred.sum() * 0.0

        target = target.to(dtype=pred.dtype)
        valid_f = valid.to(dtype=pred.dtype)
        target_valid = target.clamp_min(0.0)
        loss = pred.new_tensor(0.0)
        if self.bce_weight:
            pos_weight = self.pos_weight
            if pos_weight is not None:
                pos_weight = pos_weight.to(device=pred.device, dtype=pred.dtype)
            bce = F.binary_cross_entropy_with_logits(
                pred[valid],
                target_valid[valid],
                pos_weight=pos_weight,
                reduction="mean",
            )
            loss = loss + self.bce_weight * bce
        if self.dice_weight:
            prob = torch.sigmoid(pred) * valid_f
            target_valid = target_valid * valid_f
            dims = tuple(range(1, prob.dim()))
            intersection = (prob * target_valid).sum(dim=dims)
            denom = prob.sum(dim=dims) + target_valid.sum(dim=dims)
            dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
            sample_has_valid = valid_f.sum(dim=dims) > 0
            if bool(sample_has_valid.any()):
                loss = loss + self.dice_weight * (1.0 - dice[sample_has_valid].mean())
        return loss


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCE-with-logits that ignores pixels where target < 0."""

    def __init__(
        self,
        pos_weight: Optional[float] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Masked BCE supports reduction 'mean' or 'sum', got {reduction!r}.")
        self.reduction = reduction
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target >= 0
        if not bool(valid.any()):
            return pred.sum() * 0.0
        target_valid = target.to(dtype=pred.dtype).clamp_min(0.0)
        pos_weight = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(device=pred.device, dtype=pred.dtype)
        return F.binary_cross_entropy_with_logits(
            pred[valid],
            target_valid[valid],
            pos_weight=pos_weight,
            reduction=self.reduction,
        )


def build_first_break_loss(cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate a first-break loss from a ``{type, params}`` config block."""
    name = str(cfg.get("type", "bce_dice")).lower()
    params = dict(cfg.get("params") or {})
    if name == "bce_dice":
        return BCEDiceLoss(**params)
    if name in ("bce", "bce_with_logits"):
        return MaskedBCEWithLogitsLoss(**params)
    raise ValueError(f"Unknown first-break loss type: {name!r}.")
