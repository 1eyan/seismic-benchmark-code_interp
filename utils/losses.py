"""Loss registry + factory with basic supervised reconstruction losses.

See ``utils/README.md`` for the registry workflow (how to add a new loss).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

import torch
import torch.nn as nn

LOSS_REGISTRY: Dict[str, Type["BaseLoss"]] = {}


def register_loss(name: str) -> Callable[[Type["BaseLoss"]], Type["BaseLoss"]]:
    """Class decorator that registers a loss under ``name``."""

    def _decorator(cls: Type["BaseLoss"]) -> Type["BaseLoss"]:
        if name in LOSS_REGISTRY:
            raise KeyError(f"Loss '{name}' already registered.")
        LOSS_REGISTRY[name] = cls
        return cls

    return _decorator


class BaseLoss(nn.Module):
    """Common loss interface: ``forward(pred, target=None, **extras) -> Tensor``; ``extras`` passes optional mask / weight."""

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        raise NotImplementedError


@register_loss("mse")
class MSELoss(BaseLoss):
    """Mean squared error."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("MSELoss requires `target`.")
        return nn.functional.mse_loss(pred, target, reduction=self.reduction)


@register_loss("l1")
class L1Loss(BaseLoss):
    """Mean absolute error."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("L1Loss requires `target`.")
        return nn.functional.l1_loss(pred, target, reduction=self.reduction)


@register_loss("weighted_mse")
class WeightedMSELoss(BaseLoss):
    """MSE weighted by ``extras["weight"]``."""

    def __init__(self, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("WeightedMSELoss requires `target`.")
        weight = extras.get("weight")
        if weight is None:
            raise ValueError("WeightedMSELoss requires extras['weight'].")
        if not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight, device=pred.device, dtype=pred.dtype)
        weight = weight.to(device=pred.device, dtype=pred.dtype)
        err2 = (pred - target).pow(2)
        weighted = weight * err2
        denom = weight.sum().clamp_min(self.eps)
        return weighted.sum() / denom


# ----------------------------------------------------------------------
# First-break binary mask segmentation losses
# ----------------------------------------------------------------------


@register_loss("bce_dice")
class BCEDiceLoss(BaseLoss):
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

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("BCEDiceLoss requires `target`.")
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
            bce = nn.functional.binary_cross_entropy_with_logits(
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


@register_loss("masked_bce")
class MaskedBCEWithLogitsLoss(BaseLoss):
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

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("MaskedBCEWithLogitsLoss requires `target`.")
        valid = target >= 0
        if not bool(valid.any()):
            return pred.sum() * 0.0
        target_valid = target.to(dtype=pred.dtype).clamp_min(0.0)
        pos_weight = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(device=pred.device, dtype=pred.dtype)
        return nn.functional.binary_cross_entropy_with_logits(
            pred[valid],
            target_valid[valid],
            pos_weight=pos_weight,
            reduction=self.reduction,
        )


# ----------------------------------------------------------------------
# Self-supervised / mask-aware regression losses
# ----------------------------------------------------------------------


@register_loss("normalized_observed_l1")
class NormalizedObservedL1Loss(BaseLoss):
    """L1 loss normalised by the count of observed (non-masked) positions.

    For BTN self-supervised training: the model sees a masked input and
    predicts all traces; loss is evaluated only on *observed* positions
    to enforce the blind-trace constraint.  ``extras["mask"]`` must be a
    float tensor of the same shape as ``pred`` with 1 = observed, 0 = missing.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("NormalizedObservedL1Loss requires `target`.")
        mask = extras.get("mask")
        if mask is None:
            return nn.functional.l1_loss(pred, target, reduction="mean")
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, device=pred.device, dtype=pred.dtype)
        if mask.shape != pred.shape:
            mask = mask.reshape(pred.shape)
        mask = mask.to(device=pred.device, dtype=pred.dtype)
        abs_err = (pred - target).abs() * mask
        count = mask.sum().clamp_min(self.eps)
        return abs_err.sum() / count


@register_loss("masked_l1")
class MaskedL1Loss(BaseLoss):
    """L1 loss computed only on positions where ``extras["mask"] == 1``."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("MaskedL1Loss requires `target`.")
        mask = extras.get("mask")
        if mask is None:
            return nn.functional.l1_loss(pred, target, reduction="mean")
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, device=pred.device, dtype=pred.dtype)
        if mask.shape != pred.shape:
            mask = mask.reshape(pred.shape)
        mask = mask.to(device=pred.device, dtype=pred.dtype)
        abs_err = (pred - target).abs() * mask
        count = mask.sum().clamp_min(self.eps)
        return abs_err.sum() / count


@register_loss("masked_mse")
class MaskedMSELoss(BaseLoss):
    """MSE loss computed only on positions where ``extras["mask"] == 1``."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("MaskedMSELoss requires `target`.")
        mask = extras.get("mask")
        if mask is None:
            return nn.functional.mse_loss(pred, target, reduction="mean")
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, device=pred.device, dtype=pred.dtype)
        if mask.shape != pred.shape:
            mask = mask.reshape(pred.shape)
        mask = mask.to(device=pred.device, dtype=pred.dtype)
        sq_err = (pred - target).pow(2) * mask
        count = mask.sum().clamp_min(self.eps)
        return sq_err.sum() / count


@register_loss("weighted_composite")
class WeightedCompositeLoss(BaseLoss):
    """Weighted sum of sub-losses, each specified as a ``{type, params}`` block.

    Parameters
    ----------
    terms : list[dict]
        List of loss config blocks, e.g. ``[{"type": "mse", "params": {...}}]``.
    weights : list[float], optional
        Per-term weight multipliers. Defaults to 1.0 for each term.
    """

    def __init__(
        self,
        terms: list,
        weights: Optional[list] = None,
    ) -> None:
        super().__init__()
        if not terms:
            raise ValueError("WeightedCompositeLoss requires at least one term.")
        self.terms = nn.ModuleList([build_loss(t) for t in terms])
        if weights is None:
            weights = [1.0] * len(terms)
        if len(weights) != len(terms):
            raise ValueError(
                f"weights length ({len(weights)}) must match terms length ({len(terms)})."
            )
        self.weights = [float(w) for w in weights]

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        total = pred.new_tensor(0.0)
        for w, term in zip(self.weights, self.terms):
            total = total + w * term(pred, target, **extras)
        return total


@register_loss("anet_ssim_l1")
class ANetSSIML1Loss(BaseLoss):
    """ANet hybrid loss ``-SSIM(pred, target) + lambda_l1 * L1(pred, target)``.

    Reproduces the hybrid loss of Yu and Wu (IEEE TGRS 2022,
    DOI 10.1109/TGRS.2021.3068279): SSIM uses per-sample global-patch
    statistics (paper Eq. (1)-(6), ``c3 = c2 / 2`` merged form) and L1 is the
    plain mean absolute error.  The loss equals ``-1`` for a perfect
    reconstruction; this matches the paper and must not be shifted to
    ``1 - SSIM``.  ``lambda_l1`` weights the L1 term (paper: 1.0).
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        ssim_mode: str = "global_patch",
        c1: float = 1.0e-4,
        c2: float = 9.0e-4,
    ) -> None:
        super().__init__()
        if ssim_mode != "global_patch":
            raise ValueError(
                f"ANetSSIML1Loss supports ssim_mode='global_patch' only, got {ssim_mode!r}."
            )
        if c1 <= 0 or c2 <= 0:
            raise ValueError(f"c1 and c2 must be positive, got c1={c1}, c2={c2}.")
        self.lambda_l1 = float(lambda_l1)
        self.ssim_mode = ssim_mode
        self.c1 = float(c1)
        self.c2 = float(c2)

    def _global_patch_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-sample SSIM from global-patch statistics; returns shape ``(B,)``."""
        x = pred.flatten(start_dim=1)
        y = target.flatten(start_dim=1)
        mu_x = x.mean(dim=1)
        mu_y = y.mean(dim=1)
        var_x = x.var(dim=1, unbiased=False)
        var_y = y.var(dim=1, unbiased=False)
        cov_xy = (x * y).mean(dim=1) - mu_x * mu_y
        numerator = (2.0 * mu_x * mu_y + self.c1) * (2.0 * cov_xy + self.c2)
        denominator = (mu_x.square() + mu_y.square() + self.c1) * (var_x + var_y + self.c2)
        return numerator / denominator

    def components(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return ``loss_total`` / ``loss_ssim`` / ``loss_l1`` / ``ssim`` tensors."""
        ssim = self._global_patch_ssim(pred, target).mean()
        loss_ssim = -ssim
        loss_l1 = (pred - target).abs().flatten(start_dim=1).mean(dim=1).mean()
        loss_total = loss_ssim + self.lambda_l1 * loss_l1
        return {
            "loss_total": loss_total,
            "loss_ssim": loss_ssim,
            "loss_l1": loss_l1,
            "ssim": ssim,
        }

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("ANetSSIML1Loss requires `target`.")
        return self.components(pred, target)["loss_total"]


def build_loss(cfg: Dict[str, Any]) -> BaseLoss:
    """Instantiate a loss from a ``{type, params}`` config block."""
    name = cfg["type"]
    if name not in LOSS_REGISTRY:
        raise KeyError(
            f"Unknown loss '{name}'. Available: {sorted(LOSS_REGISTRY)}"
        )
    return LOSS_REGISTRY[name](**cfg.get("params", {}))
