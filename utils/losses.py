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


# Alias: generic SSIM+L1 hybrid loss (same formula used by ANet, CA-Unet, etc.).
LOSS_REGISTRY["ssim_l1"] = ANetSSIML1Loss


# ----------------------------------------------------------------------
# Pan2020 partial convolution composite loss
# ----------------------------------------------------------------------


@register_loss("pan2020_pconv_composite")
class Pan2020PConvLoss(BaseLoss):
    """Pan2020 composite loss: L_valid + hole_weight * L_hole + tv_weight * L_tv.

    All three terms use per-sample spatial SUM then batch MEAN reduction,
    matching the author-code convention (paper: Computers & Geosciences,
    vol. 145, 2020, DOI 10.1016/j.cageo.2020.104609).

    L_valid : L1 error on observed positions only (mask == 1).
    L_hole  : L1 error on missing positions only (mask == 0).
    L_tv    : total-variation on the hole-neighbourhood composite image,
              i.e. the reconstruction where observed pixels use the
              ground-truth target and missing pixels use the network
              prediction.  The hole mask is first dilated so the TV
              regulariser also penalises discontinuities along the
              hole boundary.

    Parameters
    ----------
    hole_weight : weight of the hole L1 term (paper: 6.0).
    tv_weight   : weight of the TV term (paper: 0.1).
    tv_dilation : kernel size for hole-mask dilation (paper: 7).
    """

    def __init__(
        self,
        hole_weight: float = 6.0,
        tv_weight: float = 0.1,
        tv_dilation: int = 7,
    ) -> None:
        super().__init__()
        if hole_weight < 0:
            raise ValueError(f"hole_weight must be non-negative, got {hole_weight}.")
        if tv_weight < 0:
            raise ValueError(f"tv_weight must be non-negative, got {tv_weight}.")
        if tv_dilation < 1 or tv_dilation % 2 == 0:
            raise ValueError(f"tv_dilation must be a positive odd integer, got {tv_dilation}.")
        self.hole_weight = float(hole_weight)
        self.tv_weight = float(tv_weight)
        self.tv_dilation = int(tv_dilation)

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("Pan2020PConvLoss requires `target`.")
        mask = extras.get("mask")
        if mask is None:
            raise ValueError("Pan2020PConvLoss requires extras['mask'] (1=observed, 0=missing).")
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, device=pred.device, dtype=pred.dtype)
        if mask.shape != pred.shape:
            mask = mask.reshape(pred.shape)
        mask = mask.to(device=pred.device, dtype=pred.dtype)

        hole_mask = 1.0 - mask

        abs_err = (pred - target).abs()

        # L_valid: per-sample spatial sum, then batch mean
        loss_valid = (abs_err * mask).sum(dim=(1, 2, 3)).mean()

        # L_hole: per-sample spatial sum, then batch mean
        loss_hole = (abs_err * hole_mask).sum(dim=(1, 2, 3)).mean()

        # TV on hole-neighbourhood composite
        if self.tv_weight > 0:
            composite = mask * target + hole_mask * pred
            # Dilate hole mask to capture boundary
            d = self.tv_dilation
            dilated = torch.nn.functional.max_pool2d(
                hole_mask, kernel_size=d, stride=1, padding=d // 2,
            )
            dilated = (dilated > 0).to(dtype=pred.dtype)

            diff_h = (composite[:, :, 1:, :] - composite[:, :, :-1, :]).abs()
            diff_w = (composite[:, :, :, 1:] - composite[:, :, :, :-1]).abs()

            # Align dilated mask with diff grids
            tv_h = (diff_h * dilated[:, :, 1:, :]).sum(dim=(1, 2, 3)).mean()
            tv_w = (diff_w * dilated[:, :, :, 1:]).sum(dim=(1, 2, 3)).mean()
            loss_tv = tv_h + tv_w
        else:
            loss_tv = pred.new_tensor(0.0)

        return loss_valid + self.hole_weight * loss_hole + self.tv_weight * loss_tv


# ----------------------------------------------------------------------
# WRDL SSIM + Huber hybrid loss
# ----------------------------------------------------------------------


def _gaussian_window_1d(size: int, sigma: float) -> torch.Tensor:
    """1D Gaussian window, normalised to sum to 1."""
    coords = torch.arange(size, dtype=torch.float32)
    half = (size - 1) / 2.0
    g = torch.exp(-((coords - half) ** 2) / (2.0 * sigma * sigma))
    return g / g.sum()


def _create_gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """2D Gaussian window ``(channels, 1, window_size, window_size)``."""
    w1d = _gaussian_window_1d(window_size, sigma)
    w2d = torch.outer(w1d, w1d)  # (WS, WS)
    w2d = w2d.unsqueeze(0).unsqueeze(0)  # (1, 1, WS, WS)
    return w2d.expand(channels, 1, window_size, window_size)


@register_loss("wrdl_ssim_huber")
class WRDLSSIMHuberLoss(BaseLoss):
    """WRDL hybrid loss: ``ssim_weight * (1 - SSIM) + huber_weight * Huber``.

    SSIM uses a local Gaussian window (not global-patch), matching the WRDL
    paper's structural similarity formulation.  Huber uses the standard
    piecewise definition with configurable delta.

    All terms and the total loss are scalar tensors suitable for backprop.

    Parameters
    ----------
    ssim_weight : weight of the SSIM term. Default 1.0.
    huber_weight : weight of the Huber term. Default 1.0.
    huber_delta : Huber transition threshold. Default 1.0.
    window_size : Gaussian window size for local SSIM. Default 11.
    window_sigma : Gaussian sigma for SSIM window. Default 1.5.
    data_range : data value range for SSIM C1/C2 computation. Default 2.0
        (for [-1, 1] normalised data).
    c1_factor : C1 = (c1_factor * data_range)^2. Default 0.01.
    c2_factor : C2 = (c2_factor * data_range)^2. Default 0.03.
    """

    def __init__(
        self,
        ssim_weight: float = 1.0,
        huber_weight: float = 1.0,
        huber_delta: float = 1.0,
        window_size: int = 11,
        window_sigma: float = 1.5,
        data_range: float = 2.0,
        c1_factor: float = 0.01,
        c2_factor: float = 0.03,
    ) -> None:
        super().__init__()
        if ssim_weight < 0 or huber_weight < 0:
            raise ValueError("Loss weights must be non-negative.")
        if huber_delta <= 0:
            raise ValueError(f"huber_delta must be positive, got {huber_delta}.")
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be an odd integer >= 3, got {window_size}."
            )

        self.ssim_weight = float(ssim_weight)
        self.huber_weight = float(huber_weight)
        self.huber_delta = float(huber_delta)
        self.window_size = int(window_size)
        self.window_sigma = float(window_sigma)
        self.data_range = float(data_range)
        self.c1 = (c1_factor * data_range) ** 2
        self.c2 = (c2_factor * data_range) ** 2

    def _local_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-channel local-window SSIM; returns scalar mean across batch and space."""
        C = pred.shape[1]
        window = _create_gaussian_window(
            self.window_size, self.window_sigma, C
        ).to(device=pred.device, dtype=pred.dtype)
        pad = self.window_size // 2

        # Local means
        mu_p = F.conv2d(pred, window, padding=pad, groups=C)
        mu_t = F.conv2d(target, window, padding=pad, groups=C)

        mu_p_sq = mu_p.square()
        mu_t_sq = mu_t.square()
        mu_pt = mu_p * mu_t

        # Local variances and covariance
        sigma_p_sq = F.conv2d(pred.square(), window, padding=pad, groups=C) - mu_p_sq
        sigma_t_sq = F.conv2d(target.square(), window, padding=pad, groups=C) - mu_t_sq
        sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=C) - mu_pt

        # SSIM map
        numerator = (2.0 * mu_pt + self.c1) * (2.0 * sigma_pt + self.c2)
        denominator = (mu_p_sq + mu_t_sq + self.c1) * (sigma_p_sq + sigma_t_sq + self.c2)
        ssim_map = numerator / (denominator + 1e-8)

        return ssim_map.mean()

    def _huber(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Standard Huber loss, mean reduction."""
        diff = pred - target
        abs_diff = diff.abs()
        quadratic = 0.5 * diff.square()
        linear = self.huber_delta * (abs_diff - 0.5 * self.huber_delta)
        mask = abs_diff <= self.huber_delta
        loss = torch.where(mask, quadratic, linear)
        return loss.mean()

    def components(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Return ``loss_total`` / ``loss_ssim`` / ``loss_huber`` / ``ssim`` tensors."""
        ssim_val = self._local_ssim(pred, target)
        loss_ssim = 1.0 - ssim_val
        loss_huber = self._huber(pred, target)
        loss_total = self.ssim_weight * loss_ssim + self.huber_weight * loss_huber
        return {
            "loss_total": loss_total,
            "loss_ssim": loss_ssim,
            "loss_huber": loss_huber,
            "ssim": ssim_val,
        }

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("WRDLSSIMHuberLoss requires `target`.")
        return self.components(pred, target)["loss_total"]


def build_loss(cfg: Dict[str, Any]) -> BaseLoss:
    """Instantiate a loss from a ``{type, params}`` config block."""
    name = cfg["type"]
    if name not in LOSS_REGISTRY:
        raise KeyError(
            f"Unknown loss '{name}'. Available: {sorted(LOSS_REGISTRY)}"
        )
    return LOSS_REGISTRY[name](**cfg.get("params", {}))
