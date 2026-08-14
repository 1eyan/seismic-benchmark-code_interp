"""Generic inference helpers: full-volume patchify-infer-unpatchify, per-shot metrics, and viz."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import (
    SSIM,
    _mae_per_sample_numpy,
    _mse_per_sample_numpy,
    _psnr_per_sample_numpy,
    _rmse_per_sample_numpy,
    _snr_per_sample_numpy,
)
from .visualization import plot_sample


# def inference_on_shots(
#     model: torch.nn.Module,
#     input_shots: np.ndarray,
#     patch_size: Tuple[int, int],
#     overlap: float = 0.0,
#     device: torch.device = torch.device("cpu"),
#     batch_size: int = 8,
# ) -> np.ndarray:
#     """Patchify a full shot volume, run the model in batches, and reconstruct.
#
#     Parameters
#     ----------
#     model        : trained ``nn.Module``; will be set to ``eval()`` inside.
#     input_shots  : ``(n_shots, n_traces, n_time)`` numpy array.
#     patch_size   : ``(trace, time)`` patch shape forwarded to ``patchify_uniform``.
#     overlap      : overlap ratio forwarded to ``patchify_uniform``.
#     device       : device to run inference on.
#     batch_size   : batch size for the internal DataLoader.
#
#     Returns
#     -------
#     pred_shots   : ``(n_shots, n_traces, n_time)`` numpy array reconstructed by
#                    ``unpatchify_uniform``.
#     """
#     from tools.patching import patchify_uniform, unpatchify_uniform
#
#     patches, info = patchify_uniform(
#         input_shots, patch_size=patch_size, overlap=overlap, output_ndim=4
#     )
#     ds = TensorDataset(torch.from_numpy(patches))
#     loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
#
#     was_training = model.training
#     model.eval()
#     preds: List[torch.Tensor] = []
#     try:
#         with torch.no_grad():
#             for (batch,) in loader:
#                 batch = batch.to(device, non_blocking=True)
#                 out = model(batch)
#                 preds.append(out.cpu())
#     finally:
#         if was_training:
#             model.train()
#
#     pred_patches = torch.cat(preds, dim=0).numpy()
#     return unpatchify_uniform(pred_patches, info)

def inference_on_shots(
    model: torch.nn.Module,
    input_shots: np.ndarray,
    patch_size: Tuple[int, int],
    overlap: float = 0.0,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 8,
    patch_normalize: bool = False,
    patch_norm_eps: float = 1e-6,
    include_mask_channel: bool = False,
    mask_mode: Optional[str] = None,
    mask_ratio_range: Optional[Tuple[float, float]] = None,
    mask_seed: Optional[int] = None,
    return_missing_mask: bool = False,
) -> "np.ndarray | Tuple[np.ndarray, np.ndarray]":
    """Patchify a full shot volume, run the model in batches, and reconstruct.

    Parameters
    ----------
    model            : trained ``nn.Module``; will be set to ``eval()`` inside.
    input_shots      : ``(n_shots, n_traces, n_time)`` numpy array.
    patch_size       : ``(trace, time)`` patch shape forwarded to ``patchify_uniform``.
    overlap          : overlap ratio forwarded to ``patchify_uniform``.
    device           : device to run inference on.
    batch_size       : batch size for the internal DataLoader.
    patch_normalize  : if True, each input patch is normalized by its own
                       visible max-absolute amplitude before inference.
    patch_norm_eps   : lower bound for the patch scale.
    include_mask_channel : if True, concatenate a binary mask channel
                       (1 = missing, 0 = observed) to the input patches,
                       producing C=2 inputs. Safe because masked traces
                       are exactly zero after normalisation.
    mask_mode        : optional per-patch missing-trace masking applied after
                       patchification; only ``"cfunet_random"`` is supported
                       (mirrors the training cfunet_random branch).
    mask_ratio_range : ``(low, high)`` per-patch missing-ratio range drawn
                       uniformly; default ``(0.5, 0.875)``.
    mask_seed        : seed for the masking RNG; ``None`` = unseeded.
    return_missing_mask : if True, also return the overlap-weighted per-position
                       missing fraction (0 = never masked, 1 = always masked).

    Returns
    -------
    pred_shots       : ``(n_shots, n_traces, n_time)`` numpy array reconstructed by
                       ``unpatchify_uniform``. When ``return_missing_mask`` is
                       True, returns ``(pred_shots, missing_frac)`` with
                       ``missing_frac`` of the same shape and values in ``[0, 1]``.
    """
    from tools.patching import patchify_uniform, unpatchify_uniform
    from tools.preprocessing import mask_traces

    patches, info = patchify_uniform(
        input_shots,
        patch_size=patch_size,
        overlap=overlap,
        output_ndim=4,
    )

    patches = patches.astype(np.float32, copy=False)

    miss_trace_mask: Optional[np.ndarray] = None
    if mask_mode is not None:
        if mask_mode != "cfunet_random":
            raise ValueError(
                f"mask_mode {mask_mode!r} is not supported (only 'cfunet_random')."
            )
        # Same per-patch protocol as the training cfunet_random branch
        # (scripts/interpolation/train_interpolation_unet.py).
        ratio_range = mask_ratio_range or (0.5, 0.875)
        rng = np.random.default_rng(mask_seed) if mask_seed is not None else None
        patches_3d = patches[:, 0, :, :]
        masked_3d, miss_trace_mask = mask_traces(
            patches_3d,
            mode="random",
            ratio_range=(float(ratio_range[0]), float(ratio_range[1])),
            rng=rng,
        )
        patches = masked_3d[:, None, :, :]

    if patch_normalize:
        scales = np.max(
            np.abs(patches),
            axis=(1, 2, 3),
            keepdims=True,
        ).astype(np.float32)
        scales = np.maximum(scales, np.float32(patch_norm_eps))
        patches_for_net = patches / scales
    else:
        scales = None
        patches_for_net = patches

    if include_mask_channel:
        # Binary mask channel: 1 = missing, 0 = observed.
        # Safe because mask_traces sets missing traces to exactly 0,
        # and max_abs / patch_normalize both preserve zeros.
        mask_channel = (patches_for_net[:, :1, :, :] == 0).astype(np.float32)
        patches_for_net = np.concatenate([patches_for_net, mask_channel], axis=1)

    # Derive per-trace observed mask for models that accept a ``mask`` kwarg.
    obs_mask = (
        np.abs(patches_for_net[:, :1, :, :]).max(axis=3, keepdims=True) > 1e-8
    ).astype(np.float32)  # (P, 1, n_traces, 1)
    obs_mask = np.broadcast_to(obs_mask, patches_for_net[:, :1, :, :].shape).copy()

    # Check whether the model accepts a ``mask`` keyword argument.
    sig = inspect.signature(
        model.module.forward if hasattr(model, "module") else model.forward
    )
    needs_mask_kwarg = "mask" in sig.parameters

    if needs_mask_kwarg:
        ds = TensorDataset(
            torch.from_numpy(patches_for_net),
            torch.from_numpy(obs_mask),
        )
    else:
        ds = TensorDataset(torch.from_numpy(patches_for_net))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    was_training = model.training
    model.eval()
    preds: List[torch.Tensor] = []

    try:
        with torch.no_grad():
            for batch_items in loader:
                if needs_mask_kwarg:
                    batch, batch_mask = batch_items
                    batch = batch.to(device, non_blocking=True)
                    batch_mask = batch_mask.to(device, non_blocking=True)
                    out = model(batch, mask=batch_mask)
                else:
                    (batch,) = batch_items
                    batch = batch.to(device, non_blocking=True)
                    out = model(batch)
                preds.append(out.cpu())
    finally:
        if was_training:
            model.train()

    pred_patches = torch.cat(preds, dim=0).numpy()

    if patch_normalize:
        pred_patches = pred_patches * scales

    pred_shots = unpatchify_uniform(pred_patches, info)

    if return_missing_mask:
        if miss_trace_mask is None:
            missing_frac = np.zeros(pred_shots.shape, dtype=np.float32)
        else:
            h, w = info["patch_size"]
            miss_full = np.broadcast_to(
                miss_trace_mask[:, :, None],
                (miss_trace_mask.shape[0], h, w),
            ).astype(np.float32)
            missing_frac = unpatchify_uniform(miss_full, info)
        return pred_shots, missing_frac

    return pred_shots

def compute_shot_metrics(
    pred_shots: np.ndarray,
    target_shots: np.ndarray,
    metric_names: List[str],
    *,
    psnr_peak: float = 1.0,
    ssim_data_range: float = 2.0,
    eps: float = 1e-12,
    mask: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Compute per-shot metrics and their means.

    Parameters
    ----------
    pred_shots      : ``(n_shots, n_traces, n_time)``.
    target_shots    : same shape as ``pred_shots``.
    metric_names    : list of metric names; supported: ``mse``, ``rmse``, ``mae``,
                      ``snr``, ``psnr``, ``ssim``.
    psnr_peak       : maximum amplitude (peak) for PSNR. For ``max_abs``
                      normalized data in ``[-1, 1]`` this is ``1.0``, **not**
                      the peak-to-peak range ``2.0``.
    ssim_data_range : peak-to-peak range for SSIM (``L`` in Wang et al. 2004).
                      For ``max_abs`` ``[-1, 1]`` data this is ``2.0``.
    eps             : small constant to avoid division by zero / log of zero.
    mask            : optional boolean array of same shape as pred_shots.
                      ``True`` = evaluate this position (missing trace).
                      When ``None``, all positions are evaluated (backward
                      compatible). Scalar metrics (MSE/MAE/RMSE/SNR/PSNR)
                      are computed only on masked positions; SSIM is
                      computed on the full image with observed positions
                      filled from target_shots.

    Returns
    -------
    per_shot : ``{metric_name: (n_shots,) ndarray}``.
    mean     : ``{metric_name: float}`` — mean over shots.
    """
    if pred_shots.shape != target_shots.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_shots.shape} vs target {target_shots.shape}."
        )
    if mask is not None and mask.shape != pred_shots.shape:
        raise ValueError(
            f"Mask shape {mask.shape} != pred shape {pred_shots.shape}."
        )

    n_shots = pred_shots.shape[0]
    pred = pred_shots.astype(np.float32)
    tgt = target_shots.astype(np.float32)
    use_mask = mask is not None
    if use_mask:
        mask_float = mask.astype(np.float32)
        counts = mask_float.sum(axis=(1, 2))  # (n_shots,) masked pixels per shot
        counts = np.maximum(counts, 1)  # avoid div-by-zero on empty shots
    else:
        counts = None

    if use_mask:
        sq_err = (pred - tgt) ** 2
        sq_err[~mask] = 0.0
        mse_arr = sq_err.sum(axis=(1, 2)) / counts

        abs_err = np.abs(pred - tgt)
        abs_err[~mask] = 0.0
        mae_arr = abs_err.sum(axis=(1, 2)) / counts

        rmse_arr = np.sqrt(mse_arr)

        signal_power = ((tgt ** 2) * mask_float).sum(axis=(1, 2))
        noise_power = ((pred - tgt) ** 2 * mask_float).sum(axis=(1, 2))
        ratio = signal_power / np.maximum(noise_power, eps)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_arr = 10.0 * np.log10(ratio)
        snr_arr = np.where((signal_power == 0.0) & (noise_power == 0.0), np.nan, snr_arr)

        psnr_arr = 10.0 * np.log10((psnr_peak ** 2) / np.maximum(mse_arr, eps))
    else:
        mse_arr = _mse_per_sample_numpy(pred, tgt)
        mae_arr = _mae_per_sample_numpy(pred, tgt)
        rmse_arr = _rmse_per_sample_numpy(pred, tgt)
        snr_arr = _snr_per_sample_numpy(pred, tgt, eps)
        psnr_arr = _psnr_per_sample_numpy(pred, tgt, psnr_peak, eps)

    per_shot: Dict[str, np.ndarray] = {}
    mean: Dict[str, float] = {}

    for name in metric_names:
        name_lower = name.lower()
        if name_lower == "mse":
            arr = mse_arr
        elif name_lower == "rmse":
            arr = rmse_arr
        elif name_lower == "mae":
            arr = mae_arr
        elif name_lower == "snr":
            arr = snr_arr
        elif name_lower == "psnr":
            arr = psnr_arr
        elif name_lower == "ssim":
            if use_mask:
                ssim_pred = np.where(mask, pred, tgt)
            else:
                ssim_pred = pred
            arr = _compute_ssim_per_shot(ssim_pred, tgt, ssim_data_range)
        else:
            raise ValueError(f"Unsupported metric for per-shot computation: {name!r}")
        per_shot[name_lower] = arr
        val = float(np.nanmean(arr))
        if name_lower == "mse":
            pass  # keep full precision
        elif name_lower in ("mae", "rmse"):
            val = round(val, 6)
        elif name_lower in ("snr", "psnr", "ssim"):
            val = round(val, 4)
        else:
            val = round(val, 2)
        mean[name_lower] = val

    return per_shot, mean


def _compute_ssim_per_shot(
    pred: np.ndarray, target: np.ndarray, data_range: float
) -> np.ndarray:
    """SSIM for each shot; pred/target are ``(n_shots, n_traces, n_time)``."""
    n_shots = pred.shape[0]
    ssim_metric = SSIM(data_range=data_range)
    values = np.empty(n_shots, dtype=np.float32)
    for i in range(n_shots):
        pred_t = torch.from_numpy(pred[i]).unsqueeze(0).unsqueeze(0)
        tgt_t = torch.from_numpy(target[i]).unsqueeze(0).unsqueeze(0)
        values[i] = ssim_metric(pred_t, tgt_t)
    return values


def compute_binned_metrics(
    pred_shots: np.ndarray,
    target_shots: np.ndarray,
    dt: float,
    *,
    eb_enabled: bool = True,
    eb_bins: Sequence[Tuple[int, int]] = ((5, 20), (20, 40), (40, 70), (70, 100)),
    eb_smooth_sigma: float = 1.0,
    fb_enabled: bool = True,
    fb_rel_threshold: float = 0.001,
    fb_band_ratios: Sequence[float] = (0.20, 0.30, 0.30, 0.20),
    fb_band_names: Sequence[str] = ("low", "mid", "high", "very_high"),
    fb_taper_width: Optional[float] = None,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """Compute mean EB-WSE and FB-FRE metrics over all shots.

    EB-WSE reports normalized error (``ne``) and SNR (``snr``) per energy
    percentile bin. FB-FRE estimates an effective frequency band from the clean
    volume, splits it into adaptive bands, and reports per-band ``ne``,
    ``snr``, and an energy ratio computed on band-pass filtered shots.

    Parameters
    ----------
    pred_shots       : ``(n_shots, n_traces, n_time)``.
    target_shots     : same shape as ``pred_shots``.
    dt               : time sampling interval in seconds.
    eb_enabled       : whether to compute EB-WSE diagnostics.
    eb_bins          : sequence of ``(low_percentile, high_percentile)`` tuples.
    eb_smooth_sigma  : Gaussian smoothing sigma for the EB-WSE energy map.
    fb_enabled       : whether to compute FB-FRE diagnostics.
    fb_rel_threshold : fraction of peak power used to define the effective band.
    fb_band_ratios   : relative widths of the adaptive FB-FRE bands; must sum to 1.
    fb_band_names    : name for each adaptive FB-FRE band.
    fb_taper_width   : optional cap on the cosine taper width in Hz. When
        ``None``, each band uses ``0.1 * band_width`` as its taper width.
        When a positive float is given, the actual taper is
        ``min(0.1 * band_width, fb_taper_width)``. ``0.0`` disables tapering
        (rectangular passband).
    eps              : small constant to avoid division by zero.

    Returns
    -------
    mean : ``{metric_key: float | list | None}`` — mean over shots. Non-finite
           values are replaced with ``None`` so the result is JSON-serializable.
           Empty when both EB-WSE and FB-FRE are disabled.
    """
    if not eb_enabled and not fb_enabled:
        return {}

    if pred_shots.shape != target_shots.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_shots.shape} vs target {target_shots.shape}."
        )

    n_shots = pred_shots.shape[0]
    pred = pred_shots.astype(np.float64, copy=False)
    tgt = target_shots.astype(np.float64, copy=False)

    def _sanitize_for_json(value: float) -> Optional[float]:
        """Cap infinities and replace NaN so JSON output stays standard.

        ``inf`` / ``-inf`` are replaced with large finite caps; ``nan`` becomes
        ``None``.
        """
        if np.isnan(value):
            return None
        if np.isposinf(value):
            return 999.0
        if np.isneginf(value):
            return -999.0
        return value

    mean: Dict[str, Any] = {}

    if eb_enabled:
        from .eb_wse_metrics import energy_binned_weak_signal_metrics

        # Run one shot to discover the bin keys (default keys are named; custom
        # keys follow ``bin_low_high``).
        eb_sample = energy_binned_weak_signal_metrics(
            tgt[0], pred[0], bins=eb_bins, smooth_sigma=eb_smooth_sigma, eps=eps
        )
        eb_keys = list(eb_sample.keys())
        eb_sums: Dict[str, Dict[str, float]] = {
            key: {"ne": 0.0, "snr": 0.0} for key in eb_keys
        }

        for i in range(n_shots):
            eb_result = energy_binned_weak_signal_metrics(
                tgt[i], pred[i], bins=eb_bins, smooth_sigma=eb_smooth_sigma, eps=eps
            )
            for key in eb_keys:
                eb_sums[key]["ne"] += eb_result[key]["NE"]
                eb_sums[key]["snr"] += eb_result[key]["SNR"]

        for key in eb_keys:
            mean[f"eb_wse_{key}_ne"] = _sanitize_for_json(
                round(eb_sums[key]["ne"] / n_shots, 6)
            )
            mean[f"eb_wse_{key}_snr"] = _sanitize_for_json(
                round(eb_sums[key]["snr"] / n_shots, 6)
            )

    if fb_enabled:
        from .fb_fre_metrics import (
            _bandpass_filter,
            build_auto_bands,
            estimate_effective_band,
        )

        # Estimate effective band on the full clean volume and reuse bands per shot.
        f_min, f_max = estimate_effective_band(
            tgt, dt=dt, rel_threshold=fb_rel_threshold
        )
        bands = build_auto_bands(
            f_min, f_max, ratios=fb_band_ratios, names=fb_band_names
        )
        fb_sums: Dict[str, Dict[str, float]] = {
            band_name: {"ne": 0.0, "snr": 0.0, "energy_ratio": 0.0}
            for band_name, _ in bands
        }

        for i in range(n_shots):
            total_energy = float(np.sum(tgt[i] ** 2))
            for band_name, (fmin, fmax) in bands:
                band_width = fmax - fmin
                taper_width = 0.1 * band_width
                if fb_taper_width is not None:
                    taper_width = min(taper_width, fb_taper_width)
                ref_band = _bandpass_filter(
                    tgt[i], dt, fmin, fmax, taper_width=taper_width
                )
                pred_band = _bandpass_filter(
                    pred[i], dt, fmin, fmax, taper_width=taper_width
                )
                diff = pred_band - ref_band
                ref_energy = float(np.sum(ref_band ** 2))
                err_energy = float(np.sum(diff ** 2))
                ne = float(np.sqrt(err_energy) / (np.sqrt(ref_energy) + eps))
                if ref_energy > 0.0:
                    snr = float(10.0 * np.log10(ref_energy / (err_energy + eps)))
                else:
                    snr = float("-inf")
                energy_ratio = float(ref_energy / (total_energy + eps))
                fb_sums[band_name]["ne"] += ne
                fb_sums[band_name]["snr"] += snr
                fb_sums[band_name]["energy_ratio"] += energy_ratio

        for band_name, (fmin, fmax) in bands:
            mean[f"fb_fre_{band_name}_ne"] = _sanitize_for_json(
                round(fb_sums[band_name]["ne"] / n_shots, 6)
            )
            mean[f"fb_fre_{band_name}_snr"] = _sanitize_for_json(
                round(fb_sums[band_name]["snr"] / n_shots, 6)
            )
            mean[f"fb_fre_{band_name}_energy_ratio"] = _sanitize_for_json(
                round(fb_sums[band_name]["energy_ratio"] / n_shots, 6)
            )
            mean[f"fb_fre_{band_name}_frequency_range_hz"] = [
                round(float(fmin), 4),
                round(float(fmax), 4),
            ]

    return mean


def build_binned_metric_kwargs(infer_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build keyword arguments for ``compute_binned_metrics`` from inference config.

    The expected YAML shape is::

        inference:
          binned_metrics:
            enabled: true
            eb_wse:
              enabled: true
              bins: [[5, 20], [20, 40], [40, 70], [70, 100]]
              smooth_sigma: 1.0
            fb_fre:
              enabled: true
              rel_threshold: 0.001
              band_ratios: [0.20, 0.30, 0.30, 0.20]
              band_names: ["low", "mid", "high", "very_high"]
              # ``null``/omitted -> taper is 10% of each band width.
              # Positive float -> cap the adaptive taper at that Hz.
              # ``0.0`` -> rectangular passband (Gibbs ringing).
              taper_width: 2.0

    Missing fields use the same defaults as ``compute_binned_metrics``, so older
    configs without ``binned_metrics`` continue to work. A config that omits
    ``taper_width`` now gets adaptive 10%-of-band tapering instead of a fixed Hz
    value.

    Parameters
    ----------
    infer_cfg : the ``inference`` block from the loaded config.

    Returns
    -------
    kwargs : keyword arguments ready to be unpacked into
             ``compute_binned_metrics(..., **kwargs)``.
    """
    binned_cfg = infer_cfg.get("binned_metrics", {})
    if not binned_cfg.get("enabled", True):
        return {"eb_enabled": False, "fb_enabled": False}

    eb_cfg = binned_cfg.get("eb_wse", {})
    fb_cfg = binned_cfg.get("fb_fre", {})

    kwargs: Dict[str, Any] = {
        "eb_enabled": eb_cfg.get("enabled", True),
        "fb_enabled": fb_cfg.get("enabled", True),
    }

    if kwargs["eb_enabled"]:
        kwargs["eb_bins"] = tuple(
            tuple(b) for b in eb_cfg.get("bins", ((5, 20), (20, 40), (40, 70), (70, 100)))
        )
        kwargs["eb_smooth_sigma"] = eb_cfg.get("smooth_sigma", 1.0)

    if kwargs["fb_enabled"]:
        kwargs["fb_rel_threshold"] = fb_cfg.get("rel_threshold", 0.001)
        kwargs["fb_band_ratios"] = tuple(
            fb_cfg.get("band_ratios", (0.20, 0.30, 0.30, 0.20))
        )
        kwargs["fb_band_names"] = tuple(
            fb_cfg.get("band_names", ("low", "mid", "high", "very_high"))
        )
        kwargs["fb_taper_width"] = fb_cfg.get("taper_width", None)

    return kwargs


def select_random_shots(
    n_shots: int, n_select: int, seed: Optional[int] = None
) -> np.ndarray:
    """Return a random subset of shot indices without replacement."""
    rng = np.random.default_rng(seed)
    n = min(n_select, n_shots)
    return rng.choice(n_shots, size=n, replace=False)


def save_shot_visualizations(
    input_shots: np.ndarray,
    pred_shots: np.ndarray,
    target_shots: np.ndarray,
    indices: np.ndarray,
    save_dir: Path,
    title_prefix: str = "shot",
    cmap: str = "seismic",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    share_scale: bool = True,
    save_npy: bool = False,
) -> List[Path]:
    """Save one 4-panel PNG per selected shot.

    Parameters
    ----------
    input_shots  : ``(n_shots, n_traces, n_time)``.
    pred_shots   : same shape.
    target_shots : same shape.
    indices      : 1-D array of shot indices to visualize.
    save_dir     : directory for output PNGs (created if missing).
    title_prefix : prefix for figure titles and filenames.
    cmap         : Matplotlib colormap name.
    save_npy     : if ``True``, also save ``input/prediction/target`` arrays for
                   each selected shot under ``save_dir/data/``.

    Returns
    -------
    paths        : list of saved PNG paths.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    npy_dir = save_dir / "data" if save_npy else None
    if npy_dir is not None:
        npy_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for idx in indices:
        idx = int(idx)
        path = save_dir / f"{title_prefix}_shot_{idx:04d}.png"
        plot_sample(
            input_data=input_shots[idx],
            prediction=pred_shots[idx],
            target=target_shots[idx],
            save_path=path,
            title=f"{title_prefix} shot {idx}",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            share_scale=share_scale,
        )
        paths.append(path)
        if npy_dir is not None:
            np.save(npy_dir / f"{title_prefix}_shot_{idx:04d}_input.npy", input_shots[idx])
            np.save(npy_dir / f"{title_prefix}_shot_{idx:04d}_prediction.npy", pred_shots[idx])
            np.save(npy_dir / f"{title_prefix}_shot_{idx:04d}_target.npy", target_shots[idx])
    return paths