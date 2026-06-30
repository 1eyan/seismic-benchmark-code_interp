"""Generic inference helpers: full-volume patchify-infer-unpatchify, per-shot metrics, and viz."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
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


def inference_on_shots(
    model: torch.nn.Module,
    input_shots: np.ndarray,
    patch_size: Tuple[int, int],
    overlap: float = 0.0,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 8,
) -> np.ndarray:
    """Patchify a full shot volume, run the model in batches, and reconstruct.

    Parameters
    ----------
    model        : trained ``nn.Module``; will be set to ``eval()`` inside.
    input_shots  : ``(n_shots, n_traces, n_time)`` numpy array.
    patch_size   : ``(trace, time)`` patch shape forwarded to ``patchify_uniform``.
    overlap      : overlap ratio forwarded to ``patchify_uniform``.
    device       : device to run inference on.
    batch_size   : batch size for the internal DataLoader.

    Returns
    -------
    pred_shots   : ``(n_shots, n_traces, n_time)`` numpy array reconstructed by
                   ``unpatchify_uniform``.
    """
    from tools.patching import patchify_uniform, unpatchify_uniform

    patches, info = patchify_uniform(
        input_shots, patch_size=patch_size, overlap=overlap, output_ndim=4
    )
    ds = TensorDataset(torch.from_numpy(patches))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    was_training = model.training
    model.eval()
    preds: List[torch.Tensor] = []
    try:
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device, non_blocking=True)
                out = model(batch)
                preds.append(out.cpu())
    finally:
        if was_training:
            model.train()

    pred_patches = torch.cat(preds, dim=0).numpy()
    return unpatchify_uniform(pred_patches, info)


def compute_shot_metrics(
    pred_shots: np.ndarray,
    target_shots: np.ndarray,
    metric_names: List[str],
    *,
    psnr_peak: float = 1.0,
    ssim_data_range: float = 2.0,
    eps: float = 1e-12,
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

    Returns
    -------
    per_shot : ``{metric_name: (n_shots,) ndarray}``.
    mean     : ``{metric_name: float}`` — mean over shots.
    """
    if pred_shots.shape != target_shots.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_shots.shape} vs target {target_shots.shape}."
        )

    n_shots = pred_shots.shape[0]
    pred = pred_shots.astype(np.float32)
    tgt = target_shots.astype(np.float32)

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
            arr = _compute_ssim_per_shot(pred, tgt, ssim_data_range)
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
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """Compute mean EB-WSE and FB-FRE metrics over all shots.

    EB-WSE reports normalized error (``ne``) and SNR (``snr``) for the four
    default energy percentile bins. FB-FRE estimates an effective frequency band
    from the clean volume, splits it into adaptive low/mid/high/very_high bands,
    and reports per-band ``ne`` and ``snr`` computed on band-pass filtered shots.

    Parameters
    ----------
    pred_shots   : ``(n_shots, n_traces, n_time)``.
    target_shots : same shape as ``pred_shots``.
    dt           : time sampling interval in seconds.
    eps          : small constant to avoid division by zero.

    Returns
    -------
    mean : ``{metric_key: float | list | None}`` — mean over shots. Non-finite
           values are replaced with ``None`` so the result is JSON-serializable.
    """
    from .eb_wse_metrics import energy_binned_weak_signal_metrics
    from .fb_fre_metrics import (
        _bandpass_filter,
        build_auto_bands,
        estimate_effective_band,
    )

    if pred_shots.shape != target_shots.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_shots.shape} vs target {target_shots.shape}."
        )

    n_shots = pred_shots.shape[0]
    pred = pred_shots.astype(np.float64, copy=False)
    tgt = target_shots.astype(np.float64, copy=False)

    def _sanitize_for_json(value: float) -> Optional[float]:
        """Cap infinities and replace NaN so JSON output stays standard.

        ``inf`` /
        ``-inf`` are replaced with large finite caps; ``nan`` becomes ``None``.
        """
        if np.isnan(value):
            return None
        if np.isposinf(value):
            return 999.0
        if np.isneginf(value):
            return -999.0
        return value

    # EB-WSE: default energy percentile bins.
    eb_bins = ((5, 20), (20, 40), (40, 70), (70, 100))
    eb_keys = ["very_weak_5_20", "weak_20_40", "medium_40_70", "strong_70_100"]
    eb_sums: Dict[str, Dict[str, float]] = {
        key: {"ne": 0.0, "snr": 0.0} for key in eb_keys
    }

    # FB-FRE: estimate effective band on the full clean volume and reuse bands per shot.
    f_min, f_max = estimate_effective_band(tgt, dt=dt, rel_threshold=0.001)
    bands = build_auto_bands(f_min, f_max)
    fb_sums: Dict[str, Dict[str, float]] = {
        band_name: {"ne": 0.0, "snr": 0.0, "energy_ratio": 0.0} for band_name, _ in bands
    }

    for i in range(n_shots):
        eb_result = energy_binned_weak_signal_metrics(
            tgt[i], pred[i], bins=eb_bins, eps=eps
        )
        for key in eb_keys:
            eb_sums[key]["ne"] += eb_result[key]["NE"]
            eb_sums[key]["snr"] += eb_result[key]["SNR"]

        total_energy = float(np.sum(tgt[i] ** 2))
        for band_name, (fmin, fmax) in bands:
            ref_band = _bandpass_filter(tgt[i], dt, fmin, fmax)
            pred_band = _bandpass_filter(pred[i], dt, fmin, fmax)
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

    mean: Dict[str, Any] = {}
    for key in eb_keys:
        mean[f"eb_wse_{key}_ne"] = _sanitize_for_json(
            round(eb_sums[key]["ne"] / n_shots, 6)
        )
        mean[f"eb_wse_{key}_snr"] = _sanitize_for_json(
            round(eb_sums[key]["snr"] / n_shots, 6)
        )
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
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    share_scale: bool = True,
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

    Returns
    -------
    paths        : list of saved PNG paths.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
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
    return paths