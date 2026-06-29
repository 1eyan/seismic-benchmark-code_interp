#!/usr/bin/env python3
"""Baseline evaluation pipeline for SEG-C3 random noise suppression.

Loads SEG-Y data, injects Poisson noise, runs a trained UNet via patch-based
inference, computes standard reconstruction metrics (MSE, RMSE, MAE, SNR, PSNR,
SSIM), and saves intermediate arrays / visualisations / metrics under
``test/outputs/``.

Usage
-----
python test/run_baseline_evaluation.py --config configs/test/config.yaml \
    --checkpoint result/checkpoint/best.pt --output_dir test/outputs

All paths are resolved relative to the repository root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# ----------------------------------------------------------------------
# 1. Bootstrap repo root
# ----------------------------------------------------------------------
_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "model").is_dir() and (p / "utils").is_dir()),
    Path(__file__).resolve().parents[1],
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.random_noise_suppression import build_model  # noqa: E402
from utils.eb_wse_metrics import energy_binned_weak_signal_metrics  # noqa: E402
from utils.fb_fre_metrics import (  # noqa: E402
    build_auto_bands,
    compute_average_amplitude_spectrum,
    estimate_effective_band,
    frequency_binned_fidelity_metrics,
)
from test.orthogonalized_local_projected_signal_leakage import (  # noqa: E402
    orthogonalized_local_projected_signal_leakage,
)
from tools.preprocessing import add_noise, normalize, denormalize  # noqa: E402
from tools.segy_read import inspect_segy, read_regular_shots  # noqa: E402
from utils.inference_utils import (  # noqa: E402
    compute_shot_metrics,
    inference_on_shots,
    save_shot_visualizations,
    select_random_shots,
)
from utils.train_utils import load_checkpoint, load_config  # noqa: E402
from utils.visualization import plot_sample  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DEFAULT_DATA_PATH = _REPO_ROOT / "data" / "SEG_45Shot_shots1-9.sgy"
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "test" / "config.yaml"
DEFAULT_CHECKPOINT_PATH = _REPO_ROOT / "result" / "checkpoint" / "best.pt"
METRIC_NAMES = ["mse", "rmse", "mae", "snr", "psnr", "ssim"]
TRACES_PER_SHOT = 201
SEED = 42
SNR_DB = -5.0


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    """Save a dict as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"  [saved] {path}")


def _save_npy(path: Path, arr: np.ndarray) -> None:
    """Save a numpy array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    print(f"  [saved] {path}  shape={arr.shape} dtype={arr.dtype}")


def _plot_olpsl_spatial_maps(
    signal_label: np.ndarray,
    alpha_map: np.ndarray,
    contribution_map: np.ndarray,
    valid_mask: np.ndarray,
    shot_idx: int,
    save_path: Path,
) -> None:
    """Plot signal label, alpha-positive, and contribution maps for one shot."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Left: ground-truth signal label (signed amplitude)
    vmax_sig = float(np.quantile(np.abs(signal_label), 0.995))
    vmin_sig = -vmax_sig if vmax_sig > 0 else -1.0
    im0 = axes[0].imshow(
        signal_label, cmap="gray", aspect="auto", origin="upper",
        vmin=vmin_sig, vmax=vmax_sig,
    )
    axes[0].set_title("Signal label (clean)")
    axes[0].set_xlabel("Trace")
    axes[0].set_ylabel("Time")
    plt.colorbar(im0, ax=axes[0])

    # Middle: local leakage ratio alpha+
    alpha_masked = np.where(valid_mask, alpha_map, np.nan)
    im1 = axes[1].imshow(alpha_masked, cmap="hot", aspect="auto", origin="upper")
    axes[1].set_title(r"Local leakage ratio $\alpha^+$")
    axes[1].set_xlabel("Trace")
    axes[1].set_ylabel("Time")
    plt.colorbar(im1, ax=axes[1])

    # Right: per-sample O-LPSL^2 contribution
    contrib_masked = np.where(valid_mask, contribution_map, np.nan)
    im2 = axes[2].imshow(contrib_masked, cmap="hot", aspect="auto", origin="upper")
    axes[2].set_title(r"O-LPSL$^2$ contribution per sample")
    axes[2].set_xlabel("Trace")
    axes[2].set_ylabel("Time")
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle(f"O-LPSL spatial maps – shot {shot_idx}")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


def _json_default(obj: Any) -> Any:
    """Fallback for JSON serialisation (numpy scalars / arrays)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def run_evaluation(
    data_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    traces_per_shot: int = TRACES_PER_SHOT,
    snr_db: float = SNR_DB,
    seed: int = SEED,
    n_viz_shots: int = 3,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the full baseline pipeline and return a summary dict."""
    output_dir = Path(output_dir)
    arrays_dir = output_dir / "arrays"
    metrics_dir = output_dir / "metrics"
    viz_dir = output_dir / "viz"
    stats_dir = output_dir / "stats"

    print("=" * 60)
    print("Baseline Evaluation Pipeline")
    print("=" * 60)
    print(f"Repo root       : {_REPO_ROOT}")
    print(f"Data path       : {data_path}")
    print(f"Config path     : {config_path}")
    print(f"Checkpoint path : {checkpoint_path}")
    print(f"Output dir      : {output_dir}")
    print(f"Device          : {device_str or 'auto'}")
    print()

    # Create output subdirectories up front so downstream plt.savefig / I/O
    # always have a valid parent path.
    for subdir in (arrays_dir, metrics_dir, viz_dir, stats_dir):
        subdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Load SEG-Y data
    # ------------------------------------------------------------------
    print("[Step 1/13] Load SEG-Y data")
    if not data_path.exists():
        raise FileNotFoundError(f"SEG-Y file not found: {data_path}")
    segy_info = inspect_segy(data_path)
    sample_interval_us = int(segy_info.get("sample_interval_us", 4000))
    dt = sample_interval_us / 1e6
    print(f"  Sample interval: {sample_interval_us} μs -> dt = {dt:.6f} s")
    clean_shots, headers = read_regular_shots(
        data_path, traces_per_shot=traces_per_shot, time_downsample=1
    )
    print(f"  Clean shots shape: {clean_shots.shape}, dtype: {clean_shots.dtype}")
    _save_npy(arrays_dir / "clean_shots.npy", clean_shots)
    if headers:
        _save_json(stats_dir / "headers_summary.json", {k: str(v) for k, v in headers.items()})

    # ------------------------------------------------------------------
    # 3. Inject Poisson noise
    # ------------------------------------------------------------------
    print(f"[Step 2/13] Inject Poisson noise at SNR = {snr_db} dB")
    rng = np.random.default_rng(seed)
    noisy_shots, noise_info = add_noise(
        clean_shots, kind="poisson", snr_db=snr_db, rng=rng
    )
    print(f"  Noisy shots shape: {noisy_shots.shape}")
    _save_npy(arrays_dir / "noisy_shots.npy", noisy_shots)
    _save_json(stats_dir / "noise_info.json", {k: v.tolist() for k, v in noise_info.items()})

    # ------------------------------------------------------------------
    # 4. Load config & build model
    # ------------------------------------------------------------------
    print("[Step 3/13] Load config & build model")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = load_config(config_path)
    print(f"  Model type: {cfg['model']['type']}")

    device = torch.device(
        device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"  Device   : {device}")
    model = build_model(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params / 1e6:.2f} M)")

    # ------------------------------------------------------------------
    # 5. Load checkpoint
    # ------------------------------------------------------------------
    print("[Step 4/13] Load checkpoint")
    if checkpoint_path.exists():
        extras = load_checkpoint(checkpoint_path, model, map_location=device)
        epoch = extras.get("epoch", "unknown")
        print(f"  Loaded checkpoint from epoch {epoch}")
        _save_json(stats_dir / "checkpoint_info.json", extras)
    else:
        print(f"  WARNING: checkpoint not found at {checkpoint_path}; using random weights.")

    # ------------------------------------------------------------------
    # 6. Normalise (training-same domain)
    # ------------------------------------------------------------------
    print("[Step 5/13] Normalise (max_abs, shot-level)")
    norm_mode = cfg.get("preprocess", {}).get("normalize_mode", "max_abs")
    norm_scope = cfg.get("preprocess", {}).get("normalize_scope", "shot")

    noisy_norm, norm_stats = normalize(noisy_shots, mode=norm_mode, per=norm_scope)
    clean_norm, _ = normalize(
        clean_shots, mode=norm_mode, per=norm_scope, override_stats=norm_stats
    )
    print(f"  Normalised shape: {noisy_norm.shape}")
    _save_npy(arrays_dir / "noisy_norm.npy", noisy_norm)
    _save_npy(arrays_dir / "clean_norm.npy", clean_norm)
    _save_json(stats_dir / "norm_stats.json", norm_stats)

    # ------------------------------------------------------------------
    # 7. Patchify -> inference -> unpatchify
    # ------------------------------------------------------------------
    print("[Step 6/13] Patch-based inference")
    patch_trace = cfg.get("preprocess", {}).get("patch_trace", 128)
    patch_time = cfg.get("preprocess", {}).get("patch_time", 256)
    patch_overlap = cfg.get("preprocess", {}).get("patch_overlap", 0.5)
    batch_size = cfg.get("data", {}).get("loader", {}).get("batch_size", 64)

    pred_norm = inference_on_shots(
        model,
        noisy_norm,
        patch_size=(patch_trace, patch_time),
        overlap=patch_overlap,
        device=device,
        batch_size=batch_size,
    )
    print(f"  Predicted normalised shape: {pred_norm.shape}")
    _save_npy(arrays_dir / "pred_norm.npy", pred_norm)

    # ------------------------------------------------------------------
    # 8. Denormalise for visualisation / export
    # ------------------------------------------------------------------
    print("[Step 7/13] Denormalise predictions")
    pred_shots = denormalize(pred_norm, norm_stats, mode=norm_mode, per=norm_scope)
    noisy_shots_denorm = denormalize(noisy_norm, norm_stats, mode=norm_mode, per=norm_scope)
    clean_shots_denorm = denormalize(clean_norm, norm_stats, mode=norm_mode, per=norm_scope)
    _save_npy(arrays_dir / "pred_shots.npy", pred_shots)
    _save_npy(arrays_dir / "noisy_shots_denorm.npy", noisy_shots_denorm)
    _save_npy(arrays_dir / "clean_shots_denorm.npy", clean_shots_denorm)

    # ------------------------------------------------------------------
    # 9. Compute metrics (normalised domain)
    # ------------------------------------------------------------------
    print("[Step 8/13] Compute metrics (normalised domain)")
    psnr_peak = 1.0          # max_abs [-1, 1] -> peak |x| = 1.0
    ssim_data_range = 2.0    # max_abs [-1, 1] -> range = 2.0

    per_shot, mean_metrics = compute_shot_metrics(
        pred_norm,
        clean_norm,
        metric_names=METRIC_NAMES,
        psnr_peak=psnr_peak,
        ssim_data_range=ssim_data_range,
    )
    print("  Mean metrics:")
    for name in METRIC_NAMES:
        print(f"    {name:5s}: {mean_metrics[name]:.4f}")

    _save_json(metrics_dir / "mean_metrics.json", mean_metrics)
    np.savez(
        metrics_dir / "per_shot_metrics.npz",
        **{name: per_shot[name] for name in METRIC_NAMES},
    )
    print(f"  [saved] {metrics_dir / 'per_shot_metrics.npz'}")

    # ------------------------------------------------------------------
    # 10. EB-WSE: Energy-Binned Weak Signal Evaluation
    # ------------------------------------------------------------------
    print("[Step 9/13] Compute EB-WSE (Energy-Binned Weak Signal Evaluation)")
    ebwse_all = {}
    for shot_idx in range(clean_norm.shape[0]):
        ebwse_shot = energy_binned_weak_signal_metrics(
            reference=clean_norm[shot_idx],
            prediction=pred_norm[shot_idx],
            bins=((5, 20), (20, 40), (40, 70), (70, 100)),
            smooth_sigma=1.0,
            eps=1e-8,
        )
        for bin_key, vals in ebwse_shot.items():
            if bin_key not in ebwse_all:
                ebwse_all[bin_key] = {
                    "NE": [], "SNR": [],
                    "num_samples": [], "ratio_to_total": [], "energy_mean": [],
                    "mean_ref_sq": [], "mean_pred_sq": [], "mean_err_sq": [],
                }
            ebwse_all[bin_key]["NE"].append(vals["NE"])
            ebwse_all[bin_key]["SNR"].append(vals["SNR"])
            ebwse_all[bin_key]["num_samples"].append(vals["num_samples"])
            ebwse_all[bin_key]["ratio_to_total"].append(vals["ratio_to_total"])
            ebwse_all[bin_key]["energy_mean"].append(vals["energy_mean"])
            ebwse_all[bin_key]["mean_ref_sq"].append(vals.get("mean_ref_sq", np.nan))
            ebwse_all[bin_key]["mean_pred_sq"].append(vals.get("mean_pred_sq", np.nan))
            ebwse_all[bin_key]["mean_err_sq"].append(vals.get("mean_err_sq", np.nan))

    # Aggregate per-bin means across shots
    ebwse_mean = {}
    for bin_key, vals in ebwse_all.items():
        ebwse_mean[bin_key] = {
            "NE_mean": float(np.nanmean(vals["NE"])),
            "SNR_mean": float(np.nanmean(vals["SNR"])),
            "NE_std": float(np.nanstd(vals["NE"])),
            "SNR_std": float(np.nanstd(vals["SNR"])),
            "num_samples_total": int(np.sum(vals["num_samples"])),
            "ratio_to_total": float(np.nanmean(vals["ratio_to_total"])),
            "energy_mean": float(np.nanmean(vals["energy_mean"])),
            "mean_ref_sq": float(np.nanmean(vals["mean_ref_sq"])),
            "mean_pred_sq": float(np.nanmean(vals["mean_pred_sq"])),
            "mean_err_sq": float(np.nanmean(vals["mean_err_sq"])),
            "energy_percentile_range": vals.get("energy_percentile_range", (0, 0)),
        }

    print("  EB-WSE (mean across shots):")
    for bin_key in ["very_weak_5_20", "weak_20_40", "medium_40_70", "strong_70_100"]:
        if bin_key in ebwse_mean:
            m = ebwse_mean[bin_key]
            print(f"    {bin_key:20s}: NE={m['NE_mean']:.4f}±{m['NE_std']:.4f}  "
                  f"SNR={m['SNR_mean']:.2f}±{m['SNR_std']:.2f}dB  "
                  f"ratio={m['ratio_to_total']:.3%}  "
                  f"E_mean={m['energy_mean']:.2e}  "
                  f"ref_sq={m['mean_ref_sq']:.2e}  "
                  f"pred_sq={m['mean_pred_sq']:.2e}  "
                  f"err_sq={m['mean_err_sq']:.2e}")

    _save_json(metrics_dir / "ebwse_mean.json", ebwse_mean)
    np.savez(
        metrics_dir / "ebwse_per_shot.npz",
        **{f"{k}_NE": np.array(v["NE"]) for k, v in ebwse_all.items()},
        **{f"{k}_SNR": np.array(v["SNR"]) for k, v in ebwse_all.items()},
    )
    print(f"  [saved] {metrics_dir / 'ebwse_per_shot.npz'}")

    # EB-WSE bar chart (NE + SNR only)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bin_labels = ["very_weak\n5-20", "weak\n20-40", "medium\n40-70", "strong\n70-100"]
    x = np.arange(len(bin_labels))

    for ax, metric, title, ylabel, hline in zip(
        axes,
        ["NE", "SNR"],
        ["Normalized Error", "SNR (dB)"],
        ["NE (lower=better)", "SNR (higher=better)"],
        [0.0, None],
    ):
        means = [ebwse_mean.get(k, {}).get(f"{metric}_mean", np.nan) for k in
                 ["very_weak_5_20", "weak_20_40", "medium_40_70", "strong_70_100"]]
        stds = [ebwse_mean.get(k, {}).get(f"{metric}_std", 0.0) for k in
                ["very_weak_5_20", "weak_20_40", "medium_40_70", "strong_70_100"]]
        ax.bar(x, means, width=0.5, yerr=stds, capsize=4, color="steelblue", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if hline is not None:
            ax.axhline(hline, color="red", linestyle="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("EB-WSE: Per-Energy-Bin Recovery Quality (mean ± std across shots)")
    plt.tight_layout()
    ebwse_viz_path = viz_dir / "ebwse_bar_chart.png"
    plt.savefig(ebwse_viz_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {ebwse_viz_path}")

    # ------------------------------------------------------------------
    # 10. FB-FRE: Frequency-Binned Fidelity and Recovery Evaluation
    # ------------------------------------------------------------------
    print("[Step 10/13] Compute FB-FRE (Frequency-Binned Fidelity and Recovery Evaluation)")

    # Estimate effective band and derive adaptive frequency bins from the full reference volume
    effective_band = estimate_effective_band(
        clean_norm, dt=dt, axis=-1, method="threshold", rel_threshold=0.0001
    )
    auto_bands = build_auto_bands(*effective_band)
    print(f"  Effective band: {effective_band[0]:.2f} - {effective_band[1]:.2f} Hz")
    print(f"  Auto bands: {auto_bands}")

    fb_fre_all = {}
    for shot_idx in range(clean_norm.shape[0]):
        fb_fre_shot = frequency_binned_fidelity_metrics(
            reference=clean_norm[shot_idx],
            prediction=pred_norm[shot_idx],
            dt=dt,
            bands=auto_bands,
            axis=-1,
            taper_width=0.0,
            eps=1e-8,
        )
        for band_key, vals in fb_fre_shot.items():
            if band_key not in fb_fre_all:
                fb_fre_all[band_key] = {
                    "BNE": [],
                    "BER": [],
                    "BCC": [],
                    "ref_band_energy": [],
                    "pred_band_energy": [],
                    "valid": [],
                    "frequency_range": vals["frequency_range"],
                    "nyquist": vals["nyquist"],
                    "effective_band": vals["effective_band"],
                    "auto_band_ratios": vals["auto_band_ratios"],
                }
            fb_fre_all[band_key]["BNE"].append(vals["BNE"])
            fb_fre_all[band_key]["BER"].append(vals["BER"])
            fb_fre_all[band_key]["BCC"].append(vals["BCC"])
            fb_fre_all[band_key]["ref_band_energy"].append(vals["ref_band_energy"])
            fb_fre_all[band_key]["pred_band_energy"].append(vals["pred_band_energy"])
            fb_fre_all[band_key]["valid"].append(vals["valid"])

    # Aggregate per-band means across shots
    fb_fre_mean = {}
    for band_key, vals in fb_fre_all.items():
        fb_fre_mean[band_key] = {
            "BNE_mean": float(np.nanmean(vals["BNE"])),
            "BER_mean": float(np.nanmean(vals["BER"])),
            "BCC_mean": float(np.nanmean(vals["BCC"])),
            "BNE_std": float(np.nanstd(vals["BNE"])),
            "BER_std": float(np.nanstd(vals["BER"])),
            "BCC_std": float(np.nanstd(vals["BCC"])),
            "ref_band_energy_mean": float(np.nanmean(vals["ref_band_energy"])),
            "pred_band_energy_mean": float(np.nanmean(vals["pred_band_energy"])),
            "valid": bool(np.all(vals["valid"])),
            "frequency_range": vals["frequency_range"],
            "nyquist": vals["nyquist"],
            "effective_band": vals["effective_band"],
            "auto_band_ratios": vals["auto_band_ratios"],
        }

    band_order = [name for name, _ in auto_bands]
    print("  FB-FRE (mean across shots):")
    for band_key in band_order:
        if band_key in fb_fre_mean:
            m = fb_fre_mean[band_key]
            print(
                f"    {band_key:20s}: BNE={m['BNE_mean']:.4f}±{m['BNE_std']:.4f}  "
                f"BER={m['BER_mean']:.4f}±{m['BER_std']:.4f}  "
                f"BCC={m['BCC_mean']:.4f}±{m['BCC_std']:.4f}  "
                f"valid={m['valid']}  "
                f"E_ref={m['ref_band_energy_mean']:.2e}  "
                f"E_pred={m['pred_band_energy_mean']:.2e}"
            )

    _save_json(metrics_dir / "fb_fre_mean.json", fb_fre_mean)
    _save_json(
        metrics_dir / "effective_band.json",
        {
            "f_min": effective_band[0],
            "f_max": effective_band[1],
            "nyquist": 1.0 / (2.0 * dt),
            "method": "threshold",
            "rel_threshold": 0.0001,
            "auto_bands": [{"name": n, "range": r} for n, r in auto_bands],
        },
    )
    np.savez(
        metrics_dir / "fb_fre_per_shot.npz",
        **{f"{k}_BNE": np.array(v["BNE"]) for k, v in fb_fre_all.items()},
        **{f"{k}_BER": np.array(v["BER"]) for k, v in fb_fre_all.items()},
        **{f"{k}_BCC": np.array(v["BCC"]) for k, v in fb_fre_all.items()},
    )
    print(f"  [saved] {metrics_dir / 'fb_fre_per_shot.npz'}")

    # Average amplitude spectrum visualization
    freqs, mean_amp = compute_average_amplitude_spectrum(clean_norm, dt=dt, axis=-1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, mean_amp, color="black", label="Reference mean amplitude")
    peak = float(np.max(mean_amp))
    ax.axhline(peak * 0.0001, color="gray", linestyle="--", label="0.01% threshold")
    band_colors = {"low": "blue", "mid": "green", "high": "orange", "very_high": "red"}
    for name, (fmin, fmax) in auto_bands:
        ax.axvspan(
            fmin,
            fmax,
            alpha=0.15,
            color=band_colors.get(name, "steelblue"),
            label=f"{name}: {fmin:.1f}-{fmax:.1f} Hz",
        )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean amplitude")
    ax.set_title("Average amplitude spectrum and effective frequency bands")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    spectrum_path = viz_dir / "average_amplitude_spectrum.png"
    plt.savefig(spectrum_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {spectrum_path}")

    # FB-FRE bar chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    band_labels = [f"{name}\n{fmin:.0f}-{fmax:.0f}" for name, (fmin, fmax) in auto_bands]
    x = np.arange(len(band_labels))
    for ax, metric, title, ylabel in zip(
        axes,
        ["BNE", "BER", "BCC"],
        ["Band Normalized Error", "Band Energy Ratio", "Band Correlation Coefficient"],
        ["BNE (lower=better)", "BER (closer to 1=better)", "BCC (closer to 1=better)"],
    ):
        means = [
            fb_fre_mean.get(k, {}).get(f"{metric}_mean", np.nan) for k in band_order
        ]
        stds = [
            fb_fre_mean.get(k, {}).get(f"{metric}_std", 0.0) for k in band_order
        ]
        ax.bar(x, means, width=0.5, yerr=stds, capsize=4, color="steelblue", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(band_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(
            1.0 if metric in ("BER", "BCC") else 0.0, color="red", linestyle="--", alpha=0.5
        )
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("FB-FRE: Per-Frequency-Band Recovery Quality (mean ± std across shots)")
    plt.tight_layout()
    fb_fre_viz_path = viz_dir / "fb_fre_bar_chart.png"
    plt.savefig(fb_fre_viz_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {fb_fre_viz_path}")

    # ------------------------------------------------------------------
    # 11. O-LPSL: Orthogonalized Local Projected Signal Leakage
    # ------------------------------------------------------------------
    print("[Step 11/13] Compute O-LPSL (Orthogonalized Local Projected Signal Leakage)")
    olpsl_window = (31, 31)
    olpsl_separability_threshold = 0.05
    olpsl_values = []
    alpha_maps = []
    contribution_maps = []
    valid_masks = []
    olpsl_spatial_dir = viz_dir / "olpsl_spatial"
    olpsl_spatial_dir.mkdir(parents=True, exist_ok=True)
    olpsl_viz_indices = select_random_shots(
        clean_norm.shape[0], n_select=n_viz_shots, seed=seed
    )
    print(f"  O-LPSL spatial visualisation shots: {olpsl_viz_indices.tolist()}")

    for shot_idx in range(clean_norm.shape[0]):
        olpsl_shot = orthogonalized_local_projected_signal_leakage(
            input_data=noisy_norm[shot_idx],
            signal_label=clean_norm[shot_idx],
            prediction=pred_norm[shot_idx],
            prediction_type="signal",
            noise_label=noisy_norm[shot_idx] - clean_norm[shot_idx],
            window=olpsl_window,
            separability_threshold=olpsl_separability_threshold,
            return_maps=True,
        )
        olpsl_values.append(olpsl_shot["O_LPSL"])
        print(
            f"    shot {shot_idx:3d}: O-LPSL={olpsl_shot['O_LPSL']:.4f}  "
            f"valid_ratio={olpsl_shot['valid_ratio']:.3%}"
        )

        alpha_map = olpsl_shot["alpha_positive_map"]
        valid_mask = olpsl_shot["valid_mask"]
        sum_sperp2 = olpsl_shot["sum_Sperp2_map"]
        weight_sum = float(olpsl_shot["weight_sum"])

        # Per-sample contribution to O-LPSL^2.  Sum over valid samples equals O_LPSL^2.
        if weight_sum > 1e-12 and np.isfinite(weight_sum):
            contribution_map = (
                alpha_map ** 2 * sum_sperp2 * valid_mask.astype(np.float64)
            ) / weight_sum
        else:
            contribution_map = np.full_like(alpha_map, np.nan)

        alpha_maps.append(alpha_map * valid_mask.astype(np.float64))
        contribution_maps.append(contribution_map)
        valid_masks.append(valid_mask.astype(np.float64))

    # Plot spatial leakage maps only for the randomly selected shots
    for shot_idx in olpsl_viz_indices:
        _plot_olpsl_spatial_maps(
            signal_label=clean_norm[shot_idx],
            alpha_map=alpha_maps[shot_idx],
            contribution_map=contribution_maps[shot_idx],
            valid_mask=valid_masks[shot_idx].astype(bool),
            shot_idx=int(shot_idx),
            save_path=olpsl_spatial_dir / f"shot_{int(shot_idx):03d}.png",
        )

    olpsl_mean = float(np.nanmean(olpsl_values))
    olpsl_std = float(np.nanstd(olpsl_values))
    print(f"  O-LPSL (mean across shots): {olpsl_mean:.4f}±{olpsl_std:.4f}")

    # Aggregate mean spatial maps across shots (only over valid pixels)
    valid_counts = np.sum(valid_masks, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_alpha_map = np.sum(alpha_maps, axis=0) / np.maximum(valid_counts, 1.0)
        mean_alpha_map = np.where(valid_counts > 0, mean_alpha_map, np.nan)
        mean_contribution_map = np.nansum(contribution_maps, axis=0) / np.maximum(valid_counts, 1.0)
        mean_contribution_map = np.where(valid_counts > 0, mean_contribution_map, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    im0 = axes[0].imshow(mean_alpha_map, cmap="hot", aspect="auto", origin="upper")
    axes[0].set_title(r"Mean local leakage ratio $\alpha^+$")
    axes[0].set_xlabel("Trace")
    axes[0].set_ylabel("Time")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(mean_contribution_map, cmap="hot", aspect="auto", origin="upper")
    axes[1].set_title(r"Mean O-LPSL$^2$ contribution per sample")
    axes[1].set_xlabel("Trace")
    axes[1].set_ylabel("Time")
    plt.colorbar(im1, ax=axes[1])
    plt.suptitle("O-LPSL spatial maps – mean across shots")
    plt.tight_layout()
    olpsl_mean_viz_path = viz_dir / "olpsl_mean_spatial_maps.png"
    plt.savefig(olpsl_mean_viz_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {olpsl_mean_viz_path}")

    olpsl_summary = {
        "O_LPSL_mean": olpsl_mean,
        "O_LPSL_std": olpsl_std,
        "window": olpsl_window,
        "separability_threshold": olpsl_separability_threshold,
        "prediction_type": "signal",
    }
    _save_json(metrics_dir / "olpsl_summary.json", olpsl_summary)
    np.savez(
        metrics_dir / "olpsl_per_shot.npz",
        O_LPSL=np.array(olpsl_values),
        alpha_maps=np.stack(alpha_maps, axis=0),
        contribution_maps=np.stack(contribution_maps, axis=0),
        valid_masks=np.stack(valid_masks, axis=0),
    )
    print(f"  [saved] {metrics_dir / 'olpsl_per_shot.npz'}")
    _save_npy(arrays_dir / "olpsl_mean_alpha_map.npy", mean_alpha_map)
    _save_npy(arrays_dir / "olpsl_mean_contribution_map.npy", mean_contribution_map)

    # O-LPSL diagnostic maps for the first shot
    olpsl_first = orthogonalized_local_projected_signal_leakage(
        input_data=noisy_norm[0],
        signal_label=clean_norm[0],
        prediction=pred_norm[0],
        prediction_type="signal",
        noise_label=noisy_norm[0] - clean_norm[0],
        window=olpsl_window,
        separability_threshold=olpsl_separability_threshold,
        return_maps=True,
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title, cmap in zip(
        axes,
        ["alpha_positive_map", "separability_map", "valid_mask"],
        ["Alpha positive", "Separability", "Valid mask"],
        ["jet", "viridis", "gray"],
    ):
        im = ax.imshow(olpsl_first[key], cmap=cmap, aspect="auto", origin="upper")
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time")
        plt.colorbar(im, ax=ax)
    plt.suptitle("O-LPSL diagnostic maps (first shot)")
    plt.tight_layout()
    olpsl_viz_path = viz_dir / "olpsl_maps_first_shot.png"
    plt.savefig(olpsl_viz_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {olpsl_viz_path}")

    # ------------------------------------------------------------------
    # 12. Visualisation
    # ------------------------------------------------------------------
    print("[Step 12/13] Save visualisations")
    viz_indices = select_random_shots(clean_shots.shape[0], n_select=n_viz_shots, seed=seed)
    print(f"  Visualising shots: {viz_indices.tolist()}")

    # Run-level symmetric colour scale (so residuals are comparable across shots)
    concat = np.concatenate([noisy_norm, pred_norm, clean_norm], axis=0)
    vmax = float(np.quantile(np.abs(concat), 0.995))
    vmin = -vmax

    saved_paths = save_shot_visualizations(
        input_shots=noisy_norm,
        pred_shots=pred_norm,
        target_shots=clean_norm,
        indices=viz_indices,
        save_dir=viz_dir,
        title_prefix="shot",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        share_scale=True,
    )
    for p in saved_paths:
        print(f"  [saved] {p}")

    # Also save a single 4-panel figure for the first shot as a reference
    first_shot_path = viz_dir / "first_shot_4panel.png"
    plot_sample(
        input_data=noisy_norm[0],
        prediction=pred_norm[0],
        target=clean_norm[0],
        save_path=first_shot_path,
        title="First Shot (Poisson -5 dB) – Normalised Domain",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        share_scale=True,
    )
    print(f"  [saved] {first_shot_path}")

    # ------------------------------------------------------------------
    # 13. Summary
    # ------------------------------------------------------------------
    print("[Step 13/13] Done")
    summary = {
        "data_shape": list(clean_shots.shape),
        "data_path": str(data_path),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(extras.get("epoch", -1)) if checkpoint_path.exists() else -1,
        "noise": {"kind": "poisson", "snr_db": snr_db, "seed": seed},
        "normalization": {"mode": norm_mode, "scope": norm_scope},
        "inference": {
            "patch_size": [patch_trace, patch_time],
            "overlap": patch_overlap,
            "batch_size": batch_size,
            "device": str(device),
        },
        "mean_metrics": mean_metrics,
        "ebwse_mean": ebwse_mean,
        "fb_fre_mean": fb_fre_mean,
        "olpsl_summary": olpsl_summary,
        "viz_indices": viz_indices.tolist(),
        "output_dir": str(output_dir),
    }
    _save_json(output_dir / "summary.json", summary)
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name in METRIC_NAMES:
        print(f"  {name:5s}: {mean_metrics[name]:.4f}")
    print(f"\n  O-LPSL (mean across shots): {olpsl_mean:.4f}±{olpsl_std:.4f}")
    print("\n  EB-WSE (mean across shots):")
    for bin_key in ["very_weak_5_20", "weak_20_40", "medium_40_70", "strong_70_100"]:
        if bin_key in ebwse_mean:
            m = ebwse_mean[bin_key]
            print(f"    {bin_key:20s}: NE={m['NE_mean']:.4f}  "
                  f"SNR={m['SNR_mean']:.2f}dB  "
                  f"ratio={m['ratio_to_total']:.3%}  E_mean={m['energy_mean']:.2e}  "
                  f"ref_sq={m['mean_ref_sq']:.2e}  err_sq={m['mean_err_sq']:.2e}")
    print("\n  FB-FRE (mean across shots):")
    for band_key in ["low_5_15", "mid_15_35", "high_35_60", "very_high_60_80"]:
        if band_key in fb_fre_mean:
            m = fb_fre_mean[band_key]
            print(f"    {band_key:20s}: BNE={m['BNE_mean']:.4f}  BER={m['BER_mean']:.4f}  "
                  f"BCC={m['BCC_mean']:.4f}  valid={m['valid']}")
    print(f"\nAll outputs saved to: {output_dir}")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baseline denoising evaluation on SEG-C3 data."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to SEG-Y file (default: data/SEG_45Shot_shots1-9.sgy).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config (default: configs/test/config.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to model checkpoint (default: result/checkpoint/best.pt).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_REPO_ROOT / "test" / "outputs",
        help="Directory for all outputs (default: test/outputs).",
    )
    parser.add_argument(
        "--snr_db",
        type=float,
        default=SNR_DB,
        help="Target SNR in dB for Poisson noise injection (default: -5.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for noise and visualisation sampling (default: 42).",
    )
    parser.add_argument(
        "--n_viz",
        type=int,
        default=3,
        help="Number of random shots to visualise (default: 3).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--traces_per_shot",
        type=int,
        default=TRACES_PER_SHOT,
        help="Traces per shot for SEG-Y reader (default: 201).",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    summary = run_evaluation(
        data_path=args.data,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        traces_per_shot=args.traces_per_shot,
        snr_db=args.snr_db,
        seed=args.seed,
        n_viz_shots=args.n_viz,
        device_str=args.device,
    )

    # Exit code 0 on success, non-zero on failure is handled by exceptions
    return summary


if __name__ == "__main__":
    main()
