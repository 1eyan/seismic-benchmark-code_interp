"""Transformer interpolation inference: full-shot reconstruction, per-shot metrics, and viz.

Example (single GPU):
    python scripts/interpolation/inference_interpolation_transformer.py \
        --config configs/interpolation/interpolation_transformer.yaml \
        --checkpoint results/interp_transformer_v11_encdec/checkpoints/best.pt \
        --output-dir results/interp_transformer_v11_encdec/inference \
        --n-viz-shots 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "model").is_dir() and (p / "utils").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError(
        "Cannot find repo root (a directory containing both ``model/`` and ``utils/``)."
    )
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.interpolation import build_model  # noqa: E402
from tools.array_io import load_volume  # noqa: E402
from tools.patching import trace_time_chunk, trace_time_unchunk  # noqa: E402
from tools.preprocessing import (  # noqa: E402
    denormalize,
    inverse_spherical_divergence_correction,
    mask_traces,
    normalize,
    spherical_divergence_correction,
)
from utils import count_parameters, load_checkpoint, load_config  # noqa: E402
from utils.inference_utils import (  # noqa: E402
    compute_shot_metrics,
    save_shot_visualizations,
    select_random_shots,
)
from utils.metrics import format_metric_value  # noqa: E402


# ---------------------------------------------------------------------------
# Coordinate extraction (same as training script)
# ---------------------------------------------------------------------------

def _extract_coords(
    headers: Dict[str, np.ndarray],
    traces_per_shot: int,
    n_shots: int,
) -> np.ndarray:
    """Extract and normalise spatial coordinates from SEG-Y headers."""
    sx = headers["SourceX"].reshape(n_shots, traces_per_shot).astype(np.float32)
    sy = headers["SourceY"].reshape(n_shots, traces_per_shot).astype(np.float32)
    rx = headers["GroupX"].reshape(n_shots, traces_per_shot).astype(np.float32)
    ry = headers["GroupY"].reshape(n_shots, traces_per_shot).astype(np.float32)

    def _norm(v: np.ndarray) -> np.ndarray:
        vmin, vmax = float(v.min()), float(v.max())
        if vmax - vmin < 1e-8:
            return np.zeros_like(v)
        return (v - vmin) / (vmax - vmin)

    return np.stack([_norm(sx), _norm(sy), _norm(rx), _norm(ry)], axis=-1)


# ---------------------------------------------------------------------------
# Transformer inference
# ---------------------------------------------------------------------------

def inference_on_shots_transformer(
    model: torch.nn.Module,
    masked_shots: np.ndarray,
    coords: np.ndarray,
    cfg: Dict[str, Any],
    device: torch.device,
    batch_size: int = 1,
) -> np.ndarray:
    """Run Transformer inference on full shot volume via chunk/unchunk.

    Parameters
    ----------
    model       : trained Transformer model.
    masked_shots: ``(n_shots, n_traces, T)`` masked shot volume.
    coords      : ``(n_shots, n_traces, 4)`` normalised spatial coordinates.
    cfg         : config dict.
    device      : inference device.

    Returns
    -------
    pred_shots  : ``(n_shots, n_traces, T)`` predicted (reconstructed) volume.
    """
    prep = cfg["preprocess"]
    chunk_length = int(prep.get("chunk_length", 256))
    overlap_ratio = float(prep.get("overlap_ratio", 0.0))

    n_shots = masked_shots.shape[0]
    T = masked_shots.shape[2]
    T_norm = max(T - 1, 1)

    all_recon: List[np.ndarray] = []
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for i in range(0, n_shots, batch_size):
                batch_masked = masked_shots[i: i + batch_size]
                batch_coords = coords[i: i + batch_size]

                input_tok, c_tok, tb, chunk_info = trace_time_chunk(
                    batch_masked, batch_coords, chunk_length, overlap_ratio,
                )
                tb_norm = tb.astype(np.float32) / T_norm
                mask = (~np.all(input_tok == 0, axis=-1)).astype(np.float32)

                t_input = torch.from_numpy(input_tok).float().to(device)
                t_coords = torch.from_numpy(c_tok).float().to(device)
                t_tb = torch.from_numpy(tb_norm).float().to(device)
                t_mask = torch.from_numpy(mask).float().to(device)

                pred_tok = model(t_input, coords=t_coords, time_bounds=t_tb, mask=t_mask)

                pred_np = pred_tok.cpu().numpy()
                recon = trace_time_unchunk(pred_np, chunk_info)
                all_recon.append(recon)
    finally:
        if was_training:
            model.train()

    return np.concatenate(all_recon, axis=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Transformer interpolation inference. "
        "CLI arguments override config values."
    )
    parser.add_argument("--config", type=str, default="configs/interpolation/interpolation_transformer.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-viz-shots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-npy", action="store_true", default=None)
    parser.add_argument("--replace-observed", action="store_true", default=None,
                        help="Replace all traces (including observed) with model predictions.")
    parser.add_argument("--mask-mode", type=str, default=None, choices=["uniform", "random", "continuous"])
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument(
        "--continuous-missing-traces",
        type=int,
        default=None,
        help="Number of contiguous missing traces for continuous masking.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    infer_cfg = cfg.get("inference", {})

    # Resolve parameters: CLI > config.inference > defaults
    checkpoint = args.checkpoint if args.checkpoint is not None else infer_cfg.get("checkpoint")
    if checkpoint is None:
        parser = argparse.ArgumentParser()
        parser.error("--checkpoint is required (or set inference.checkpoint in config).")

    output_dir = args.output_dir if args.output_dir is not None else infer_cfg.get("output_dir")
    n_viz_shots = (
        args.n_viz_shots if args.n_viz_shots is not None
        else infer_cfg.get("n_viz_shots", 5)
    )
    seed = (
        args.seed if args.seed is not None
        else infer_cfg.get("seed", cfg["experiment"]["seed"])
    )
    device = torch.device(
        args.device if args.device is not None
        else infer_cfg.get("device", cfg["experiment"].get("device", "cpu"))
    )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU.")
            device = torch.device("cpu")
        elif device.index is None:
            device = torch.device("cuda:0")
        elif device.index >= torch.cuda.device_count():
            print(
                f"Warning: CUDA device {device} not available "
                f"(found {torch.cuda.device_count()} GPU(s)), falling back to cuda:0."
            )
            device = torch.device("cuda:0")
    print(f"Using device: {device}")
    batch_size = (
        args.batch_size if args.batch_size is not None
        else infer_cfg.get("batch_size", 1)
    )
    save_npy = (
        args.save_npy if args.save_npy is not None
        else infer_cfg.get("save_npy", False)
    )
    replace_observed = (
        args.replace_observed if args.replace_observed is not None
        else infer_cfg.get("replace_observed", False)
    )

    prep = cfg.get("preprocess", {})
    mask_mode = (
        args.mask_mode if args.mask_mode is not None
        else str(prep.get("mask_mode", "uniform"))
    )
    mask_ratio = (
        args.mask_ratio if args.mask_ratio is not None
        else float(prep.get("mask_ratio", 0.5))
    )
    continuous_missing_traces = (
        args.continuous_missing_traces if args.continuous_missing_traces is not None
        else prep.get("continuous_missing_traces")
    )

    # --- Build model and load checkpoint ---
    model = build_model(cfg["model"]).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model_type = str(cfg["model"]["type"])
    print(f"Model: {model_type} | {count_parameters(model)}")

    # --- Load volume and preprocess ---
    infer_data_cfg = infer_cfg.get("data", {})
    data_cfg = None
    for key in ("segy", "npy", "mat"):
        if key in infer_data_cfg:
            data_cfg = infer_data_cfg[key]
            break
    if data_cfg is None:
        for key in ("segy", "npy", "mat"):
            if key in cfg["data"]:
                data_cfg = cfg["data"][key]
                break
    if data_cfg is None:
        raise ValueError("No data source found in config.")

    shots = load_volume(data_cfg)
    if prep.get("max_shots") is not None:
        shots = shots[: int(prep["max_shots"])]

    skip = set(prep.get("skip", []))

    if "spherical_divergence_correction" not in skip:
        shots = spherical_divergence_correction(
            shots,
            dt=float(prep["dt"]),
            power=float(prep.get("spherical_power", 1.2)),
            t0=float(prep.get("t0", 0.0)),
        )

    stats: Optional[Dict[str, Any]] = None
    norm_mode = str(prep.get("normalize_mode", "max_abs"))
    norm_scope = str(prep.get("normalize_scope", "global"))
    if "normalize" not in skip:
        shots, stats = normalize(shots, mode=norm_mode, per=norm_scope)

    mask_kwargs: Dict[str, Any] = {"mode": mask_mode, "ratio": mask_ratio}
    if mask_mode == "uniform":
        mask_kwargs["uniform_stride"] = int(prep.get("uniform_stride", 2))
    if continuous_missing_traces is not None:
        mask_kwargs["missing_traces"] = int(continuous_missing_traces)
    masked, trace_mask = mask_traces(shots, **mask_kwargs)

    shots_norm = shots
    masked_norm = masked

    # --- Coordinates ---
    n_shots = shots.shape[0]
    headers: Dict[str, np.ndarray] = {}
    if data_cfg.get("path", "").lower().endswith((".sgy", ".segy")):
        from tools.segy_read import read_regular_shots
        _, headers = read_regular_shots(
            data_cfg["path"],
            traces_per_shot=int(data_cfg.get("traces_per_shot", 201)),
            time_downsample=int(data_cfg.get("time_downsample", 1)),
            return_headers=True,
        )
    if not headers:
        raise ValueError(
            "Transformer inference requires SEG-Y headers for spatial coordinates."
        )
    coords = _extract_coords(headers, int(data_cfg.get("traces_per_shot", 201)), n_shots)

    # --- Inference ---
    infer_start = time.time()
    pred_norm = inference_on_shots_transformer(
        model=model,
        masked_shots=masked_norm,
        coords=coords,
        cfg=cfg,
        device=device,
        batch_size=batch_size,
    )
    mask_3d = trace_mask[..., None]
    if replace_observed:
        recon_norm = pred_norm
    else:
        recon_norm = np.where(mask_3d, pred_norm, shots_norm)
    infer_elapsed = time.time() - infer_start
    print(f"Inference time: {infer_elapsed:.2f}s")

    # --- Inverse preprocessing ---
    def _inverse(arr: np.ndarray) -> np.ndarray:
        if "normalize" not in skip and stats is not None:
            arr = denormalize(arr, stats, mode=norm_mode, per=norm_scope)
        if "spherical_divergence_correction" not in skip:
            arr = inverse_spherical_divergence_correction(
                arr,
                dt=float(prep["dt"]),
                power=float(prep.get("spherical_power", 1.2)),
                t0=float(prep.get("t0", 0.0)),
            )
        return arr

    # --- Metrics (normalised domain) ---
    metric_cfg = cfg.get("metrics", [])
    metric_names = [m["name"] for m in metric_cfg]
    if norm_mode == "max_abs":
        psnr_peak = 1.0
        ssim_data_range = 2.0
    elif norm_mode == "minmax":
        psnr_peak = 1.0
        ssim_data_range = 1.0
    else:
        psnr_peak = float(np.max(np.abs(shots_norm)))
        ssim_data_range = float(np.max(shots_norm) - np.min(shots_norm))
        if psnr_peak <= 0.0:
            psnr_peak = 1.0
        if ssim_data_range <= 0.0:
            ssim_data_range = 1.0

    for m in metric_cfg:
        if m["name"] == "psnr" and "data_range" in m.get("params", {}):
            psnr_peak = float(m["params"]["data_range"])
        elif m["name"] == "ssim" and "data_range" in m.get("params", {}):
            ssim_data_range = float(m["params"]["data_range"])

    per_shot, mean = compute_shot_metrics(
        recon_norm, shots_norm,
        metric_names=metric_names,
        psnr_peak=psnr_peak,
        ssim_data_range=ssim_data_range,
    )

    # --- Inverse preprocessing for output ---
    pred_shots = _inverse(recon_norm)
    target_shots = _inverse(shots_norm)
    input_shots = _inverse(masked_norm)

    # --- Save outputs ---
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        exp = cfg.get("experiment", {})
        out_dir = Path(exp.get("output_dir", "results")) / exp.get("name", "exp") / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "metrics_per_shot.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        header = ["shot_idx"] + list(per_shot.keys())
        f.write(",".join(header) + "\n")
        for i in range(n_shots):
            row = [str(i)] + [
                format_metric_value(k, float(per_shot[k][i])) for k in per_shot.keys()
            ]
            f.write(",".join(row) + "\n")

    summary = dict(mean)
    summary["inference_time_seconds"] = round(infer_elapsed, 3)
    summary_path = out_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if save_npy:
        npy_dir = out_dir / "npy"
        npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy_dir / "pred_shots.npy", pred_shots)
        np.save(npy_dir / "target_shots.npy", target_shots)
        np.save(npy_dir / "input_shots.npy", input_shots)
        print(f"Saved .npy files to {npy_dir}")

    # --- Visualize ---
    viz_dir = out_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    indices = select_random_shots(n_shots, n_viz_shots, seed=seed)

    vmax = float(np.quantile(np.abs(np.concatenate([
        input_shots.ravel(), pred_shots.ravel(), target_shots.ravel()
    ])), 0.995))

    save_shot_visualizations(
        input_shots=input_shots,
        pred_shots=pred_shots,
        target_shots=target_shots,
        indices=indices,
        save_dir=viz_dir,
        title_prefix=f"interp_{model_type}",
        vmin=-vmax,
        vmax=vmax,
    )

    # --- Summary ---
    print(f"Inference complete. Outputs saved to: {out_dir}")
    print(f"Visualized shots: {list(indices)}")
    print("Mean metrics (normalized domain):")
    for k, v in mean.items():
        print(f"  {k}: {format_metric_value(k, v)}")


if __name__ == "__main__":
    main()
