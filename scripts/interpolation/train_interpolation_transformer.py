"""Transformer interpolation: trace-time tokenization + supervised training (DDP).

CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
    scripts/interpolation/train_interpolation_transformer.py \
    --config configs/interpolation/interpolation_transformer.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

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
    mask_traces,
    normalize,
    spherical_divergence_correction,
)
from utils import (  # noqa: E402
    TrainingLogger,
    build_loss,
    build_metrics,
    build_optimizer,
    build_scheduler,
    compute_metrics,
    count_parameters,
    default_config_relpath_for_train_script,
    destroy_distributed,
    init_distributed,
    load_config,
    load_checkpoint,
    maybe_save_best_checkpoint,
    maybe_wrap_ddp,
    save_checkpoint,
    set_seed,
    setup_experiment_dir_distributed,
    training_device,
    unwrap_ddp,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InterpolationTokenDataset(Dataset):
    """Token-level interpolation dataset for Transformer models."""

    def __init__(
        self,
        input_tokens: np.ndarray,
        target_tokens: np.ndarray,
        coords: np.ndarray,
        time_bounds: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        self.input_tokens = torch.from_numpy(input_tokens).float()
        self.target_tokens = torch.from_numpy(target_tokens).float()
        self.coords = torch.from_numpy(coords).float()
        self.time_bounds = torch.from_numpy(time_bounds).float()
        self.mask = torch.from_numpy(mask).float()

    def __len__(self) -> int:
        return self.input_tokens.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.input_tokens[idx],
            self.target_tokens[idx],
            self.coords[idx],
            self.time_bounds[idx],
            self.mask[idx],
        )


# ---------------------------------------------------------------------------
# Coordinate extraction
# ---------------------------------------------------------------------------

def _extract_coords(
    headers: Dict[str, np.ndarray],
    traces_per_shot: int,
    n_shots: int,
) -> np.ndarray:
    """Extract and normalize spatial coordinates from SEG-Y headers.

    Returns ``(n_shots, traces_per_shot, 4)`` in ``[0, 1]`` with channels
    ``[source_x, source_y, receiver_x, receiver_y]``.
    """
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
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess_shots(
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load volume, preprocess, and return ``(masked, target, per_shot_ffid, headers)``."""
    prep = cfg["preprocess"]

    data_cfg = None
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

    if "normalize" not in skip:
        shots, _ = normalize(
            shots,
            mode=str(prep.get("normalize_mode", "max_abs")),
            per=str(prep.get("normalize_scope", "global")),
        )

    mask_mode = str(prep.get("mask_mode", "uniform"))
    mask_ratio = float(prep.get("mask_ratio", 0.5))
    mask_kwargs: Dict[str, Any] = {"mode": mask_mode, "ratio": mask_ratio}
    if mask_mode == "uniform":
        mask_kwargs["uniform_stride"] = int(prep.get("uniform_stride", 2))
    masked, _ = mask_traces(shots, **mask_kwargs)

    headers: Dict[str, np.ndarray] = {}
    if data_cfg.get("path", "").lower().endswith((".sgy", ".segy")):
        from tools.segy_read import read_regular_shots

        _, headers = read_regular_shots(
            data_cfg["path"],
            traces_per_shot=int(data_cfg.get("traces_per_shot", 201)),
            time_downsample=int(data_cfg.get("time_downsample", 1)),
            return_headers=True,
        )
        per_shot_ffid = headers["FieldRecord"][:, 0]
    else:
        per_shot_ffid = np.arange(shots.shape[0])

    return masked, shots, per_shot_ffid, headers


def _build_transformer_tokens(
    masked_shots: np.ndarray,
    target_shots: np.ndarray,
    coords: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Convert shot gathers into Transformer token format.

    Returns
    -------
    input_tokens  : ``(B, L, chunk_length)``
    target_tokens : ``(B, L, chunk_length)``
    coords_tokens : ``(B, L, 4)``
    time_bounds   : ``(B, L, 2)`` normalised to ``[0, 1]``
    mask          : ``(B, L)`` — 1 = observed, 0 = missing
    chunk_info    : dict for :func:`trace_time_unchunk`
    """
    prep = cfg["preprocess"]
    chunk_length = int(prep.get("chunk_length", 256))
    overlap_ratio = float(prep.get("overlap_ratio", 0.0))

    # Chunk target to get chunk_info and time_bounds
    _, coords_ref, time_bounds_raw, chunk_info = trace_time_chunk(
        target_shots, coords, chunk_length, overlap_ratio,
    )

    # Chunk masked and target
    masked_tokens, _, _, _ = trace_time_chunk(
        masked_shots, coords, chunk_length, overlap_ratio,
    )
    target_tokens, _, _, _ = trace_time_chunk(
        target_shots, coords, chunk_length, overlap_ratio,
    )

    # Token-level mask: 1 if token has any non-zero sample
    mask = (~np.all(masked_tokens == 0, axis=-1)).astype(np.float32)

    # Normalise time_bounds to [0, 1]
    T = target_shots.shape[2]
    time_bounds = time_bounds_raw.astype(np.float32) / max(T - 1, 1)

    return (
        masked_tokens.astype(np.float32),
        target_tokens.astype(np.float32),
        coords_ref.astype(np.float32),
        time_bounds,
        mask,
        chunk_info,
    )


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch_transformer(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scheduler: Optional[Any] = None,
    grad_clip: Optional[float] = None,
    log_interval: int = 50,
    logger: Optional[Any] = None,
) -> Dict[str, float]:
    """Transformer-specific training loop with multi-argument forward."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for step, batch in enumerate(loader):
        input_tok, target_tok, coords, time_bounds, mask = [
            b.to(device, non_blocking=True) for b in batch
        ]

        optimizer.zero_grad(set_to_none=True)
        pred = model(input_tok, coords=coords, time_bounds=time_bounds, mask=mask)
        loss = loss_fn(pred, target_tok)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.detach().item())
        n_batches += 1
        if logger is not None and (step + 1) % max(1, int(log_interval)) == 0:
            logger.info(
                f"[epoch={epoch} step={step + 1}/{len(loader)}] "
                f"train_step_loss={loss.detach().item():.6g}"
            )

    if scheduler is not None:
        scheduler.step()

    mean_loss = total_loss / max(n_batches, 1)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        ws = torch.distributed.get_world_size()
        if ws > 1:
            stat = torch.tensor(
                [float(total_loss), float(n_batches)],
                device=device, dtype=torch.float64,
            )
            torch.distributed.all_reduce(stat, op=torch.distributed.ReduceOp.SUM)
            mean_loss = float(stat[0].item() / max(stat[1].item(), 1.0))

    return {"train": float(mean_loss)}


@torch.no_grad()
def evaluate_transformer(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    metrics: Optional[Dict[str, Any]],
    device: torch.device,
    *,
    distributed: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Transformer-specific evaluation in token space."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    metric_sums: Dict[str, float] = {k: 0.0 for k in (metrics or {})}

    for batch in loader:
        input_tok, target_tok, coords, time_bounds, mask = [
            b.to(device, non_blocking=True) for b in batch
        ]

        pred = model(input_tok, coords=coords, time_bounds=time_bounds, mask=mask)
        loss = loss_fn(pred, target_tok)
        total_loss += float(loss.detach().item())
        n_batches += 1

        if metrics:
            batch_metrics = compute_metrics(metrics, pred, target_tok)
            for k, v in batch_metrics.items():
                metric_sums[k] += float(v)

    if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        ws = torch.distributed.get_world_size()
        if ws > 1:
            stat = torch.tensor(
                [float(total_loss), float(n_batches)],
                device=device, dtype=torch.float64,
            )
            torch.distributed.all_reduce(stat, op=torch.distributed.ReduceOp.SUM)
            total_loss = float(stat[0].item())
            n_batches = int(stat[1].item())
            for k in list(metric_sums.keys()):
                v = torch.tensor([metric_sums[k]], device=device, dtype=torch.float64)
                torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
                metric_sums[k] = float(v.item())

    mean_loss = total_loss / max(n_batches, 1)
    mean_metrics = {k: v / max(n_batches, 1) for k, v in metric_sums.items()}
    return {"val": float(mean_loss)}, mean_metrics


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_transformer_shot(
    model: torch.nn.Module,
    masked_shots: np.ndarray,
    target_shots: np.ndarray,
    coords: np.ndarray,
    cfg: Dict[str, Any],
    shot_idx: int,
    save_path: Path,
    device: torch.device,
    title: Optional[str] = None,
) -> None:
    """Run single-shot inference, unchunk, and plot 4-panel figure."""
    from utils.visualization import plot_sample

    prep = cfg["preprocess"]
    chunk_length = int(prep.get("chunk_length", 256))
    overlap_ratio = float(prep.get("overlap_ratio", 0.0))

    model.eval()
    with torch.no_grad():
        single_masked = masked_shots[shot_idx: shot_idx + 1]
        single_coords = coords[shot_idx: shot_idx + 1]

        input_tok, _, c_tok, tb, chunk_info_single = trace_time_chunk(
            single_masked, single_coords, chunk_length, overlap_ratio,
        )
        T = target_shots.shape[2]
        tb_norm = tb.astype(np.float32) / max(T - 1, 1)
        mask = (~np.all(input_tok == 0, axis=-1)).astype(np.float32)

        t_input = torch.from_numpy(input_tok).float().to(device)
        t_coords = torch.from_numpy(c_tok).float().to(device)
        t_tb = torch.from_numpy(tb_norm).float().to(device)
        t_mask = torch.from_numpy(mask).float().to(device)

        pred_tok = model(t_input, coords=t_coords, time_bounds=t_tb, mask=t_mask)
        pred_np = pred_tok.cpu().numpy()
        recon = trace_time_unchunk(pred_np, chunk_info_single)

        plot_sample(
            input_data=single_masked[0],
            prediction=recon[0],
            target=target_shots[shot_idx],
            save_path=save_path,
            title=title or f"Transformer shot {shot_idx}",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Transformer interpolation from a single SEG-Y volume. "
            "Multi-GPU: torchrun --nproc_per_node=N ..."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_relpath_for_train_script(__file__),
        help="Path to transformer interpolation config.",
    )
    parser.add_argument(
        "--mask-mode",
        type=str,
        default="continuous",
        choices=["uniform", "random", "continuous"],
        help="Trace masking mode.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.2,
        help="Trace missing ratio in (0, 1).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    cfg.setdefault("preprocess", {})
    cfg["preprocess"]["mask_mode"] = args.mask_mode
    cfg["preprocess"]["mask_ratio"] = args.mask_ratio

    ratio_pct = int(round(args.mask_ratio * 100))
    cfg["experiment"]["name"] = f"{cfg['experiment']['name']}_{args.mask_mode}_miss{ratio_pct}"

    distributed, rank, local_rank, world_size = init_distributed()

    set_seed(int(cfg["experiment"]["seed"]))
    exp_dir = setup_experiment_dir_distributed(cfg, rank, distributed, base_dir=_REPO_ROOT)
    device = training_device(cfg, distributed=distributed, local_rank=local_rank)

    # --- Data ---
    masked_shots, target_shots, per_shot_ffid, headers = _preprocess_shots(cfg)

    data_cfg = cfg["data"]["segy"]
    traces_per_shot = int(data_cfg["traces_per_shot"])
    n_shots = masked_shots.shape[0]

    if not headers:
        raise ValueError(
            "Transformer training requires SEG-Y headers for spatial coordinates. "
            "Please use a .sgy/.segy data source."
        )
    coords = _extract_coords(headers, traces_per_shot, n_shots)

    # --- Build tokens ---
    input_tokens, target_tokens, coords_tokens, time_bounds, token_mask, chunk_info = \
        _build_transformer_tokens(masked_shots, target_shots, coords, cfg)

    # --- Train / val split ---
    split = float(cfg["data"].get("test_ratio", 0.1))
    n_total = input_tokens.shape[0]
    n_test = max(1, int(round(n_total * split)))
    n_train = max(1, n_total - n_test)

    idx = np.arange(n_total)
    rng = np.random.default_rng(int(cfg["experiment"]["seed"]))
    rng.shuffle(idx)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    if test_idx.size == 0:
        test_idx = train_idx[:1]

    train_ds = InterpolationTokenDataset(
        input_tokens[train_idx], target_tokens[train_idx],
        coords_tokens[train_idx], time_bounds[train_idx], token_mask[train_idx],
    )
    val_ds = InterpolationTokenDataset(
        input_tokens[test_idx], target_tokens[test_idx],
        coords_tokens[test_idx], time_bounds[test_idx], token_mask[test_idx],
    )

    train_sampler: Optional[DistributedSampler] = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, seed=int(cfg["experiment"]["seed"]),
        )

    loader_cfg = cfg["data"].get("loader", {})
    batch_size = int(loader_cfg.get("batch_size", 4))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=int(loader_cfg.get("num_workers", 0)),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # --- Model ---
    model = build_model(cfg["model"]).to(device)
    model = maybe_wrap_ddp(model, distributed=distributed, device=device, local_rank=local_rank)
    model_type = str(cfg["model"]["type"])

    if rank == 0:
        print(f"Model: {model_type} | {count_parameters(model)}")
        print(f"Train/val shots: {n_train} / {len(test_idx)}")
        print(f"Token shape: (L={input_tokens.shape[1]}, D={input_tokens.shape[2]})")

    loss_fn = build_loss(cfg["loss"]).to(device)
    metrics = build_metrics(cfg.get("metrics", []))
    optimizer = build_optimizer(model, cfg["optim"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], int(cfg["train"]["epochs"]))

    metric_names = list(metrics.keys())
    logger: Optional[TrainingLogger] = None
    if rank == 0:
        logger = TrainingLogger(
            log_dir=exp_dir / cfg["log"].get("log_dir", "logs"),
            loss_keys=["train", "val"],
            metric_keys=[f"train_{m}" for m in metric_names] + [f"val_{m}" for m in metric_names],
            plot_interval=int(cfg["log"].get("plot_interval", 5)),
        )

    # --- Resume ---
    start_epoch = 0
    resume_path = cfg["train"].get("resume")
    if resume_path:
        start_epoch = load_checkpoint(resume_path, model, optimizer=optimizer, scheduler=scheduler)
        if rank == 0 and logger is not None:
            logger.info(f"Resumed from {resume_path} (epoch {start_epoch})")

    # --- Training loop ---
    total_epochs = int(cfg["train"]["epochs"])
    eval_interval = int(cfg["train"].get("eval_interval", 1))
    ckpt_interval = int(cfg["train"].get("ckpt_interval", 5))
    vis_interval = int(cfg["train"].get("vis_interval", 5))
    log_step = bool(cfg["train"].get("log_step", False))

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch_transformer(
            model=model, loader=train_loader, loss_fn=loss_fn,
            optimizer=optimizer, device=device, epoch=epoch,
            scheduler=scheduler,
            grad_clip=cfg["train"].get("grad_clip"),
            log_interval=int(cfg["train"].get("log_interval", 20)),
            logger=logger if log_step else None,
        )

        val_losses: Dict[str, float] = {"val": float("nan")}
        val_metrics: Dict[str, float] = {}
        train_metrics: Dict[str, float] = {n: float("nan") for n in metric_names}

        if rank == 0:
            _, train_metrics = evaluate_transformer(
                model=model, loader=val_loader,
                loss_fn=loss_fn, metrics=metrics, device=device,
            )
            if (epoch + 1) % eval_interval == 0:
                val_losses, val_metrics = evaluate_transformer(
                    model=model, loader=val_loader,
                    loss_fn=loss_fn, metrics=metrics, device=device,
                )
                best_val_loss = maybe_save_best_checkpoint(
                    exp_dir / "checkpoints" / "best.pt",
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, val_loss=val_losses["val"],
                    best_val_loss=best_val_loss,
                    extras={"config": cfg}, logger=logger,
                )

        metric_row: Dict[str, float] = {}
        for name in metric_names:
            metric_row[f"train_{name}"] = train_metrics.get(name, float("nan"))
            metric_row[f"val_{name}"] = val_metrics.get(name, float("nan"))

        if logger is not None:
            logger.log_epoch(
                epoch=epoch,
                losses={
                    "train": train_stats["train"],
                    "val": val_losses.get("val", float("nan")),
                },
                metrics=metric_row,
                extras={"lr": optimizer.param_groups[0]["lr"]},
            )

        if rank == 0 and (epoch + 1) % ckpt_interval == 0:
            save_checkpoint(
                exp_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, extras={"config": cfg},
            )

        if rank == 0 and (epoch + 1) % vis_interval == 0:
            shot_idx = int(test_idx[0]) if len(test_idx) > 0 else 0
            try:
                visualize_transformer_shot(
                    model=model,
                    masked_shots=masked_shots,
                    target_shots=target_shots,
                    coords=coords,
                    cfg=cfg,
                    shot_idx=shot_idx,
                    save_path=exp_dir / "visualizations" / f"epoch_{epoch:04d}.png",
                    device=device,
                    title=f"Transformer {model_type} epoch {epoch}",
                )
            except Exception as e:
                if logger is not None:
                    logger.info(f"Visualization failed: {e}")

    elapsed = time.time() - start_time
    if logger is not None:
        logger.info(
            f"Transformer {model_type} training finished in "
            f"{elapsed:.2f}s ({elapsed / 60:.2f} min)."
        )
        logger.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
