"""Shared training entry for first-break mask segmentation scripts."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, TextIO

import numpy as np
import torch

try:  # works when imported as a package
    from .first_break_data import (
        build_first_break_loaders,
        summarize_first_break_index,
    )
    from .first_break_losses import build_first_break_loss
    from .first_break_metrics import build_first_break_metrics
except ImportError:  # works when the training script is executed as a file
    from first_break_data import (  # type: ignore
        build_first_break_loaders,
        summarize_first_break_index,
    )
    from first_break_losses import build_first_break_loss  # type: ignore
    from first_break_metrics import build_first_break_metrics  # type: ignore

from model.first_break_picking import build_model
from utils import (
    TrainingLogger,
    barrier_if_distributed,
    build_optimizer,
    build_scheduler,
    default_config_relpath_for_train_script,
    destroy_distributed,
    init_distributed,
    load_checkpoint,
    load_config,
    maybe_save_best_checkpoint,
    maybe_wrap_ddp,
    sampler_set_epoch,
    save_checkpoint,
    set_seed,
    setup_experiment_dir_distributed,
    training_device,
    unwrap_ddp,
)


def parse_args(script_file: str) -> argparse.Namespace:
    default_config = default_config_relpath_for_train_script(script_file)
    default_path = Path(default_config)
    seed42_config = str(default_path.with_name(f"{default_path.stem}_seed42{default_path.suffix}"))
    repo_root = Path(script_file).resolve().parents[2]
    if (repo_root / seed42_config).exists():
        default_config = seed42_config

    parser = argparse.ArgumentParser(
        description="Train first-break picking as binary mask segmentation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to first-break picking YAML config.",
    )
    return parser.parse_args()


def _squeeze_sample(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    if arr.ndim == 4:
        return arr[0, 0]
    if arr.ndim == 3:
        return arr[0]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 2D/3D/4D sample, got shape {arr.shape}.")


def _symmetric_clip(arr: np.ndarray, q: float = 0.99) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    v = float(np.quantile(np.abs(finite), q))
    return v if v > 0 else float(np.max(np.abs(finite))) or 1.0


def _first_pick(
    mask: np.ndarray,
    threshold: float = 0.5,
    valid_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    binary = mask > threshold
    if valid_mask is not None:
        binary = binary & valid_mask
    valid = binary.any(axis=1)
    pick = np.argmax(binary, axis=1)
    return pick.astype(np.float32), valid


def _unpack_first_break_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        x, y, target_pick = batch
        return x, y, target_pick
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        x, y = batch
        return x, y, None
    raise ValueError("first-break loaders must yield (input, mask, target_pick).")


def _with_progress(
    iterable: Iterable[Any],
    *,
    enabled: bool,
    desc: str,
    total: Optional[int] = None,
) -> Iterable[Any]:
    """Wrap an iterable with tqdm when available."""
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, leave=False)


def _safe_csv_float(value: float) -> float:
    return value if math.isfinite(value) else float("nan")


def _plot_step_loss_curve(
    global_steps: list[int],
    losses: list[float],
    save_path: Path,
    *,
    log_y: bool = False,
) -> None:
    """Plot per-step training loss against global optimizer step."""
    import matplotlib.pyplot as plt

    steps = np.asarray(global_steps, dtype=np.int64)
    values = np.asarray(losses, dtype=np.float64)
    valid = np.isfinite(values)

    fig, ax = plt.subplots(figsize=(8, 4))
    if steps.size > 0 and np.any(valid):
        ax.plot(steps[valid], values[valid], linewidth=1.0, label="train_step")
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    if log_y:
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_title("Step Loss (log scale)")
    else:
        ax.set_title("Step Loss")
    ax.set_xlabel("global step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


class StepLossLogger:
    """Append per-train-step loss rows and refresh step-level loss curves."""

    _columns = [
        "epoch",
        "step",
        "global_step",
        "loss",
        "lr",
        "step_time_sec",
        "epoch_elapsed_sec",
    ]

    def __init__(self, log_dir: Path, *, flush_interval: int = 50) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.flush_interval = max(1, int(flush_interval))
        self._path = self.log_dir / "step_loss_history.csv"
        self._curve_path = self.log_dir / "step_loss_curve.png"
        self._curve_log_path = self.log_dir / "step_loss_curve_log.png"
        self._global_steps: list[int] = []
        self._losses: list[float] = []
        self._rows_since_flush = 0
        self._dirty = False
        self._closed = False
        self._file: Optional[TextIO] = None
        self._writer: Optional[csv.DictWriter[str]] = None

        self._rehydrate_history()
        self._open()

    @property
    def next_global_step(self) -> int:
        return (max(self._global_steps) + 1) if self._global_steps else 1

    def _rehydrate_history(self) -> None:
        if not self._path.exists() or self._path.stat().st_size == 0:
            return
        with self._path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    self._global_steps.append(int(row["global_step"]))
                    self._losses.append(float(row["loss"]))
                except (KeyError, TypeError, ValueError):
                    continue

    def _open(self) -> None:
        existed = self._path.exists() and self._path.stat().st_size > 0
        self._file = self._path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self._columns)
        if not existed:
            self._writer.writeheader()
            self._file.flush()

    def log_step(
        self,
        *,
        epoch: int,
        step: int,
        global_step: int,
        loss: float,
        lr: float,
        step_time_sec: float,
        epoch_elapsed_sec: float,
    ) -> None:
        if self._closed:
            raise RuntimeError("StepLossLogger is closed.")
        assert self._writer is not None
        self._writer.writerow(
            {
                "epoch": int(epoch),
                "step": int(step),
                "global_step": int(global_step),
                "loss": _safe_csv_float(float(loss)),
                "lr": _safe_csv_float(float(lr)),
                "step_time_sec": _safe_csv_float(float(step_time_sec)),
                "epoch_elapsed_sec": _safe_csv_float(float(epoch_elapsed_sec)),
            }
        )
        self._global_steps.append(int(global_step))
        self._losses.append(float(loss))
        self._dirty = True
        self._rows_since_flush += 1
        if self._rows_since_flush >= self.flush_interval:
            self.flush()

    def refresh_curves(self) -> None:
        if not self._dirty:
            return
        _plot_step_loss_curve(self._global_steps, self._losses, self._curve_path)
        _plot_step_loss_curve(
            self._global_steps,
            self._losses,
            self._curve_log_path,
            log_y=True,
        )
        self._dirty = False

    def flush(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.flush()
        self._rows_since_flush = 0

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.refresh_curves()
        except Exception:
            pass
        finally:
            if self._file is not None and not self._file.closed:
                self._file.flush()
                self._file.close()
            self._closed = True


def train_one_epoch_first_break(
    model: torch.nn.Module,
    loader: Any,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scheduler: Optional[Any] = None,
    grad_clip: Optional[float] = None,
    log_interval: int = 50,
    logger: Optional[Any] = None,
    progress_bar: bool = False,
    step_loss_logger: Optional[StepLossLogger] = None,
) -> Dict[str, float]:
    """Run one first-break training epoch with ``(input, mask, target_pick)`` batches."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    epoch_start = time.perf_counter()
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    train_iter = _with_progress(
        loader,
        enabled=progress_bar,
        desc=f"epoch {epoch + 1} train",
        total=total_batches,
    )
    for step, batch in enumerate(train_iter):
        step_start = time.perf_counter()
        x, y, _ = _unpack_first_break_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_value = float(loss.detach().item())
        step_time_sec = time.perf_counter() - step_start
        epoch_elapsed_sec = time.perf_counter() - epoch_start
        total_loss += loss_value
        n_batches += 1
        mean_so_far = total_loss / max(n_batches, 1)
        lr = float(optimizer.param_groups[0]["lr"])

        if hasattr(train_iter, "set_postfix"):
            train_iter.set_postfix(
                loss=f"{loss_value:.4g}",
                avg_loss=f"{mean_so_far:.4g}",
                lr=f"{lr:.3g}",
            )
        if step_loss_logger is not None:
            if total_batches is not None:
                global_step = epoch * int(total_batches) + step + 1
            else:
                global_step = step_loss_logger.next_global_step
            step_loss_logger.log_step(
                epoch=epoch,
                step=step + 1,
                global_step=global_step,
                loss=loss_value,
                lr=lr,
                step_time_sec=step_time_sec,
                epoch_elapsed_sec=epoch_elapsed_sec,
            )
        if logger is not None and (step + 1) % max(1, int(log_interval)) == 0:
            logger.info(
                f"[epoch={epoch} step={step + 1}/{len(loader)}] "
                f"train_step_loss={loss_value:.6g}"
            )

    if step_loss_logger is not None:
        step_loss_logger.flush()
    if scheduler is not None:
        scheduler.step()
    mean_loss = total_loss / max(n_batches, 1)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        ws = torch.distributed.get_world_size()
        if ws > 1:
            stat = torch.tensor(
                [float(total_loss), float(n_batches)],
                device=device,
                dtype=torch.float64,
            )
            torch.distributed.all_reduce(stat, op=torch.distributed.ReduceOp.SUM)
            mean_loss = float(stat[0].item() / max(stat[1].item(), 1.0))
    return {"train": float(mean_loss)}


@torch.no_grad()
def evaluate_first_break(
    model: torch.nn.Module,
    loader: Any,
    loss_fn: torch.nn.Module,
    metrics: Optional[Dict[str, Any]],
    device: torch.device,
    *,
    desc: str = "eval",
    progress_bar: bool = False,
    loss_key: str = "val",
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Evaluate first-break losses and metrics, skipping NaN metric batches."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    metric_sums: Dict[str, float] = {k: 0.0 for k in (metrics or {})}
    metric_counts: Dict[str, int] = {k: 0 for k in (metrics or {})}
    eval_iter = _with_progress(
        loader,
        enabled=progress_bar,
        desc=desc,
        total=len(loader) if hasattr(loader, "__len__") else None,
    )
    for batch in eval_iter:
        x, y, target_pick = _unpack_first_break_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if target_pick is not None:
            target_pick = target_pick.to(device, non_blocking=True)

        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += float(loss.detach().item())
        n_batches += 1

        for name, metric in (metrics or {}).items():
            value = float(metric(pred, y, target_pick))
            if np.isfinite(value):
                metric_sums[name] += value
                metric_counts[name] += 1

    losses = {loss_key: float(total_loss / max(n_batches, 1))}
    out_metrics = {
        name: float(metric_sums[name] / metric_counts[name])
        if metric_counts[name] > 0
        else float("nan")
        for name in metric_sums
    }
    return losses, out_metrics


def _write_final_test_metrics(
    path: Path,
    *,
    best_epoch: int,
    test_loss: float,
    metrics: Dict[str, float],
    metric_names: Iterable[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["split", "best_epoch", "loss", *metric_names]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        row = {
            "split": "test",
            "best_epoch": int(best_epoch),
            "loss": _safe_csv_float(float(test_loss)),
        }
        for name in metric_names:
            row[name] = _safe_csv_float(float(metrics.get(name, float("nan"))))
        writer.writerow(row)


def _format_final_test_summary(
    *,
    best_epoch: int,
    test_loss: float,
    metrics: Dict[str, float],
    metric_names: Iterable[str],
) -> str:
    parts = [f"best_epoch={best_epoch}", f"loss={test_loss:.6g}"]
    for name in metric_names:
        value = float(metrics.get(name, float("nan")))
        parts.append(f"{name}={value:.6g}" if math.isfinite(value) else f"{name}=nan")
    return "Final test: " + ", ".join(parts)


def visualize_first_break_sample(
    model: torch.nn.Module,
    loader: Any,
    save_path: Path,
    device: torch.device,
    *,
    title: str,
    threshold: float = 0.5,
    seed: Optional[int] = None,
) -> None:
    """Save input/probability/target/overlay diagnostic for one random patch."""
    import matplotlib.pyplot as plt

    dataset = getattr(loader, "dataset", None)
    if dataset is None or len(dataset) == 0:
        raise ValueError("visualize_first_break_sample requires a non-empty dataset.")

    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, len(dataset)))
    x, y, target_pick = dataset[idx]
    sample_desc = ""
    describe_ref = getattr(dataset, "describe_ref", None)
    if callable(describe_ref):
        sample_desc = " | " + str(describe_ref(idx))
    x_batch = x.unsqueeze(0).to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prob = torch.sigmoid(model(x_batch))
    finally:
        if was_training:
            model.train()

    x2d = _squeeze_sample(x_batch)
    y2d = _squeeze_sample(y.unsqueeze(0))
    p2d = _squeeze_sample(prob)
    v = _symmetric_clip(x2d)

    target_pick_np = target_pick.detach().cpu().numpy()
    target_valid = target_pick_np >= 0
    valid_pixels = y2d >= 0
    pred_pick, pred_valid = _first_pick(p2d, threshold=threshold, valid_mask=valid_pixels)
    traces = np.arange(x2d.shape[0])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        ("input", x2d, "gray", -v, v),
        ("probability", p2d, "magma", 0.0, 1.0),
        ("target", np.where(valid_pixels, y2d, np.nan), "gray", 0.0, 1.0),
    ]
    for ax, (name, arr, cmap, vmin, vmax) in zip(axes[:3], panels):
        im = ax.imshow(arr.T, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(name)
        ax.set_xlabel("trace")
        ax.set_ylabel("time")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[3].imshow(x2d.T, cmap="gray", vmin=-v, vmax=v, aspect="auto")
    axes[3].plot(traces[target_valid], target_pick_np[target_valid], color="lime", linewidth=1.0, label="target")
    axes[3].plot(traces[pred_valid], pred_pick[pred_valid], color="red", linewidth=1.0, label="pred")
    axes[3].set_title("overlay")
    axes[3].set_xlabel("trace")
    axes[3].set_ylabel("time")
    axes[3].legend(loc="upper right", fontsize=8)

    fig.suptitle(f"{title} | sample idx={idx}{sample_desc}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_training(script_file: str) -> None:
    args = parse_args(script_file)
    cfg = load_config(args.config)

    distributed, rank, local_rank, world_size = init_distributed()
    set_seed(int(cfg["experiment"]["seed"]))
    exp_dir = setup_experiment_dir_distributed(cfg, rank, distributed, base_dir=Path(script_file).resolve().parents[2])
    device = training_device(cfg, distributed=distributed, local_rank=local_rank)

    train_loader, val_loader, test_loader, train_sampler, eval_train_loader, index = build_first_break_loaders(
        cfg,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
    )

    model = build_model(cfg["model"]).to(device)
    model = maybe_wrap_ddp(
        model,
        distributed=distributed,
        device=device,
        local_rank=local_rank,
    )
    model_type = str(cfg["model"]["type"])
    eval_model = unwrap_ddp(model) if distributed else model
    loss_fn = build_first_break_loss(cfg["loss"]).to(device)
    metrics = build_first_break_metrics(cfg.get("metrics", []))
    optimizer = build_optimizer(model, cfg["optim"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], int(cfg["train"]["epochs"]))

    metric_names = list(metrics.keys())
    logger: Optional[TrainingLogger] = None
    step_loss_logger: Optional[StepLossLogger] = None
    log_dir = exp_dir / cfg["log"].get("log_dir", "logs")
    if rank == 0:
        logger = TrainingLogger(
            log_dir=log_dir,
            loss_keys=["train", "val"],
            metric_keys=[f"train_{m}" for m in metric_names] + [f"val_{m}" for m in metric_names],
            plot_interval=int(cfg["log"].get("plot_interval", 5)),
        )
        step_loss_logger = StepLossLogger(
            log_dir=log_dir,
            flush_interval=int(cfg["train"].get("log_interval", 20)),
        )
        logger.info(summarize_first_break_index(index))
        logger.info(
            f"Model {model_type} | train/val/test patches: "
            f"{len(train_loader.dataset)} / {len(val_loader.dataset)} / {len(test_loader.dataset)}"
        )

    start_epoch = 0
    resume = cfg.get("train", {}).get("resume")
    if resume:
        extras = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(extras.get("epoch", -1)) + 1
        if logger is not None:
            logger.info(f"Resumed from {resume} at epoch {start_epoch}.")

    total_epochs = int(cfg["train"]["epochs"])
    eval_interval = int(cfg["train"].get("eval_interval", 1))
    ckpt_interval = int(cfg["train"].get("ckpt_interval", 5))
    vis_interval = int(cfg["train"].get("vis_interval", 5))
    log_step = bool(cfg["train"].get("log_step", False))
    threshold = float(cfg["data"].get("prediction_threshold", 0.5))
    progress_bar = bool(cfg["train"].get("progress_bar", False))
    vis_seed = int(cfg["train"].get("vis_seed", cfg["experiment"]["seed"]))

    best_val_loss = float("inf")
    start_time = time.time()
    for epoch in range(start_epoch, total_epochs):
        epoch_wall_start = time.time()
        sampler_set_epoch(train_sampler, epoch)
        train_stats = train_one_epoch_first_break(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scheduler=scheduler,
            grad_clip=cfg["train"].get("grad_clip"),
            log_interval=int(cfg["train"].get("log_interval", 20)),
            logger=logger if log_step else None,
            progress_bar=progress_bar and rank == 0,
            step_loss_logger=step_loss_logger,
        )

        val_losses = {"val": float("nan")}
        val_metrics: Dict[str, float] = {}
        train_metrics: Dict[str, float] = {n: float("nan") for n in metric_names}

        if rank == 0 and eval_train_loader is not None:
            _, train_metrics = evaluate_first_break(
                model=eval_model,
                loader=eval_train_loader,
                loss_fn=loss_fn,
                metrics=metrics,
                device=device,
                desc=f"epoch {epoch + 1} eval-train",
                progress_bar=progress_bar,
            )
            if (epoch + 1) % eval_interval == 0:
                val_losses, val_metrics = evaluate_first_break(
                    model=eval_model,
                    loader=val_loader,
                    loss_fn=loss_fn,
                    metrics=metrics,
                    device=device,
                    desc=f"epoch {epoch + 1} val",
                    progress_bar=progress_bar,
                )
                best_val_loss = maybe_save_best_checkpoint(
                    exp_dir / "checkpoints" / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    val_loss=val_losses["val"],
                    best_val_loss=best_val_loss,
                    extras={"config": cfg},
                    logger=logger,
                )

        metric_row: Dict[str, float] = {}
        for name in metric_names:
            metric_row[f"train_{name}"] = train_metrics.get(name, float("nan"))
            metric_row[f"val_{name}"] = val_metrics.get(name, float("nan"))

        if logger is not None:
            epoch_sec = time.time() - epoch_wall_start
            elapsed_min = (time.time() - start_time) / 60.0
            logger.log_epoch(
                epoch=epoch,
                losses={
                    "train": train_stats["train"],
                    "val": val_losses.get("val", float("nan")),
                },
                metrics=metric_row,
                extras={
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch_sec": epoch_sec,
                    "elapsed_min": elapsed_min,
                },
            )
        if step_loss_logger is not None:
            step_loss_logger.refresh_curves()

        if rank == 0 and (epoch + 1) % ckpt_interval == 0:
            save_checkpoint(
                exp_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                extras={"config": cfg},
            )

        if rank == 0 and (epoch + 1) % vis_interval == 0:
            visualize_first_break_sample(
                model=eval_model,
                loader=val_loader,
                save_path=exp_dir / "visualizations" / f"epoch_{epoch:04d}.png",
                device=device,
                title=f"First-break {model_type} epoch {epoch}",
                threshold=threshold,
                seed=vis_seed,
            )

        barrier_if_distributed()

    if rank == 0 and logger is not None:
        best_path = exp_dir / "checkpoints" / "best.pt"
        if best_path.exists():
            best_extras = load_checkpoint(
                best_path,
                model=eval_model,
                optimizer=None,
                scheduler=None,
                map_location=device,
            )
            best_epoch = int(best_extras.get("epoch", -1))
            test_losses, test_metrics = evaluate_first_break(
                model=eval_model,
                loader=test_loader,
                loss_fn=loss_fn,
                metrics=metrics,
                device=device,
                desc="final test",
                progress_bar=progress_bar,
                loss_key="test",
            )
            test_loss = test_losses.get("test", float("nan"))
            _write_final_test_metrics(
                log_dir / "final_test_metrics.csv",
                best_epoch=best_epoch,
                test_loss=test_loss,
                metrics=test_metrics,
                metric_names=metric_names,
            )
            logger.info(
                _format_final_test_summary(
                    best_epoch=best_epoch,
                    test_loss=test_loss,
                    metrics=test_metrics,
                    metric_names=metric_names,
                )
            )
        else:
            logger.info(f"Final test skipped: best checkpoint not found at {best_path}.")

    elapsed = time.time() - start_time
    if logger is not None:
        logger.info(
            f"First-break {model_type} training finished in "
            f"{elapsed:.2f}s ({elapsed / 60:.2f} min)."
        )
    if step_loss_logger is not None:
        step_loss_logger.close()
    if logger is not None:
        logger.close()
    destroy_distributed()


__all__ = ["run_training"]
