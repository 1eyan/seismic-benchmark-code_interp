"""Consolidate batch best.pt inference outputs into summary CSVs and per-experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

_NAME_RE = re.compile(
    r"^interp_(.+)_seed([0-9]+)_(uniform|random|continuous)_miss(.+)$"
)

CORE_KEYS = ["snr", "psnr", "ssim", "mae", "mse", "rmse", "inference_time_seconds"]

REQUIRED_FILES = [
    "inference/INFERENCE_SUCCESS",
    "inference/metrics_summary.json",
    "inference/metrics_per_shot.csv",
    "inference/npy/pred_shots.npy",
    "inference/npy/target_shots.npy",
    "inference/npy/input_shots.npy",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str, repo_root: Path = _REPO_ROOT) -> Path:
    """Resolve a CLI path relative to the repo root when not absolute."""
    p = Path(path)
    return p if p.is_absolute() else (repo_root / p).resolve()


def _fmt(value: Any) -> Any:
    """Serialize list/dict metric values for CSV output."""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def _parse_name(name: str) -> Tuple[str, str, str, str]:
    """Split an experiment dir name into model, seed, mask mode, and mask value."""
    m = _NAME_RE.match(name)
    if m is None:
        return name, "", "", ""
    return m.group(1), m.group(2), m.group(3), m.group(4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify batch inference outputs and centralized params/ + configs/ "
            "files, and emit summary.csv / manifest.csv / failures.txt."
        )
    )
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--collect-root", type=str, default="collected")
    parser.add_argument("--summary-out", type=str, default=None)
    parser.add_argument("--manifest-out", type=str, default=None)
    parser.add_argument("--failures-out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = _resolve(args.results_root)
    collect_root = _resolve(args.collect_root)
    summary_out = _resolve(args.summary_out or str(collect_root / "summary.csv"))
    manifest_out = _resolve(args.manifest_out or str(collect_root / "manifest.csv"))
    failures_out = _resolve(args.failures_out or str(collect_root / "failures.txt"))

    if not results_root.is_dir():
        print(f"ERROR: results root not found: {results_root}", file=sys.stderr)
        sys.exit(1)

    exp_dirs = sorted(
        p
        for p in results_root.iterdir()
        if p.is_dir() and (p / "checkpoints" / "best.pt").is_file()
    )

    rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, str]] = []
    failures: List[str] = []
    all_metric_keys: Dict[str, None] = {}

    for exp_dir in exp_dirs:
        name = exp_dir.name
        src_cfg = exp_dir / "config.yaml"
        src_ckpt = exp_dir / "checkpoints" / "best.pt"
        inference_dir = collect_root / name / "inference"
        params_pt = collect_root / "params" / f"{name}.pt"
        config_yaml = collect_root / "configs" / f"{name}.yaml"

        for label, flat_path in (
            ("params_pt", params_pt),
            ("config_yaml", config_yaml),
        ):
            if not flat_path.is_file():
                print(
                    f"WARNING: {name}: centralized {label} missing: {flat_path}",
                    file=sys.stderr,
                )

        missing = [
            f for f in REQUIRED_FILES if not (collect_root / name / f).is_file()
        ]
        if missing:
            reason = "missing outputs: " + ", ".join(missing)
            failures.append(f"{name}: {reason}")
            manifest_rows.append(
                {
                    "experiment": name,
                    "status": "failed",
                    "reason": reason,
                    "source_config": str(src_cfg),
                    "source_checkpoint": str(src_ckpt),
                    "params_pt": str(params_pt),
                    "config_yaml": str(config_yaml),
                    "inference_dir": str(inference_dir),
                }
            )
            continue

        model, seed, mask_mode, mask_value = _parse_name(name)
        model_type = ""
        try:
            model_type = str(yaml.safe_load(src_cfg.read_text())["model"]["type"])
        except (OSError, KeyError, TypeError, yaml.YAMLError):
            pass
        if model_type and model and model not in model_type and model_type not in model:
            print(
                f"WARNING: {name}: parsed model '{model}' does not match "
                f"config model.type '{model_type}'",
                file=sys.stderr,
            )

        summary_path = inference_dir / "metrics_summary.json"
        try:
            metrics: Dict[str, Any] = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            failures.append(f"{name}: unreadable metrics_summary.json")
            manifest_rows.append(
                {
                    "experiment": name,
                    "status": "failed",
                    "reason": "unreadable metrics_summary.json",
                    "source_config": str(src_cfg),
                    "source_checkpoint": str(src_ckpt),
                    "params_pt": str(params_pt),
                    "config_yaml": str(config_yaml),
                    "inference_dir": str(inference_dir),
                }
            )
            continue

        row: Dict[str, Any] = {
            "experiment": name,
            "model": model,
            "seed": seed,
            "mask_mode": mask_mode,
            "mask_value": mask_value,
        }
        for key, value in metrics.items():
            row[key] = value
            all_metric_keys[key] = None
        rows.append(row)
        manifest_rows.append(
            {
                "experiment": name,
                "status": "ok",
                "reason": "",
                "source_config": str(src_cfg),
                "source_checkpoint": str(src_ckpt),
                "params_pt": str(params_pt),
                "config_yaml": str(config_yaml),
                "inference_dir": str(inference_dir),
            }
        )

    columns = (
        ["experiment", "model", "seed", "mask_mode", "mask_value"]
        + [k for k in CORE_KEYS if k in all_metric_keys]
        + sorted(k for k in all_metric_keys if k.startswith("eb_wse_"))
        + sorted(k for k in all_metric_keys if k.startswith("fb_fre_"))
        + sorted(
            k
            for k in all_metric_keys
            if k not in CORE_KEYS
            and not k.startswith("eb_wse_")
            and not k.startswith("fb_fre_")
        )
    )

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k, "")) for k in columns})

    manifest_columns = [
        "experiment",
        "status",
        "reason",
        "source_config",
        "source_checkpoint",
        "params_pt",
        "config_yaml",
        "inference_dir",
    ]
    with manifest_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_columns)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({k: row.get(k, "") for k in manifest_columns})

    with failures_out.open("w", encoding="utf-8") as f:
        f.write("\n".join(failures) + ("\n" if failures else ""))

    print(f"Experiments with best.pt : {len(exp_dirs)}")
    print(f"Collected (ok)           : {len(rows)}")
    print(f"Failed / missing         : {len(failures)}")
    print(f"Summary                  : {summary_out}")
    print(f"Manifest                 : {manifest_out}")
    print(f"Failures                 : {failures_out}")
    print(f"Collected at             : {datetime.now().isoformat(timespec='seconds')}")

    if failures:
        print("\nFailures:", file=sys.stderr)
        for item in failures:
            print(f"  {item}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
