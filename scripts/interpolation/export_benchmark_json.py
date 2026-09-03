#!/usr/bin/env python
"""Export interpolation benchmark results to leaderboard JSON files.

Reads the "Interpolation" sheet of ``batch_evaluation_part.xlsx`` (produced by
``fill_batch_evaluation_xlsx.py``) and emits, for each model, a pair of JSON
files that follow the same schema as the first-arrival-picking reference files
``first_arrival_picking_model_<name>.json`` / ``first_arrival_picking_result_<name>.json``:

- ``interpolation_model_<name>.json``  — model metadata (single object).
- ``interpolation_result_<name>.json`` — list of ``{model_id, benchmark_id,
  scores, paper_url, code_url, date_added}``, one entry per mask scenario.

Scores are the reconstruction metrics (SNR/PSNR/SSIM/MAE/MSE/RMSE) plus the
binned EB-WSE / FB-FRE diagnostics, keyed by the lowercased workbook column
name.  ``mean±std`` cells are collapsed to their leading mean; ``—`` cells
become ``null``.  Scenario ``benchmark_id`` follows ``interp-<mode>-<param>``.

Usage::

    python scripts/interpolation/export_benchmark_json.py
    python scripts/interpolation/export_benchmark_json.py \
        --input batch_evaluation_part.xlsx --output-dir .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_URL = "https://github.com/1eyan/seismic-benchmark-code_interp/blob/main"

_DASH = "—"

# Model types excluded from the leaderboard export (kept in collected/ but no
# longer reported).
EXCLUDED_MODEL_TYPES = {"guo2023_mst"}

# Curated metadata per registered model type (sourced from
# model/interpolation/<name>_notes.md and the cited papers).
MODELS: Dict[str, Dict[str, Any]] = {
    "chai2020_unet": {
        "name": "Chai2020 UNet",
        "authors": "Xintao Chai, Genyang Tang, Shangxu Wang, Ronghua Peng, Wei Chen, Jingnan Li",
        "year": 2020,
        "paper_url": "https://doi.org/10.1109/TGRS.2019.2961015",
    },
    "li2022_caunet": {
        "name": "CA-Unet",
        "authors": "Xinze Li, Bangyu Wu, Xu Zhu, Hui Yang",
        "year": 2022,
        "paper_url": "https://doi.org/10.1109/LGRS.2021.3128511",
    },
    "liu2022_wrdl": {
        "name": "WRDL",
        "authors": "Naihao Liu, Lukun Wu, Jiale Wang, Hao Wu, Jinghuai Gao, Dehua Wang",
        "year": 2022,
        "paper_url": "https://doi.org/10.1109/TGRS.2022.3152984",
    },
    "pan2020_pconv_unet": {
        "name": "PConv U-Net",
        "authors": "Shulin Pan, Kai Chen, Jingyi Chen, Ziyu Qin, Qinghui Cui, Jing Li",
        "year": 2020,
        "paper_url": "https://doi.org/10.1016/j.cageo.2020.104609",
    },
    "park2022_cfunet": {
        "name": "CFunet",
        "authors": "Junhwan Park, Jungkyun Shin, Soon Jee Seol, Joongmoo Byun",
        "year": 2022,
        "paper_url": "https://doi.org/10.1109/TGRS.2022.3190292",
    },
    "gated_transformer_v9": {
        "name": "Gated Transformer v9",
        "authors": None,
        "year": None,
        "paper_url": None,
    },
    "yu2022_anet": {
        "name": "ANet",
        "authors": "Jiaxu Yu, Bangyu Wu",
        "year": 2022,
        "paper_url": "https://doi.org/10.1109/TGRS.2021.3068279",
    },
}


def _parse_mean(cell: Any) -> Optional[float]:
    """Return the leading mean of a ``mean±std`` cell, or None for ``—``."""
    if cell is None:
        return None
    if isinstance(cell, (int, float)):
        return float(cell)
    text = str(cell).strip()
    if text in ("", _DASH, "-", "None"):
        return None
    try:
        return float(text.split("±")[0])
    except ValueError:
        return None


def _benchmark_id(method: str) -> Optional[str]:
    """Map a method label suffix to ``interp-<mode>-<param>``."""
    if "(" not in method:
        return None
    suffix = method.split("(", 1)[1].rstrip(")").strip()
    parts = suffix.split()
    if len(parts) != 2:
        return None
    mode, param = parts
    return f"interp-{mode}-{param}"


def _model_type(method: str) -> str:
    return method.split(" (", 1)[0].strip()


def _read_sheet(path: Path) -> tuple[List[str], List[List[Any]]]:
    import openpyxl

    wb = openpyxl.load_workbook(str(path), data_only=True)
    ws = wb["Interpolation"]
    rows = list(ws.iter_rows(values_only=True))
    headers = [str(h) if h is not None else "" for h in rows[0]]
    return headers, rows[1:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default="batch_evaluation_part.xlsx")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--date-added", type=str, default="2026-08-21")
    parser.add_argument(
        "--exclude", type=str, default="",
        help="Extra model types to skip, space-separated (default excludes 'guo2023_mst').",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = _REPO_ROOT / input_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded = set(EXCLUDED_MODEL_TYPES)
    if args.exclude.strip():
        excluded |= {s for s in args.exclude.split() if s}

    headers, rows = _read_sheet(input_path)
    method_idx = headers.index("Method")
    params_idx = headers.index("Parameters (M)") if "Parameters (M)" in headers else None
    metric_cols = [
        (i, h.lower()) for i, h in enumerate(headers)
        if h not in ("Method", "Parameters (M)")
    ]

    results_by_model: Dict[str, List[Dict[str, Any]]] = {}
    params_by_model: Dict[str, Optional[float]] = {}
    for row in rows:
        if row[method_idx] is None:
            continue
        method = str(row[method_idx]).strip()
        mtype = _model_type(method)
        bid = _benchmark_id(method)
        if mtype in excluded:
            print(f"skipping excluded model type {mtype!r}")
            continue
        if mtype not in MODELS:
            raise SystemExit(f"ERROR: unknown model type {mtype!r}; add it to MODELS.")
        if bid is None:
            raise SystemExit(f"ERROR: could not derive benchmark_id from {method!r}.")

        if params_idx is not None:
            params_by_model[mtype] = _parse_mean(row[params_idx])

        scores: Dict[str, Optional[float]] = {}
        for col_idx, key in metric_cols:
            scores[key] = _parse_mean(row[col_idx])

        results_by_model.setdefault(mtype, []).append(
            {
                "model_id": f"{mtype}_interpolation",
                "benchmark_id": bid,
                "scores": scores,
            }
        )

    for mtype, meta in MODELS.items():
        if mtype not in results_by_model:
            continue
        model_id = f"{mtype}_interpolation"
        code_url = f"{_REPO_URL}/model/interpolation/{mtype}.py"

        model_doc = {
            "id": model_id,
            "name": meta["name"],
            "authors": meta["authors"],
            "year": meta["year"],
            "type": "deep_learning",
            "tasks": ["interpolation"],
            "paper_url": meta["paper_url"],
            "code_url": code_url,
            "weights_url": None,
            "parameters_m": params_by_model.get(mtype),
        }

        result_docs = []
        for entry in results_by_model[mtype]:
            result_docs.append(
                {
                    "model_id": entry["model_id"],
                    "benchmark_id": entry["benchmark_id"],
                    "scores": entry["scores"],
                    "paper_url": meta["paper_url"],
                    "code_url": code_url,
                    "date_added": args.date_added,
                }
            )

        model_path = output_dir / f"interpolation_model_{mtype}.json"
        result_path = output_dir / f"interpolation_result_{mtype}.json"
        model_path.write_text(json.dumps(model_doc, indent=2) + "\n")
        result_path.write_text(json.dumps(result_docs, indent=4) + "\n")
        print(f"{mtype:<22s} {len(result_docs)} scenarios -> "
              f"{model_path.name}, {result_path.name}")


if __name__ == "__main__":
    main()
