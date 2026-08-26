#!/usr/bin/env python
"""Collect interpolation inference npy files into one hierarchical folder.

Groups ``collected/configs/*.yaml`` by (model_type, mask scenario) — the same
grouping the "Interpolation" sheet of ``batch_evaluation_part.xlsx`` uses — and
copies/moves/symlinks the selected ``inference/npy`` files into

    <output-root>/<model_type>/<scenario>/seed<NN>/<file>.npy

where ``<scenario>`` is ``<mode>_<miss>`` (e.g. ``continuous_20tr``,
``random_10-30``, ``uniform_50``).

Modes are copy / move / symlink.  ``move`` relocates the source file (deletes
the original); use it only if you no longer need the source copy under
``collected/``.  ``--dry-run`` prints the plan without touching anything.

Usage::

    python scripts/interpolation/collect_inference_npy.py --dry-run
    python scripts/interpolation/collect_inference_npy.py \
        --mode move --files pred --seeds 42 \
        --output-root /NAS/czt/mount/interpolation_npy
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]

_NAME_RE = re.compile(r"^interp_(.+)_seed([0-9]+)_(.+)$")

FILES = {
    "pred": ["pred_shots.npy"],
    "input_target": ["input_shots.npy", "target_shots.npy"],
    "all": ["input_shots.npy", "pred_shots.npy", "target_shots.npy"],
}


def _resolve(path: str, repo_root: Path = _REPO_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (repo_root / p).resolve()


def _scenario(config_name: str) -> Optional[str]:
    m = _NAME_RE.match(config_name)
    if m is None:
        return None
    rest = m.group(3)
    mode, _, miss = rest.partition("_miss")
    if not miss:
        return None
    return f"{mode}_{miss}"


def _seed(config_name: str) -> str:
    m = _NAME_RE.match(config_name)
    return m.group(2) if m else ""


def _model_type(config_path: Path) -> str:
    try:
        return str(yaml.safe_load(config_path.read_text())["model"]["type"])
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return "?"


def _build_plan(
    config_dir: Path,
    collect_root: Path,
    seeds: List[str],
    files: List[str],
    include_images: bool = False,
) -> List[Tuple[Path, Path]]:
    plan: List[Tuple[Path, Path]] = []
    missing: List[str] = []
    for cfg in sorted(config_dir.glob("*.yaml")):
        name = cfg.stem
        seed = _seed(name)
        if seed not in seeds:
            continue
        scenario = _scenario(name)
        mtype = _model_type(cfg)
        if scenario is None:
            continue
        for fname in files:
            src = collect_root / name / "inference" / "npy" / fname
            if not src.is_file():
                missing.append(str(src))
                continue
            plan.append((src, (mtype, scenario, seed, Path(fname))))
        if include_images:
            viz_dir = collect_root / name / "inference" / "visualizations"
            for img in sorted(viz_dir.glob("*.png")):
                plan.append((img, (mtype, scenario, seed, Path("visualizations") / img.name)))
    return plan, missing


def _apply(plan, output_root: Path, mode: str, dry_run: bool) -> None:
    n = len(plan)
    for i, (src, (mtype, scenario, seed, fname)) in enumerate(plan, start=1):
        dst = output_root / mtype / scenario / f"seed{seed}" / fname
        if dry_run:
            print(f"[{i:>2}/{n}] {src} -> {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print(f"  skip (exists): {dst}")
            continue
        if mode == "move":
            shutil.move(str(src), str(dst))
        elif mode == "copy":
            shutil.copy2(str(src), str(dst))
        elif mode == "symlink":
            rel = os.path.relpath(src, dst.parent)
            os.symlink(rel, str(dst))
        print(f"[{i:>2}/{n}] {mode:>7} {dst}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=str, default="collected/configs")
    parser.add_argument("--collect-root", type=str, default="collected")
    parser.add_argument("--output-root", type=str, default="/NAS/czt/mount/interpolation_npy")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Space-separated seeds to collect (default '42').")
    parser.add_argument("--files", type=str, default="pred",
                        choices=["pred", "input_target", "all"],
                        help="Which npy files per run: pred / input_target / all (default 'pred').")
    parser.add_argument("--images", action="store_true",
                        help="Also collect inference visualizations (*.png) into seed<NN>/visualizations/.")
    parser.add_argument("--mode", type=str, default="copy",
                        choices=["copy", "move", "symlink"],
                        help="copy / move / symlink (default 'copy').")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan without touching files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = _resolve(args.config_dir)
    collect_root = _resolve(args.collect_root)
    output_root = Path(args.output_root)
    seeds = args.seeds.split()
    files = FILES[args.files]

    if not config_dir.is_dir():
        print(f"ERROR: config dir not found: {config_dir}", file=sys.stderr)
        sys.exit(1)

    plan, missing = _build_plan(config_dir, collect_root, seeds, files, include_images=args.images)
    if missing:
        print(f"WARNING: {len(missing)} source npy missing:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)

    total = sum(src.stat().st_size for src, _ in plan)
    print(f"Files to collect : {len(plan)}")
    print(f"Total size       : {total / 1e9:.2f} GB")
    print(f"Output root      : {output_root}")
    print(f"Mode             : {'dry-run' if args.dry_run else args.mode}")

    if args.dry_run:
        _apply(plan, output_root, args.mode, dry_run=True)
        return

    confirm = input("Proceed? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return
    _apply(plan, output_root, args.mode, dry_run=False)


if __name__ == "__main__":
    main()
