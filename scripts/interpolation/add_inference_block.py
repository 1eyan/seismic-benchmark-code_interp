#!/usr/bin/env python
"""Add a held-out ``inference.data`` block to collected configs that lack one.

Some collected configs (``liu2022_wrdl``) were copied from training without an
``inference:`` block, so batch inference fell back to the training volume
(shots1-9).  This script inserts an ``inference.data.segy`` block pointing at the
held-out shots10-18 volume, matching the other collected interpolation configs,
immediately before the top-level ``paper_alignment:`` key.  Configs that already
carry an ``inference:`` block are left untouched.

Usage::

    python scripts/interpolation/add_inference_block.py --config-dir collected/configs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_INFERENCE_BLOCK = """inference:
  data:
    segy:
      path: /NAS/czt/mount/chengzhitong/data/SEGC3/SEG_45Shot_shots10-18.sgy
      traces_per_shot: 201
      time_downsample: 1
  n_viz_shots: 5
  replace_observed: true
"""


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (_REPO_ROOT / p).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=str, default="collected/configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = _resolve(args.config_dir)
    if not config_dir.is_dir():
        print(f"ERROR: config dir not found: {config_dir}", file=sys.stderr)
        sys.exit(1)

    changed = 0
    skipped = 0
    for path in sorted(config_dir.glob("*.yaml")):
        text = path.read_text()
        if "inference:\n" in text:
            skipped += 1
            continue
        if "paper_alignment:\n" not in text:
            print(f"SKIP (no paper_alignment anchor): {path.name}", file=sys.stderr)
            skipped += 1
            continue
        new_text = text.replace(
            "paper_alignment:\n", _INFERENCE_BLOCK + "\npaper_alignment:\n", 1
        )
        path.write_text(new_text)
        changed += 1
        print(f"  added inference block: {path.name}")

    print(f"Changed : {changed}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
