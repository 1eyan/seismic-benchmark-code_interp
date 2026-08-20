#!/usr/bin/env python
"""Normalize the metrics block of interpolation configs to snr/psnr/ssim/mae/mse.

Rewrites the ``metrics:`` ... ``optim:`` block of every YAML under a config dir
so every config reports the same five scalar metrics: ``snr``, ``psnr``, ``ssim``,
``mae``, ``mse``.  ``psnr`` (data_range 1.0) and ``mae`` are inserted where
missing, and duplicate metric entries are removed.  Every config is normalized to
SSIM ``data_range: 2.0`` (peak-to-peak for max_abs-normalized [-1, 1] data).

Usage::

    python scripts/interpolation/normalize_metrics.py --config-dir collected/configs
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_BLOCK_RE = re.compile(r"^metrics:\n.*?(?=^optim:\n)", re.MULTILINE | re.DOTALL)

TEMPLATE = """metrics:
- name: snr
  params:
    reduction: per_sample
- name: psnr
  params:
    data_range: 1.0
    reduction: per_sample
- name: ssim
  params:
    data_range: 2.0
    window_size: 11
    sigma: 1.5
- name: mae
  params: {}
- name: mse
  params: {}
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
        if _BLOCK_RE.search(text) is None:
            print(f"SKIP (no metrics block): {path.name}", file=sys.stderr)
            skipped += 1
            continue
        new_text = _BLOCK_RE.sub(TEMPLATE, text)
        if new_text == text:
            continue
        path.write_text(new_text)
        changed += 1
        print(f"  updated {path.name}")

    print(f"Changed : {changed}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
