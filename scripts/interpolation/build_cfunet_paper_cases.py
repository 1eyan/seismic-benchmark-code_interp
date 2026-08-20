#!/usr/bin/env python
"""Generate cfunet inference configs + checkpoint symlinks for the one-checkpoint benchmark.

Park2022 CFunet is evaluated with a single cfunet_random-trained checkpoint (the
``paper`` run) across the standard mask-scenario matrix instead of one checkpoint
per scenario.  This script derives, for each seed, the full scenario matrix
(uniform 30/50/70, random 30/50, continuous 20/30/40 traces) from the paper
config and writes:

  - ``collected/configs/interp_park2022_cfunet_paper_seed<seed>_<scenario>.yaml``
  - ``collected/params/interp_park2022_cfunet_paper_seed<seed>_<scenario>.pt``
    (symlink -> ``..._cfunet_random_miss50-88.pt``)

Each generated config carries the normalized metrics block (snr/psnr/ssim
data_range 2.0/mae/mse) and an ``inference.data`` block pointing at the held-out
shots10-18 volume, matching the other collected interpolation configs.

Usage::

    python scripts/interpolation/build_cfunet_paper_cases.py \
        --template collected/configs/interp_park2022_cfunet_paper_seed42_cfunet_random_miss50-88.yaml \
        --params-dir collected/params \
        --out-dir collected/configs
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

SEEDS = [42, 43, 44]

# (scenario suffix, mask_mode, mask_ratio, continuous_missing_traces or None)
# Note: uniform_miss30 is deliberately omitted — it derives stride 2, identical
# to uniform_miss50, so it would only duplicate that row.
_SCENARIOS = [
    ("uniform_miss50", "uniform", "0.5", None),
    ("uniform_miss70", "uniform", "0.7", None),
    ("random_miss30", "random", "0.3", None),
    ("random_miss50", "random", "0.5", None),
    ("continuous_miss20tr", "continuous", "0.1", 20),
    ("continuous_miss30tr", "continuous", "0.1", 30),
    ("continuous_miss40tr", "continuous", "0.1", 40),
]

_MASK_BLOCK_RE = re.compile(
    r"  mask_mode: cfunet_random\n  mask_ratio_range:\n  - 0\.5\n  - 0\.875\n"
)
_METRICS_BLOCK_RE = re.compile(r"^metrics:\n.*?(?=^optim:\n)", re.MULTILINE | re.DOTALL)

METRICS_TEMPLATE = """metrics:
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

_INFERENCE_BLOCK = """inference:
  data:
    segy:
      path: /NAS/czt/mount/chengzhitong/data/SEGC3/SEG_45Shot_shots10-18.sgy
      traces_per_shot: 201
      time_downsample: 1
  n_viz_shots: 5
"""


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (_REPO_ROOT / p).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=str,
        default="collected/configs/interp_park2022_cfunet_paper_seed42_cfunet_random_miss50-88.yaml",
    )
    parser.add_argument("--params-dir", type=str, default="collected/params")
    parser.add_argument("--out-dir", type=str, default="collected/configs")
    return parser.parse_args()


def _mask_block(mask_mode: str, mask_ratio: str) -> str:
    return f"  mask_mode: {mask_mode}\n  mask_ratio: {mask_ratio}\n"


def main() -> None:
    args = parse_args()
    template_path = _resolve(args.template)
    params_dir = _resolve(args.params_dir)
    out_dir = _resolve(args.out_dir)

    if not template_path.is_file():
        print(f"ERROR: template not found: {template_path}", file=sys.stderr)
        sys.exit(1)
    if not params_dir.is_dir():
        print(f"ERROR: params dir not found: {params_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    template = template_path.read_text()
    if _MASK_BLOCK_RE.search(template) is None:
        print("ERROR: template lacks the cfunet_random mask block.", file=sys.stderr)
        sys.exit(1)

    created_cfg = 0
    created_link = 0
    for seed in SEEDS:
        src_ckpt = params_dir / (
            f"interp_park2022_cfunet_paper_seed{seed}_cfunet_random_miss50-88.pt"
        )
        if not src_ckpt.is_file():
            print(f"ERROR: paper checkpoint missing: {src_ckpt}", file=sys.stderr)
            sys.exit(1)

        for scenario, mask_mode, mask_ratio, missing in _SCENARIOS:
            name = f"interp_park2022_cfunet_paper_seed{seed}_{scenario}"

            text = template
            text = text.replace(
                "name: interp_park2022_cfunet_paper_seed42_cfunet_random_miss50-88",
                f"name: {name}",
                1,
            )
            text = text.replace("  seed: 42\n", f"  seed: {seed}\n", 1)
            text = _MASK_BLOCK_RE.sub(_mask_block(mask_mode, mask_ratio), text, count=1)
            text = _METRICS_BLOCK_RE.sub(METRICS_TEMPLATE, text, count=1)
            if missing is not None:
                text = text.replace(
                    "  max_shots: null\n",
                    f"  max_shots: null\n  continuous_missing_traces: {missing}\n",
                    1,
                )
            text = text.replace(
                "paper_alignment:\n", _INFERENCE_BLOCK + "\npaper_alignment:\n", 1
            )

            cfg_path = out_dir / f"{name}.yaml"
            cfg_path.write_text(text)
            created_cfg += 1

            link_path = params_dir / f"{name}.pt"
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            link_path.symlink_to(src_ckpt.name)
            created_link += 1
            print(f"  {cfg_path.name} -> {src_ckpt.name}")

    print(f"Created configs : {created_cfg}")
    print(f"Created symlinks: {created_link}")


if __name__ == "__main__":
    main()
