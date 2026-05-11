"""
Upload the SEG C3 ground-roll dataset (SEG-Y files) to a Hugging Face dataset repository.

Usage:
    export HF_NAMESPACE="your-org-name"   # or HF_USERNAME (personal account)
    export HF_TOKEN="your_hf_token"
    python tools/upload_dataset_to_hf.py

Optional:
    --repo-name NAME      HF repo name (default: seg-c3-ground-roll)
    --data-root PATH      Override data root (default: /data/shared/benchmark/ground_roll)
    --asset-dir PATH      Asset images directory (default: ./asset)
    --levels LIST         Comma-separated noise levels to upload (default: all)
    --dry-run             Scan and print what would be uploaded without uploading
"""

import argparse
import logging
import os
import re
import sys

from huggingface_hub import HfApi, create_repo, upload_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_ROOT = "/data/shared/benchmark/ground_roll"
DEFAULT_REPO = "seg-c3-ground-roll"

FILE_PATTERN = re.compile(r"SEGC3_shots1_9_(noisy|noise)_([\d.]+)\.sgy")


def generate_dataset_card(entries, repo_id: str, asset_entries=None) -> str:
    """Generate a Hugging Face dataset card README.md."""
    levels = sorted({e["level"] for e in entries}, key=float)
    levels_str = ", ".join(levels)
    n_noisy = sum(1 for e in entries if e["kind"] == "noisy")
    n_noise = sum(1 for e in entries if e["kind"] == "noise")

    level_rows = "\n".join(
        f"| {lv} | SEGC3_shots1_9_noisy_{lv}.sgy | SEGC3_shots1_9_noise_{lv}.sgy | ~951 MB × 2 |"
        for lv in levels
    )

    # Build image gallery sections
    image_section = ""
    if asset_entries:
        groups = {"noisy": "Noisy Shot Gather", "clean": "Clean Reference Signal", "noise": "Ground-Roll Noise (Label)"}
        for kind, heading in groups.items():
            imgs = sorted(
                [a for a in asset_entries if a["kind"] == kind],
                key=lambda x: x["filename"],
            )
            if not imgs:
                continue
            img_tags = "\n".join(
                f'  <img src="assets/{a["filename"]}" alt="{heading} example {i+1}" width="48%">'
                for i, a in enumerate(imgs)
            )
            captions = " &nbsp;|&nbsp; ".join(f"Example {i+1}" for i in range(len(imgs)))
            image_section += f"""
### {heading}

<div align="center">

{img_tags}

</div>

<div align="center"><i>{captions} — {heading.lower()} for a representative shot gather at noise level 3.0.</i></div>

"""

    card = f"""---
tags:
- seismic
- ground-roll
- denoising
- seg-c3
- geophysics
- synthetic
task_categories:
- image-to-image
- other
size_categories:
- 10M-100M
pretty_name: SEG C3 Ground-Roll Dataset
---

# SEG C3 Ground-Roll Dataset

Paired noisy-input / noise-label SEG-Y volumes for supervised ground-roll attenuation, derived from the SEG C3 synthetic velocity model.

## Task

**Noise-label regression**: given a noisy pre-stack shot gather, predict the additive ground-roll noise component. The clean signal is recovered as:

```
denoised = noisy_input - predicted_noise
```

The noise labels serve as regression targets. Both input and label are 3D SEG-Y volumes with identical geometry.

## Dataset Description

- **Source**: SEG C3 synthetic velocity model ([wiki.seg.org/wiki/C3](https://wiki.seg.org/wiki/C3))
- **Geometry**: 9 regular shot gathers, 201 traces × 625 time samples per shot, dt = 2 ms
- **Noise modeling**: Reflection signals modeled with the acoustic wave equation; ground roll modeled with the elastic wave equation to capture its dispersive, low-velocity character
- **Noise intensity levels**: {levels_str}
- **Format**: Pre-stack SEG-Y (revision 1), IBM float
{image_section}
## File Structure

Each noise level has a matched pair of SEG-Y files:

| Level | Noisy Input | Noise Label | Size (approx) |
|-------|------------|-------------|---------------|
{level_rows}

**Total**: {n_noisy} noisy + {n_noise} noise SEG-Y files

## Loading Data

```python
import segyio
import numpy as np

def read_shot_gather(path, traces_per_shot=201):
    '''Read a regular SEG-Y file into (n_shots, n_traces, n_time).'''
    with segyio.open(path, "r", strict=False) as src:
        n_traces_total = src.tracecount
        n_shots = n_traces_total // traces_per_shot
        n_time = src.samples.size
        data = np.zeros((n_shots, traces_per_shot, n_time), dtype=np.float32)
        for i in range(n_shots):
            for j in range(traces_per_shot):
                data[i, j, :] = src.trace[i * traces_per_shot + j]
    return data

# Load a level-3.0 pair
noisy  = read_shot_gather("noisy/SEGC3_shots1_9_noisy_3.0.sgy")
noise  = read_shot_gather("noise/SEGC3_shots1_9_noise_3.0.sgy")
signal = noisy - noise   # clean reference
```

With `huggingface_hub`:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="{repo_id}",
    filename="noisy/SEGC3_shots1_9_noisy_3.0.sgy",
    repo_type="dataset",
)
```

## Train / Val / Test Split

Shot-level sequential split by FFID (field file ID), avoiding trace leakage:

| Split | Shots | Fraction |
|-------|-------|----------|
| Train | 7     | 77.8%    |
| Val   | 1     | 11.1%    |
| Test  | 1     | 11.1%    |

The split is done at loading time (not pre-saved as separate files) so users can adjust the ratios.

## Preprocessing Recipe

The companion benchmark applies:

1. **Normalization**: `max_abs`, global scope — the entire noisy volume scaled to [-1, 1]; same stats applied to the noise label
2. **Patching**: Overlapping 2D patches (128 traces × 256 time samples), 50% overlap, yielding (1, H, W) tensors

No spherical-divergence correction is applied (raw amplitudes are used).

## Benchmark Results

See the companion model repository for full benchmark results across UNet, ResUNet, DnCNN, and Attention UNet architectures at each noise level.

## Citation

If you use this dataset, please cite the SEG C3 model and the companion benchmark:

```bibtex
@misc{{seg_c3_ground_roll,
  title={{SEG C3 Ground-Roll Attenuation Benchmark}},
  howpublished={{https://huggingface.co/datasets/{repo_id}}},
}}
```

## References

- SEG C3 Velocity Model: https://wiki.seg.org/wiki/C3
- `segyio` library: https://github.com/equinor/segyio
"""
    return card


def scan_data(data_root: str, levels=None):
    """Scan data directory for SEG-Y file pairs and return upload entries."""
    entries = []
    noisy_dir = os.path.join(data_root, "noisy")
    noise_dir = os.path.join(data_root, "noise")

    if not os.path.isdir(noisy_dir):
        logger.warning("Noisy directory not found: %s", noisy_dir)
    if not os.path.isdir(noise_dir):
        logger.warning("Noise directory not found: %s", noise_dir)

    for dirpath, kind in [(noisy_dir, "noisy"), (noise_dir, "noise")]:
        if not os.path.isdir(dirpath):
            continue
        for fname in sorted(os.listdir(dirpath)):
            m = FILE_PATTERN.match(fname)
            if not m:
                continue
            file_kind, level = m.group(1), m.group(2)
            if file_kind != kind:
                continue
            if levels is not None and level not in levels:
                continue
            fpath = os.path.join(dirpath, fname)
            fsize = os.path.getsize(fpath)
            entries.append(
                {
                    "kind": kind,
                    "level": level,
                    "filename": fname,
                    "local_path": fpath,
                    "repo_path": f"{kind}/{fname}",
                    "size_mb": fsize / (1024 * 1024),
                }
            )
    entries.sort(key=lambda x: (float(x["level"]), x["kind"]))
    return entries


def scan_assets(asset_dir: str):
    """Scan asset directory for sample images and return upload entries."""
    if not os.path.isdir(asset_dir):
        return []
    entries = []
    for fname in sorted(os.listdir(asset_dir)):
        if not fname.lower().endswith(".png"):
            continue
        fpath = os.path.join(asset_dir, fname)
        fsize = os.path.getsize(fpath)
        # Determine kind from filename prefix: noisy1.png → noisy, clean2.png → clean, noise3.png → noise
        stem = os.path.splitext(fname)[0]
        kind = stem.rstrip("0123456789")
        entries.append(
            {
                "kind": kind,
                "filename": fname,
                "local_path": fpath,
                "repo_path": f"assets/{fname}",
                "size_mb": fsize / (1024 * 1024),
            }
        )
    return entries


def main():
    parser = argparse.ArgumentParser(description="Upload SEG C3 dataset to Hugging Face.")
    parser.add_argument(
        "--repo-name", default=DEFAULT_REPO, help=f"HF repo name (default: {DEFAULT_REPO})"
    )
    parser.add_argument(
        "--data-root", default=DATA_ROOT, help=f"Data root (default: {DATA_ROOT})"
    )
    parser.add_argument(
        "--asset-dir", default="./asset", help="Asset images directory (default: ./asset)"
    )
    parser.add_argument(
        "--levels",
        default=None,
        help="Comma-separated noise levels to upload (default: all found)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Scan and print what would be uploaded"
    )
    parser.add_argument(
        "--no-dataset-card", action="store_true", help="Skip uploading the dataset card"
    )
    parser.add_argument(
        "--no-assets", action="store_true", help="Skip uploading asset images"
    )
    args = parser.parse_args()

    namespace = os.environ.get("HF_NAMESPACE") or os.environ.get("HF_USERNAME")
    token = os.environ.get("HF_TOKEN")

    level_set = None
    if args.levels:
        level_set = {s.strip() for s in args.levels.split(",")}

    entries = scan_data(args.data_root, level_set)
    if not entries:
        logger.warning("No matching SEG-Y files found in %s", args.data_root)
        sys.exit(0)

    asset_entries = [] if args.no_assets else scan_assets(args.asset_dir)

    repo_id = f"{namespace}/{args.repo_name}" if namespace else args.repo_name
    total_size = sum(e["size_mb"] for e in entries)
    asset_size = sum(a["size_mb"] for a in asset_entries)
    n_noisy = sum(1 for e in entries if e["kind"] == "noisy")
    n_noise = sum(1 for e in entries if e["kind"] == "noise")
    n_levels = len({e["level"] for e in entries})

    logger.info(
        "Found %d SEG-Y files (%d noisy + %d noise) across %d level(s), %.1f MB total",
        len(entries), n_noisy, n_noise, n_levels, total_size,
    )
    if asset_entries:
        logger.info("Found %d asset image(s), %.1f MB total", len(asset_entries), asset_size)

    if args.dry_run:
        logger.info("Dry-run mode — files that would be uploaded:")
        for e in entries:
            logger.info(
                "  [upload] %-50s  %7.1f MB  →  %s",
                e["filename"], e["size_mb"], e["repo_path"],
            )
        for a in asset_entries:
            logger.info(
                "  [upload] %-50s  %7.1f MB  →  %s",
                a["filename"], a["size_mb"], a["repo_path"],
            )
        if not args.no_dataset_card:
            logger.info("  [upload] README.md (dataset card)")
        return

    if not token:
        logger.error("HF_TOKEN environment variable is not set.")
        sys.exit(1)

    api = HfApi()
    logger.info("Creating / ensuring dataset repo: %s", repo_id)
    create_repo(repo_id, token=token, exist_ok=True, private=False, repo_type="dataset")

    # Fetch existing files in the repo for incremental (skip-already-existing) upload
    try:
        existing = set(api.list_repo_files(repo_id, repo_type="dataset", token=token))
        logger.info("Repo has %d existing file(s); will skip those.", len(existing))
    except Exception:
        logger.info("Could not list repo files (repo may be empty); uploading all.")
        existing = set()

    uploaded = 0
    skipped = 0
    failed = 0

    # Upload dataset card (always overwrite — it is regenerated each run)
    if not args.no_dataset_card:
        card = generate_dataset_card(entries, repo_id, asset_entries)
        try:
            api.upload_file(
                path_or_fileobj=card.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            logger.info("  [OK]     README.md (dataset card)")
            uploaded += 1
        except Exception as exc:
            logger.error("  [FAIL]   README.md — %s", exc)
            failed += 1

    # Upload SEG-Y files
    for e in entries:
        if e["repo_path"] in existing:
            logger.info("  [SKIP]   %-50s  %7.1f MB  (already exists)", e["repo_path"], e["size_mb"])
            skipped += 1
            continue
        try:
            upload_file(
                path_or_fileobj=e["local_path"],
                path_in_repo=e["repo_path"],
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            logger.info("  [OK]     %-50s  %7.1f MB", e["repo_path"], e["size_mb"])
            uploaded += 1
        except Exception as exc:
            logger.error("  [FAIL]   %s — %s", e["repo_path"], exc)
            failed += 1

    # Upload asset images
    for a in asset_entries:
        if a["repo_path"] in existing:
            logger.info("  [SKIP]   %-50s  %7.1f MB  (already exists)", a["repo_path"], a["size_mb"])
            skipped += 1
            continue
        try:
            upload_file(
                path_or_fileobj=a["local_path"],
                path_in_repo=a["repo_path"],
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            logger.info("  [OK]     %-50s  %7.1f MB", a["repo_path"], a["size_mb"])
            uploaded += 1
        except Exception as exc:
            logger.error("  [FAIL]   %s — %s", a["repo_path"], exc)
            failed += 1

    logger.info("Done. %d uploaded, %d skipped, %d error(s).", uploaded, skipped, failed)


if __name__ == "__main__":
    main()
