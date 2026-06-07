"""
Upload trained model checkpoints (best.pt + config.yaml) to a Hugging Face repository.

Usage:
    export HF_NAMESPACE=GeoBrain  # or HF_USERNAME (personal account)
    export HF_TOKEN="your_hf_token"
    python tools/upload_to_hf.py

Optional:
    --repo-name NAME     HF repo name (default: ground-roll-attenuation)
    --results-dir PATH   Override results root (default: /data/shared/benchmark/ground_roll/results)
    --dry-run            Scan and print what would be uploaded without uploading
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

RESULTS_ROOT = "/data/shared/benchmark/ground_roll/results"
DEFAULT_REPO = "ground-roll-attenuation"

FOLDER_PATTERN = re.compile(
    r"denoise_(?P<model>.+)_base\d+_level(?P<level>[\d.]+)_seed(?P<seed>\d+)"
)

MODEL_DISPLAY = {
    "unet": "UNet",
    "res_unet": "ResUNet",
    "dncnn": "DnCNN",
    "atten_unet": "Attention UNet",
}

MODEL_DESCRIPTION = {
    "unet": "Classic encoder-decoder with skip connections (Ronneberger et al., 2015). Base channels: 32, depth: 4.",
    "res_unet": "U-Net with residual blocks replacing plain double-conv layers (He et al., 2016; Zhang et al., 2018). Base channels: 32, depth: 4.",
    "dncnn": "Flat 17-layer Conv-BN-ReLU stack with residual learning (Zhang et al., 2017, IEEE TIP). Base channels: 64.",
    "atten_unet": "U-Net with additive attention gates on skip connections (Oktay et al., 2018, MIDL). Base channels: 32, depth: 4.",
}

NOISE_LEVELS = [1.0, 3.0, 5.0, 7.0, 9.0]


def generate_model_card(entries, repo_name: str) -> str:
    """Generate a Hugging Face model card README.md."""
    model_keys = sorted({e["model"] for e in entries})
    models_str = "\n".join(
        f"- **{MODEL_DISPLAY.get(k, k)}** (`{k}`) — {MODEL_DESCRIPTION.get(k, '')}"
        for k in model_keys
    )

    # Count experiments per model
    count_str = "\n".join(
        f"  - {MODEL_DISPLAY.get(k, k)}: {sum(1 for e in entries if e['model'] == k)} checkpoints"
        for k in model_keys
    )

    card = f"""---
tags:
- seismic
- ground-roll
- denoising
- unet
- resunet
- dncnn
- attention-unet
- pytorch
library_name: pytorch
---

# Ground-Roll Attenuation Benchmark

Deep-learning-based coherent noise (ground roll) suppression on pre-stack seismic shot gathers, using the SEG C3 synthetic dataset.

## Task

Given a noisy shot gather contaminated by dispersive ground-roll noise, the model predicts the additive noise component. The denoised signal is obtained by:

```
denoised = noisy_input - predicted_noise
```

This is a **paired regression** task trained with a noise-label objective (the ground-truth noise component). The supervision target is the residual between the noisy input and the clean reference.

## Dataset

- **Source**: SEG C3 pre-stack synthetic data, 9 regular shot gathers
- **Geometry**: 201 traces × 625 time samples per shot, dt = 2 ms
- **Noise modeling**: Reflection signals modeled with the acoustic wave equation; ground roll modeled with the elastic wave equation to capture its dispersive, low-velocity character
- **Split**: Shot-level (FFID) sequential 7:1:1 — 7 training shots, 1 validation, 1 held-out test

### Noise Intensity Levels

Five ground-roll intensity levels produce paired noisy / noise-label records:

| Level | SNR (dB) | PSNR (dB) | SSIM | MAE | MSE | RMSE |
|-------|----------|-----------|------|-----|-----|------|
| 1.0   | 2.71     | 15.81     | 0.9480 | 0.030624 | 0.026232 | 0.161964 |
| 3.0   | -6.83    | 6.27      | 0.9418 | 0.091871 | 0.236091 | 0.485892 |
| 5.0   | -11.27   | 1.83      | 0.9402 | 0.153118 | 0.655809 | 0.809820 |
| 7.0   | -14.19   | -1.09     | 0.9395 | 0.214366 | 1.285386 | 1.133749 |
| 9.0   | -16.37   | -3.27     | 0.9390 | 0.275613 | 2.124822 | 1.457677 |

*Metrics computed on the full dataset in the original amplitude domain before normalization.*

## Model Architectures

{models_str}

### Preprocessing

- **Normalization**: `max_abs`, global scope — the entire dataset scaled to [-1, 1]
- **Patching**: Overlapping 2D patches (128 × 256) with 50% overlap, channel-last format (1, H, W)

## Repository Structure

```
models/
├── unet/
│   ├── level1.0_seed42/
│   │   ├── best.pt          # Best checkpoint (minimum validation loss)
│   │   └── config.yaml      # Full training configuration
│   ├── level1.0_seed43/
│   ├── level1.0_seed44/
│   ├── level3.0_seed42/
│   └── ...
└── res_unet/
    └── ...
```

Each subdirectory corresponds to one experiment: a model architecture trained at a specific noise level with a specific random seed.

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Loss | MSE (predicted noise vs. label noise) |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | Cosine annealing (min_lr=1e-6) |
| Epochs | 200 |
| Gradient clipping | 1.0 (max norm) |
| Batch size | 196 |
| Distributed training | 2 × NVIDIA RTX 4090, DDP |
| Seeds | 42, 43, 44 per experiment |

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download a checkpoint
repo = "{repo_name}"
model_key = "res_unet"
level = "3.0"
seed = "42"

ckpt_path = hf_hub_download(
    repo_id=repo,
    filename=f"models/{{model_key}}/level{{level}}_seed{{seed}}/best.pt",
)

# Load state dict
state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

# For full model loading, instantiate the corresponding architecture
# and load the state dict (see config.yaml for exact architecture params).
```

See the companion benchmark documentation for detailed experimental setup and full evaluation results.

## Results (SNR dB)

| Model | Params (M) | Level 1.0 | Level 3.0 | Level 5.0 | Level 7.0 | Level 9.0 |
|-------|:----------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| Raw (noisy) | — | 2.71 | −6.83 | −11.27 | −14.19 | −16.37 |
| UNet | 7.76 | 28.39±1.09 | 23.07±0.10 | 20.22±0.32 | 17.90±0.28 | 16.60±0.46 |
| ResUNet | 8.11 | 31.62±0.56 | 22.65±0.93 | 18.11±2.12 | 16.60±1.44 | 14.61±0.01 |
| DnCNN | 0.56 | 31.67±0.11 | 30.02±0.43 | 22.91±3.74 | — | — |
| Attention UNet | 7.85 | 28.79±0.46 | 23.10±1.00 | 19.66±0.22 | 17.57±0.11 | 16.61±0.19 |

Mean ± std over 3 seeds. All models achieve 15–31 dB SNR improvement. DnCNN delivers the best performance at low-to-mid noise levels with the smallest footprint (0.56 M parameters). See the benchmark documentation for per-level detailed metrics (PSNR, SSIM, MAE, MSE, RMSE).

## References

- Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI 2015
- He et al., Deep Residual Learning for Image Recognition, CVPR 2016
- Zhang et al., Image Denoising via Deep CNN (DnCNN), IEEE TIP 2017
- Oktay et al., Attention U-Net: Learning Where to Look for the Pancreas, MIDL 2018
- SEG C3 Velocity Model: https://wiki.seg.org/wiki/C3
"""
    return card


def scan_results(results_dir: str):
    """Scan results directory and return list of (folder_name, model, level, seed)."""
    entries = []
    for name in os.listdir(results_dir):
        m = FOLDER_PATTERN.match(name)
        if not m:
            continue
        fpath = os.path.join(results_dir, name)
        if not os.path.isdir(fpath):
            continue
        best_pt = os.path.join(fpath, "checkpoints", "best.pt")
        config_yaml = os.path.join(fpath, "config.yaml")
        entries.append(
            {
                "folder": name,
                "path": fpath,
                "model": m.group("model"),
                "level": m.group("level"),
                "seed": m.group("seed"),
                "best_pt": best_pt if os.path.isfile(best_pt) else None,
                "config_yaml": config_yaml if os.path.isfile(config_yaml) else None,
            }
        )
    entries.sort(key=lambda x: (x["model"], x["level"], x["seed"]))
    return entries


def hf_path(entry: dict, filename: str) -> str:
    """Construct the path inside the HF repo, e.g. models/res_unet/level1.0_seed42/best.pt."""
    subdir = f"{entry['model']}/level{entry['level']}_seed{entry['seed']}"
    return f"models/{subdir}/{filename}"


def main():
    parser = argparse.ArgumentParser(description="Upload model checkpoints to Hugging Face.")
    parser.add_argument("--repo-name", default=DEFAULT_REPO, help=f"HF repo name (default: {DEFAULT_REPO})")
    parser.add_argument(
        "--results-dir", default=RESULTS_ROOT, help=f"Results root (default: {RESULTS_ROOT})"
    )
    parser.add_argument("--dry-run", action="store_true", help="Scan and print what would be uploaded")
    parser.add_argument(
        "--no-model-card", action="store_true", help="Skip uploading the model card"
    )
    args = parser.parse_args()

    # namespace = os.environ.get("HF_NAMESPACE") or os.environ.get("HF_USERNAME")
    namespace = os.environ.get("HF_NAMESPACE")

    token = os.environ.get("HF_TOKEN")

    entries = scan_results(args.results_dir)
    if not entries:
        logger.warning("No matching result folders found in %s", args.results_dir)
        sys.exit(0)

    repo_id = f"{namespace}/{args.repo_name}" if namespace else args.repo_name
    total = len(entries)
    found_pt = sum(1 for e in entries if e["best_pt"])
    found_yaml = sum(1 for e in entries if e["config_yaml"])

    logger.info("Found %d experiment folders:", total)
    logger.info("  - %d have checkpoints/best.pt", found_pt)
    logger.info("  - %d have config.yaml", found_yaml)

    if args.dry_run:
        logger.info("Dry-run mode — files that would be uploaded:")
        for e in entries:
            if e["best_pt"]:
                logger.info("  [upload] %s", hf_path(e, "best.pt"))
            if e["config_yaml"]:
                logger.info("  [upload] %s", hf_path(e, "config.yaml"))
        if not args.no_model_card:
            logger.info("  [upload] README.md (model card)")
        return

    if not token:
        logger.error("HF_TOKEN environment variable is not set.")
        sys.exit(1)

    api = HfApi()
    logger.info("Creating / ensuring repo: %s", repo_id)
    create_repo(repo_id, token=token, exist_ok=True, private=False)

    # Fetch existing files in the repo for incremental (skip-already-existing) upload
    try:
        existing = set(api.list_repo_files(repo_id, token=token))
        logger.info("Repo has %d existing file(s); will skip those.", len(existing))
    except Exception:
        logger.info("Could not list repo files (repo may be empty); uploading all.")
        existing = set()

    uploaded = 0
    skipped = 0
    failed = 0

    # Upload model card (always overwrite — it is regenerated each run)
    if not args.no_model_card:
        card = generate_model_card(entries, repo_id)
        try:
            api.upload_file(
                path_or_fileobj=card.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                token=token,
            )
            logger.info("  [OK]     README.md (model card)")
            uploaded += 1
        except Exception as exc:
            logger.error("  [FAIL]   README.md — %s", exc)
            failed += 1

    # Upload checkpoints and configs
    for e in entries:
        files_to_upload = {}
        if e["best_pt"]:
            files_to_upload["best.pt"] = e["best_pt"]
        if e["config_yaml"]:
            files_to_upload["config.yaml"] = e["config_yaml"]

        for fname, local_path in files_to_upload.items():
            remote = hf_path(e, fname)
            if remote in existing:
                logger.info("  [SKIP]   %s (already exists)", remote)
                skipped += 1
                continue
            try:
                upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=remote,
                    repo_id=repo_id,
                    token=token,
                )
                logger.info("  [OK]     %s", remote)
                uploaded += 1
            except Exception as exc:
                logger.error("  [FAIL]   %s — %s", remote, exc)
                failed += 1

    logger.info("Done. %d uploaded, %d skipped, %d error(s).", uploaded, skipped, failed)


if __name__ == "__main__":
    main()
