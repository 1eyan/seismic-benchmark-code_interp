# Tutorial Source Notes

> Verified facts extracted from the repository for the tutorial document.
> Date: 2026-07-10

## Project Identity

- Repository: `seismic-benchmark-code`
- Purpose: PyTorch benchmark template for seismic data processing (interpolation, denoising, supervised restoration) on SEG-Y / NPY / MAT volumes.
- Core pattern: registry + factory for models, losses, metrics, datasets.
- Training scripts are component-agnostic; they parse CLI args, load YAML configs, and wire factories. They never hard-code concrete components.

## Directory Responsibilities

- `model/` — `nn.Module` definitions and `MODEL_REGISTRY`. Task subpackages register their own models.
- `utils/` — Training infrastructure: datasets, losses, metrics, visualization, logging, optimizer/scheduler builders, train/eval loops, DDP helpers, checkpoint I/O.
- `tools/` — Data utilities: I/O (`array_io.py`, `segy_read.py`), preprocessing (`preprocessing.py`), patching (`patching.py`).
- `configs/` — One YAML file per experiment; hyper-parameters never hard-coded in source.
- `scripts/` — CLI entry points for training (`train_*.py`) and inference (`inference_*.py`), plus bash launchers (`*.sh`).
- `results/` — Experiment outputs (checkpoints, logs, CSVs, PNGs). Gitignored.
- `memory/` — Project memory: design decisions, update log, techniques, research references.

## Registry + Factory

Files:
- `model/registry.py` — `MODEL_REGISTRY`, `@register_model("name")`, `build_model(cfg)`
- `utils/datasets.py` — `DATASET_REGISTRY`, `@register_dataset("name")`, `BaseArrayDataset`
- `utils/losses.py` — `LOSS_REGISTRY`, `@register_loss("name")`, `BaseLoss`
- `utils/metrics.py` — `METRIC_REGISTRY`, `@register_metric("name")`, `BaseMetric`

YAML shape for every pluggable component:
```yaml
component:
  type: registered_name
  params:
    key: value
```

Model registration requires adding `from . import <file>  # noqa: F401` to `model/<task>/__init__.py` so the decorator runs at import time.

## Dependencies

No centralized requirements.txt / pyproject.toml yet. Required packages:
- torch
- numpy
- matplotlib
- pyyaml
- segyio
- scipy

## Random Noise Suppression Task Facts

### Data

- Source: SEG C3 pre-stack synthetic data.
- File: `data/SEG_45Shot_shots1-9.sgy` (or configured path).
- Geometry: 201 traces per shot, dt = 0.008 s (8 ms), but the YAML default is `dt: 0.008`.
- Volume shape: `(n_shots, n_traces, n_time)` = `(9, 201, n_time)`.
- Default split: `shot_split: {train: 7, val: 1, test: 1}`.

### Supported Models

- `unet`
- `res_unet`
- `dncnn`
- `atten_unet`
- `SCRN`

### Preprocessing Pipeline (random_noise_suppression)

Functions from `tools/preprocessing.py`:
- `add_noise(shots, kind="gaussian"|"poisson", snr_db=20.0, rng=None)`
- `normalize(shots, mode="minmax"|"max_abs"|"mean_std", per="shot"|"trace"|"global", override_stats=None)`
- `spherical_divergence_correction(shots, dt, t0=0.0, power=2.0)`
- `denormalize(...)` and `inverse_spherical_divergence_correction(...)` for inference.

Default config in `denoise_unet.yaml`:
- `normalize_mode: max_abs`
- `normalize_scope: shot`
- `noise_kind: gaussian`
- `snr_db: 5.0`
- `skip: ["spherical_divergence_correction"]`

Patching from `tools/patching.py`:
- `patchify_uniform(data, patch_size=(trace, time), overlap=0.0, output_ndim=3|4)`
- `unpatchify_uniform(patches, info)` — averages overlaps via `sum / count`.
- Default config: `patch_trace: 128`, `patch_time: 256`, `patch_overlap: 0.5`.

### Training CLI

Script: `scripts/random_noise_suppression/train_denoise_unet.py`

Args:
- `--config` (default: `configs/random_noise_suppression/denoise_unet.yaml`)
- `--resume` — path to checkpoint for resuming

Single-GPU command:
```bash
python scripts/random_noise_suppression/train_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml
```

Multi-GPU command:
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  scripts/random_noise_suppression/train_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml
```

Training outputs under `results/<experiment.name>/`:
- `checkpoints/epoch_*.pt`
- `checkpoints/best.pt` (lowest validation loss)
- `logs/train_log.txt`
- `logs/loss_history.csv`
- `logs/metrics_history.csv`
- `logs/loss_curve.png`
- `logs/metrics_curve.png`
- `visualizations/epoch_*.png`
- `config_used.yaml`

### Inference CLI

Script: `scripts/random_noise_suppression/inference_denoise_unet.py`

Args:
- `--config`
- `--checkpoint` (required if not in config)
- `--output-dir`
- `--n-viz-shots`
- `--seed`
- `--device`
- `--batch-size`
- `--save-npy`
- `--noise-kind` (`gaussian` | `poisson`)
- `--snr-db`

Single-GPU command with overrides:
```bash
python scripts/random_noise_suppression/inference_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml \
  --checkpoint results/random_noise/random_noise_unet_base/checkpoints/best.pt \
  --output-dir results/random_noise/random_noise_unet_base/inference \
  --noise-kind gaussian \
  --snr-db 5 \
  --n-viz-shots 5
```

Inference outputs under `--output-dir`:
- `inference.log`
- `metrics_summary.json` — mean scalar metrics + EB-WSE + FB-FRE
- `metrics_per_shot.csv` — per-shot scalar metrics
- `visualizations/` — sample PNGs
- `npy/` — optional arrays when `--save-npy` is set

### Metrics

Implemented in `utils/metrics.py`: `mse`, `rmse`, `mae`, `snr`, `psnr`, `ssim`.
`rmse` / `snr` / `psnr` accept `reduction="per_sample"` (default; mean of per-sample scores) or `"global"`.

Inference metric groups in `metrics_summary.json`:
- `noisy` — noisy input vs clean target
- `denoised` — model prediction vs clean target
- `delta` — `denoised - noisy`

### Binned Metrics

Configured under `inference.binned_metrics`:
- `enabled: true`
- `eb_wse.enabled: true` — energy-binned weak-signal evaluation
- `eb_wse.bins: [[5, 20], [20, 40], [40, 70], [70, 100]]`
- `eb_wse.smooth_sigma: 1.0`
- `fb_fre.enabled: true` — frequency-binned fidelity evaluation
- `fb_fre.rel_threshold: 0.001`
- `fb_fre.band_ratios: [0.20, 0.30, 0.30, 0.20]`
- `fb_fre.band_names: ["low", "mid", "high", "very_high"]`
- `fb_fre.taper_width: 0.0`

### Batch Sweep Shell Scripts

- `scripts/random_noise_suppression/train_denoise_unet.sh` — sweeps noise kind, SNR, seed by rewriting temporary YAML.
- `scripts/random_noise_suppression/inference_denoise_unet.sh` — same sweep + aggregates mean/std across seeds.
- `scripts/random_noise_suppression/run_all_random_noise_models.sh` — trains and infers unet, dncnn, res_unet, atten_unet sequentially.

Default sweep parameters:
- Noise kinds: `gaussian`, `poisson`
- SNRs: `-5`, `0`, `5` dB
- Seeds: `42`, `43`, `44`

## Other Tasks

### Ground-Roll Attenuation

- Paired noisy / noise-label volumes.
- Entry scripts: `scripts/ground_roll_attenuation/train_denoise_*.py`, `inference_denoise_*.py`.
- Configs: `configs/ground_roll_attenuation/denoise_*.yaml`.
- No synthetic noise injection; dataset provides pairs.

### Multiples Attenuation

- Paired noisy / noise-label volumes.
- Entry scripts: `scripts/multiples_attenuation/train_denoise_*.py`, `inference_denoise_*.py`.
- Configs: `configs/multiples_attenuation/denoise_*.yaml`.
- Similar to ground-roll attenuation; task semantics differ.

### Interpolation

- Single volume + trace masking.
- Entry scripts: `scripts/interpolation/train_interpolation_unet.py`, `train_paired_unet.py`, `inference_interpolation.py`.
- Configs: `configs/interpolation/interpolation_unet.yaml`, `paired_unet.yaml`.
- Uses `mask_traces` for spatial missing-trace simulation.

## How to Add a New Model (Minimal Example)

```python
# model/random_noise_suppression/my_net.py
import torch.nn as nn
from model.registry import register_model

@register_model("my_net")
class MyNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv(x)
```

```python
# model/random_noise_suppression/__init__.py
from . import unet  # noqa: F401
from . import my_net  # noqa: F401
```

```yaml
# config snippet
model:
  type: my_net
  params:
    in_channels: 1
    out_channels: 1
    base_channels: 32
```

## Common Pitfalls

- `shot_split` in `inference` must match `data.shot_split` from training.
- `data_range` for SSIM / PSNR must match `normalize_mode`:
  - `max_abs` → SSIM `data_range: 2.0`, PSNR peak (data_range) `1.0`
  - `minmax` → SSIM `data_range: 1.0`, PSNR peak `1.0`
- Missing `from . import my_model` in `model/<task>/__init__.py` causes `KeyError: Unknown model`.
- `segyio` must be installed separately.
- OOM: reduce `batch_size`, `patch_trace`, or `patch_time`.

## Project Rules (from `.cursor/rules/`)

1. memory-first — Read `memory/` before any code change; append important changes to `memory/updates.md`.
2. no-duplication — Search `utils/` and `tools/` before adding new logic.
3. efficiency-first — Avoid Python-level loops when vectorized ops work.
4. research-first — Prefer mature open-source implementations; cite sources.
5. clarify-before-execute — Ask when uncertain; present plan and diff summary; wait for confirmation.
6. no-auto-run — Never execute `python`, `pip`, `torchrun`, or training scripts autonomously.
7. english-only — All project docs, READMEs, source comments, log strings, CLI help, exception messages, config comments, commit messages in English.
8. concise-docs — Docstrings are one sentence + Parameters + Returns only.
