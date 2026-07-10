# Seismic Benchmark Tutorial

A hands-on guide to the `seismic-benchmark-code` repository.

## Table of Contents

- [Chapter 1: Introduction](#chapter-1-introduction)
- [Chapter 2: Project Structure and Core Concepts](#chapter-2-project-structure-and-core-concepts)
- [Chapter 3: Quick Start](#chapter-3-quick-start)
- [Chapter 4: Complete End-to-End Example — Data and Preprocessing](#chapter-4-complete-end-to-end-example--data-and-preprocessing)
- [Chapter 5: Extending to Other Tasks](#chapter-5-extending-to-other-tasks)
- [Chapter 6: Customizing and Extending the Library](#chapter-6-customizing-and-extending-the-library)
- [Chapter 7: Troubleshooting and Quick Reference](#chapter-7-troubleshooting-and-quick-reference)

---

## Chapter 1: Introduction

### 1.1 What this repository is

`seismic-benchmark-code` is a PyTorch benchmark template for exploration geophysics (seismic) data processing. It supports tasks such as interpolation, denoising, and supervised restoration on volumes stored in SEG-Y, NPY, or MAT formats. The codebase is built around a **registry + factory** pattern: models, datasets, losses, and metrics are registered as plugins and selected from YAML configuration files, so training scripts stay component-agnostic.

### 1.2 Who this tutorial is for

This tutorial is for both:

- **Beginners** who want to run their first seismic-denoising experiment without reading every source file.
- **Experienced practitioners** who need a quick reference for CLI commands, YAML fields, and the registry pattern.

You do not need a deep background in seismology to follow the worked example, but familiarity with deep-learning concepts helps.

### 1.3 Prerequisites

Before you begin, you should be comfortable with:

- Basic PyTorch (`nn.Module`, `DataLoader`, training loops).
- YAML syntax and command-line flags.
- NumPy array shapes and indexing.

Optional but helpful:

- Some exposure to pre-stack seismic shot gathers (SEG-Y, FFID, trace headers).
- A CUDA-capable GPU for training (CPU training is possible but slow).

### 1.4 Dependencies

There is no centralized `requirements.txt` or `pyproject.toml` yet. Install the following packages manually:

- `torch`
- `numpy`
- `matplotlib`
- `pyyaml`
- `segyio`
- `scipy`

You can install them with:

```bash
pip install torch numpy matplotlib pyyaml segyio scipy
```

---

## Chapter 2: Project Structure and Core Concepts

### 2.1 Directory overview

The repository is organized into self-contained directories. Each directory has a single responsibility.

| Directory | Purpose |
|-----------|---------|
| `tools/` | Data utilities: I/O (`array_io.py`, `segy_read.py`), preprocessing (`preprocessing.py`), and patching (`patching.py`). |
| `model/` | Neural network definitions and the `MODEL_REGISTRY`. Task-specific subpackages register their own models. |
| `utils/` | Training infrastructure: datasets, losses, metrics, visualization, logging, optimizer/scheduler builders, training/evaluation loops, and checkpoint I/O. |
| `configs/` | One YAML file per experiment. Hyper-parameters are never hard-coded in source. |
| `scripts/` | CLI entry points for training (`train_*.py`) and inference (`inference_*.py`), plus bash launchers (`*.sh`). |
| `results/` | Experiment outputs: checkpoints, logs, CSVs, and PNGs. This directory is gitignored. |
| `memory/` | Project memory: design decisions, update log, techniques, and research references. |

### 2.2 Registry + factory pattern

The project uses a **registry + factory** pattern so that new components can be added without modifying the training scripts.

Each kind of component has its own registry and decorator:

| Component | Registry file | Decorator | Factory |
|-----------|---------------|-----------|---------|
| Model | `model/registry.py` | `@register_model("name")` | `build_model(cfg)` |
| Dataset | `utils/datasets.py` | `@register_dataset("name")` | `build_dataset(cfg)` |
| Loss | `utils/losses.py` | `@register_loss("name")` | `build_loss(cfg)` |
| Metric | `utils/metrics.py` | `@register_metric("name")` | `build_metrics(cfg)` |

Every pluggable block in YAML follows the same shape:

```yaml
component:
  type: registered_name
  params:
    key: value
```

For example, a UNet model is declared as:

```yaml
model:
  type: unet
  params:
    in_channels: 1
    out_channels: 1
    base_channels: 32
    depth: 4
```

Pseudocode for registering a custom model:

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
from . import unet      # noqa: F401
from . import my_net    # noqa: F401
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

Note: model registration requires adding `from . import <file>  # noqa: F401` to `model/<task>/__init__.py` so the decorator runs at import time. The top-level `model/__init__.py` only exposes registry primitives (`MODEL_REGISTRY`, `register_model`, `build_model`) and a placeholder model; it does **not** import every concrete model file. Task-specific models are registered only when their task subpackage (`model/<task>/`) is imported.

### 2.3 Component-agnostic training scripts

`scripts/train.py` is intentionally component-agnostic. It only parses CLI arguments, loads the YAML config, and wires up the factory functions. It never imports a concrete model, dataset, loss, or metric directly.

Task-specific scripts such as `scripts/random_noise_suppression/train_denoise_unet.py` follow the same pattern: parse CLI, load config, build components, and run the task-specific pipeline. All concrete behavior is driven by the YAML config.

---

## Chapter 3: Quick Start

This section shows you how to train and evaluate a random-noise suppression model in a few commands. The example uses the SEG C3 45-shot synthetic dataset.

### 3.1 Data placement

The default config points to:

```
data/SEG_45Shot_shots1-9.sgy
```

relative to the repository root. The config file also accepts NPY or MAT files by swapping the `data` block in `configs/random_noise_suppression/denoise_unet.yaml`.

### 3.2 Train a model in one command

```bash
python scripts/random_noise_suppression/train_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml
```

The script reads the YAML config, builds the model/loss/dataset/metrics from the registries, and runs the training loop.

### 3.3 Run inference in one command

After training, run inference with the best-validation checkpoint:

```bash
python scripts/random_noise_suppression/inference_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml \
  --checkpoint results/random_noise/random_noise_unet_base/checkpoints/best.pt \
  --output-dir results/random_noise/random_noise_unet_base/inference \
  --noise-kind gaussian \
  --snr-db 5 \
  --n-viz-shots 5
```

You can override noise kind, SNR, batch size, device, and other inference settings from the command line without editing the YAML file.

### 3.4 Expected output tree

After training, the experiment directory contains:

```
results/random_noise/random_noise_unet_base/
├── checkpoints/
│   ├── epoch_0020.pt
│   ├── epoch_0040.pt
│   ├── ...
│   └── best.pt
├── logs/
│   ├── train_log.txt
│   ├── loss_history.csv
│   ├── metrics_history.csv
│   ├── loss_curve.png
│   └── metrics_curve.png
├── visualizations/
│   └── epoch_*.png
└── config_used.yaml
```

After inference, the output directory contains:

```
results/random_noise/random_noise_unet_base/inference/
├── inference.log
├── metrics_summary.json
├── metrics_per_shot.csv
├── visualizations/
│   └── shot_*.png
└── npy/                    # only when --save-npy is set
    ├── input_shots.npy
    ├── pred_shots.npy
    └── target_shots.npy
```

### 3.5 Note on execution

Training and inference scripts are not run automatically by this tutorial. Copy the commands above into your terminal and execute them manually. Training can take minutes to hours depending on GPU, batch size, and epoch count.

---

## Chapter 4: Complete End-to-End Example — Data and Preprocessing

This chapter walks through the data and preprocessing stages of the random-noise suppression example. The YAML config used here is `configs/random_noise_suppression/denoise_unet.yaml`.

### 4.1 Data format and loading

#### SEG-Y reading

The repository reads SEG-Y files via `tools/segy_read.py`. The function used for regular shot gathers is:

```python
read_regular_shots(path, traces_per_shot, time_downsample=1)
```

It returns a NumPy array of shape `(n_shots, n_traces, n_time)` and a dictionary of trace headers. Regularity is verified by checking that each shot slice shares a single `FieldRecord` (FFID) header value.

For the SEG C3 45-shot volume used in this tutorial:

- `n_shots = 9`
- `traces_per_shot = 201`
- `dt = 0.008` s (8 ms sampling interval)

The loaded array shape is `(9, 201, n_time)`.

#### Switching to NPY or MAT in YAML

The `data` block in the config supports one format at a time. The default uses SEG-Y:

```yaml
data:
  segy:
    path: /path/to/SEG_45Shot_shots1-9.sgy
    traces_per_shot: 201
    time_downsample: 1
```

To use an NPY or MAT file instead, uncomment the corresponding block and comment out the SEG-Y block:

```yaml
data:
  # npy:
  #   path: /path/to/SEG_45Shot_shots1-9.npy
  # mat:
  #   path: /path/to/SEG_45Shot_shots1-9.mat
  #   key: shots
```

All three loaders return a volume of shape `(n_shots, n_traces, n_time)` as `float32`. MAT files are loaded with `scipy.io.loadmat`; if the configured `key` is not present in the file, the loader raises a `KeyError` listing the available variable names.

#### Shape convention

Throughout the repository, a seismic volume is stored as:

```
(n_shots, n_traces, n_time)
```

where:

- `n_shots` is the number of shot gathers (or FFIDs).
- `n_traces` is the number of receiver traces per shot.
- `n_time` is the number of time samples.

When a 2D conv model operates on a single shot, the patch shape is `(n_traces, n_time)`, with a singleton channel dimension added for the network: `(1, n_traces, n_time)`.

### 4.2 Preprocessing pipeline

The preprocessing block in `configs/random_noise_suppression/denoise_unet.yaml` defines the transformations applied to the raw volume before training and inference. The same values must be used in both stages.

```yaml
preprocess:
  dt: 0.008
  t0: 0.0
  spherical_power: 0
  noise_kind: gaussian
  snr_db: 5.0
  normalize_mode: max_abs
  normalize_scope: shot
  patch_time: 256
  patch_trace: 128
  patch_overlap: 0.5
  max_shots: null
  skip: ["spherical_divergence_correction"]
```

#### Spherical divergence correction

`spherical_divergence_correction(shots, dt, t0, power)` multiplies each sample by `(t + t0) ** power`, where `t` is the time axis. It compensates for amplitude decay caused by spherical spreading. In this example, `spherical_power: 0` and the step is listed in `skip`, so the correction is disabled.

#### Normalization

`normalize(shots, mode, per)` scales the data into a model-friendly range. The example config uses:

```yaml
normalize_mode: max_abs
normalize_scope: shot
```

This maps each shot to the range `[-1, 1]` by dividing by the maximum absolute value inside that shot. Other supported modes are `minmax` (maps to `[0, 1]`) and `mean_std` (zero mean, unit variance). Other scopes are `trace` and `global`.

`normalize_mode` must agree with the `data_range` settings used by SSIM and PSNR in the `metrics` block. For `max_abs`, SSIM uses `data_range: 2.0` and PSNR uses `data_range: 1.0`. For `minmax`, both use `data_range: 1.0`.

> **Beginner note: why SSIM and PSNR use different `data_range` values**
>
> With `max_abs`, the normalized volume spans `[-1, 1]`, so the full range is `max - min = 1 - (-1) = 2.0`. SSIM expects `data_range` to be this full range, so it is set to `2.0`.
>
> PSNR, on the other hand, is defined in terms of the peak signal amplitude. For `[-1, 1]` data the peak absolute amplitude is `1.0`, so PSNR uses `data_range: 1.0`.
>
> If you switch to `minmax` normalization (`[0, 1]`), the full range and the peak amplitude are both `1.0`, so both SSIM and PSNR use `data_range: 1.0`. Always keep these values consistent with the chosen `normalize_mode`.

#### Synthetic noise injection

Random-noise suppression is trained on synthetically noised versions of the clean volume. The noise step uses:

```python
add_noise(shots, kind="gaussian"|"poisson", snr_db=5.0, rng=None)
```

The SNR is defined in decibels as:

```
SNR_dB = 10 * log10(var_signal / var_noise)
```

The example config uses `noise_kind: gaussian` and `snr_db: 5.0`. Smaller values produce stronger noise. You can override these at inference time with `--noise-kind` and `--snr-db`.

#### Patching

After normalization and noise injection, each shot is cut into overlapping 2D patches for the UNet. The repository uses `tools/patching.py`:

```python
patchify_uniform(data, patch_size=(trace, time), overlap=0.0, output_ndim=3|4)
unpatchify_uniform(patches, info)
```

For this example:

- `patch_size = (128, 256)` (trace, time)
- `patch_overlap = 0.5`
- `output_ndim = 4`, so patches are returned as `(P, 1, 128, 256)` for direct use by `nn.Conv2d` layers.

`unpatchify_uniform` reconstructs the original shot by averaging overlapping regions (`sum / count`). Overlapping patches reduce edge artifacts during full-shot inference.

#### Shot-level split

The dataset is split at the shot level using `data.shot_split`:

```yaml
shot_split:
  train: 7
  val: 1
  test: 1
```

The 9 shots are divided sequentially by FFID: the first 7 shots are used for training, the 8th for validation, and the 9th for testing. This prevents data leakage that could occur if patches from the same shot were placed in both train and test sets.

If `shot_split` is omitted, the code falls back to a patch-level random split.

### 4.3 YAML Config Walkthrough

The full config used for the random-noise suppression example is `configs/random_noise_suppression/denoise_unet.yaml`. Each top-level block maps directly to one stage of the pipeline. Values that affect preprocessing must be identical at training and inference time.

#### Experiment block

```yaml
experiment:
  name: random_noise_unet_base
  output_dir: results/random_noise
  seed: 42
  device: cuda
```

- `name` — the final experiment directory is `output_dir / name`.
- `output_dir` — can be relative to the repo root (e.g. `results/random_noise`) or an absolute path (e.g. `/data/experiments`).
- `seed` — global random seed used by noise injection, shot selection, and data loading.
- `device` — training device; this is overridden by `LOCAL_RANK` when running under `torchrun`.

#### Data block

```yaml
data:
  segy:
    path: /data/liuqi/code/Seismic-bench/seismic-benchmark-code/data/SEG_45Shot_shots1-9.sgy
    traces_per_shot: 201
    time_downsample: 1
  # npy:
  #   path: ...
  # mat:
  #   path: ...
  #   key: shots
  shot_split:
    train: 7
    val: 1
    test: 1
  # test_ratio: 0.1
  loader:
    batch_size: 192
    num_workers: 4
    pin_memory: true
```

Only one format block (`segy`, `npy`, or `mat`) should be active at a time. All loaders return a volume of shape `(n_shots, n_traces, n_time)` as `float32`.

`shot_split` controls the train/val/test split at the shot (FFID) level. When it is present, `test_ratio` is ignored. The split is sequential: the first 7 unique FFIDs go to train, the next to validation, and the last to test. This prevents data leakage from overlapping patches. If `shot_split` is omitted, the code falls back to a patch-level random split.

`loader` sets the training `DataLoader` arguments. Inference can use its own `inference.batch_size` to reduce memory without changing this block.

#### Preprocess block

```yaml
preprocess:
  dt: 0.008
  t0: 0.0
  spherical_power: 0
  noise_kind: gaussian
  snr_db: 5.0
  normalize_mode: max_abs
  normalize_scope: shot
  patch_time: 256
  patch_trace: 128
  patch_overlap: 0.5
  max_shots: null
  skip: ["spherical_divergence_correction"]
```

- `dt` — time sampling interval in seconds, used for spherical-divergence correction and FB-FRE frequency estimation.
- `t0` — reference time offset for the gain `gain = (t + t0) ** power`.
- `spherical_power` — power for spherical-divergence correction. Set to `0` and add the step to `skip` to disable it.
- `noise_kind` — synthetic noise type for random-noise suppression: `"gaussian"` or `"poisson"`.
- `snr_db` — target SNR of the injected noise in dB. Smaller values mean stronger noise.
- `normalize_mode` — `max_abs` maps to `[-1, 1]`, `minmax` maps to `[0, 1]`, `mean_std` maps to zero mean and unit variance.
- `normalize_scope` — whether statistics are computed per `shot`, per `trace`, or `globally`.
- `patch_time` / `patch_trace` — patch size along the time and trace axes.
- `patch_overlap` — overlap ratio for overlapping patches during inference. `0.0` means no overlap.
- `max_shots` — optional limit for quick smoke tests. `null` means use all shots.
- `skip` — list of preprocessing steps to skip. Common examples are `["spherical_divergence_correction"]`, `["normalize"]`, or `["add_noise"]`.

#### Model block

```yaml
model:
  type: unet
  params:
    in_channels: 1
    out_channels: 1
    base_channels: 32
    depth: 4
```

`type` must be a name registered in `MODEL_REGISTRY`. The file `model/random_noise_suppression/__init__.py` imports the task models so the decorators run. `params` is passed straight to the model constructor.

For the random-noise suppression task, the registered models include `unet`, `dncnn`, `res_unet`, `atten_unet`, and `SCRN`. You can switch models by changing only `type` and, if necessary, the model-specific `params`. Note that `SCRN` is also available as a registered model, but it is not included in the default `run_all_random_noise_models.sh` sweep; use its individual `train_denoise_SCRN.sh` and `inference_denoise_SCRN.sh` scripts to run it.

#### Loss, optimizer, and scheduler blocks

```yaml
loss:
  type: mse
  params:
    reduction: mean

optim:
  type: adamw
  params:
    lr: 1.0e-4
    weight_decay: 1.0e-5

scheduler:
  type: cosine
  params:
    min_lr: 1.0e-6
```

- `loss` — registered in `LOSS_REGISTRY`. `mse` with `reduction: mean` is the standard choice for denoising.
- `optim` — registered in the optimizer builder. `adamw` here uses `lr=1e-4` and `weight_decay=1e-5`.
- `scheduler` — cosine annealing from the optimizer's initial LR down to `min_lr`.

#### Metrics block

```yaml
metrics:
  - name: snr
    params: { reduction: per_sample }
  - name: psnr
    params: { data_range: 1.0, reduction: per_sample }
  - name: ssim
    params: { data_range: 2.0, window_size: 11, sigma: 1.5 }
  - name: mae
    params: {}
  - name: mse
    params: {}
  - name: rmse
    params: { reduction: per_sample }
```

These metrics are computed during training and inference. `rmse`, `snr`, and `psnr` support `reduction: per_sample` (mean of per-shot scores) or `reduction: global`.

> **Important:** `data_range` must match the chosen `normalize_mode`.
>
> With `max_abs`, the normalized volume spans `[-1, 1]`. SSIM expects the full peak-to-peak range, so `data_range: 2.0`. PSNR uses the peak absolute amplitude, so `data_range: 1.0`.
>
> With `minmax`, the volume spans `[0, 1]`, so both SSIM and PSNR use `data_range: 1.0`.
>
> If you switch `normalize_mode`, update these two values consistently.

#### Train and log blocks

```yaml
train:
  epochs: 200
  grad_clip: 1.0
  log_step: false
  log_interval: 10
  eval_interval: 1
  ckpt_interval: 20
  vis_interval: 5
  resume: null

log:
  log_dir: logs
  plot_interval: 5
```

- `epochs` — total number of training epochs.
- `grad_clip` — gradient clipping value.
- `log_step` — if `true`, log every training step; otherwise log every `log_interval` steps.
- `log_interval` — how often to print training-batch summaries.
- `eval_interval` — how often to run validation evaluation.
- `ckpt_interval` — how often to save periodic checkpoints (`epoch_*.pt`).
- `vis_interval` — how often to save a random validation visualization.
- `resume` — placeholder for a checkpoint path; the current `train_denoise_unet.py` script does not parse it from the CLI.
- `log_dir` — subdirectory under `output_dir / name` for text and CSV logs.
- `plot_interval` — how often to redraw `loss_curve.png` and `metrics_curve.png`. Set to `0` to disable.

#### Inference block

```yaml
inference:
  data:
    segy:
      path: /data/liuqi/code/Seismic-bench/seismic-benchmark-code/data/SEG_45Shot_shots1-9.sgy
      traces_per_shot: 201
      time_downsample: 1
  shot_split:
    train: 7
    val: 1
    test: 1
  checkpoint: results/random_noise/random_noise_unet_base/checkpoints/best.pt
  output_dir: results/random_noise/random_noise_unet_base/inference
  n_viz_shots: 5
  device: cuda:1
  batch_size: 48
  binned_metrics:
    enabled: true
    eb_wse:
      enabled: true
      bins: [[5, 20], [20, 40], [40, 70], [70, 100]]
      smooth_sigma: 1.0
    fb_fre:
      enabled: true
      rel_threshold: 0.001
      band_ratios: [0.20, 0.30, 0.30, 0.20]
      band_names: ["low", "mid", "high", "very_high"]
      taper_width: 0.0
```

- `inference.data` — optional inference-specific data source. If omitted, the training `data` block is used.
- `inference.shot_split` — must match the training split so the test shot is selected consistently.
- `checkpoint` — path to the model checkpoint to load. Usually `results/<exp>/checkpoints/best.pt`.
- `output_dir` — directory for inference outputs.
- `n_viz_shots` — number of random test shots to visualize.
- `device` — inference device, e.g. `cuda:0` or `cpu`.
- `batch_size` — inference batch size, independent of `data.loader.batch_size`.
- `binned_metrics` — EB-WSE and FB-FRE diagnostics. The `enabled` flag turns the whole subsystem on or off. Individual `eb_wse.enabled` and `fb_fre.enabled` switches control the two metrics.
  - `eb_wse.smooth_sigma` — Gaussian smoothing sigma applied to the energy map before binning.
  - `fb_fre.rel_threshold` — relative power threshold for keeping a frequency in the effective band.
  - `fb_fre.taper_width` — cosine taper width in Hz at band edges.

### 4.4 Training in Detail

#### Single-GPU command

```bash
python scripts/random_noise_suppression/train_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml
```

The script reads the YAML file, builds the model/loss/dataset/metrics from the registries, and runs the training loop. It does not require editing the script itself.

#### Output directory structure

Training artifacts are written to `results/<experiment.output_dir>/<experiment.name>/`. For the default config, that is:

```
results/random_noise/random_noise_unet_base/
├── checkpoints/
│   ├── epoch_0020.pt
│   ├── epoch_0040.pt
│   ├── ...
│   └── best.pt
├── logs/
│   ├── train_log.txt
│   ├── loss_history.csv
│   ├── metrics_history.csv
│   ├── loss_curve.png
│   └── metrics_curve.png
├── visualizations/
│   └── epoch_*.png
└── config_used.yaml
```

- `checkpoints/` — periodic checkpoints (`epoch_*.pt`) and `best.pt` (lowest validation loss).
- `logs/` — human-readable log, CSV histories, and auto-refreshed curve plots.
- `visualizations/` — random validation samples saved every `vis_interval` epochs.
- `config_used.yaml` — a copy of the resolved config for reproducibility.

#### Log files and curve images

`logs/train_log.txt` contains timestamped one-line summaries per epoch. `logs/loss_history.csv` has columns `epoch, lr, train, val`, and `logs/metrics_history.csv` has columns `epoch, train_<metric>, val_<metric>`. The `TrainingLogger` rehydrates any existing CSVs when resuming, so curves stay continuous across restarts.

#### Resuming from a checkpoint

At the time of writing, `train_denoise_unet.py` only parses `--config`; a `--resume` CLI flag is not yet implemented. If you need to resume, you can either:

1. Load the checkpoint inside the script before the epoch loop by calling `load_checkpoint(...)` from `utils.train_utils`.
2. Add a `--resume` argument that calls `load_checkpoint(...)` and restores the optimizer/scheduler state.

The expected command after adding resume support would look like:

```bash
python scripts/random_noise_suppression/train_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml \
  --resume results/random_noise/random_noise_unet_base/checkpoints/epoch_0020.pt
```

#### Multi-GPU command

Multi-GPU training uses `torchrun` with one process per GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  scripts/random_noise_suppression/train_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml
```

When `WORLD_SIZE > 1`, the script automatically:

- Wraps the model with `DistributedDataParallel`.
- Uses `DistributedSampler` for the training loader.
- Performs rank-0-only checkpointing, logging, and visualization.
- All-reduces the training loss across processes.

`experiment.device` is ignored in distributed mode; the process uses `cuda:LOCAL_RANK`.

> **Note:** Do not run training scripts automatically from this tutorial. Copy the commands above into your terminal and execute them manually. Training time depends on GPU, batch size, and epoch count.

### 4.5 Inference and Evaluation

#### Command with explicit overrides

```bash
python scripts/random_noise_suppression/inference_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml \
  --checkpoint results/random_noise/random_noise_unet_base/checkpoints/best.pt \
  --output-dir results/random_noise/random_noise_unet_base/inference \
  --noise-kind gaussian \
  --snr-db 5 \
  --n-viz-shots 5 \
  --device cuda:0 \
  --batch-size 48
```

All overrides are optional. If omitted, the script falls back to the values in the `inference` block or the `preprocess` block. The available CLI arguments are:

- `--checkpoint` — path to the `.pt` checkpoint (required if `inference.checkpoint` is not set).
- `--output-dir` — directory for inference outputs.
- `--n-viz-shots` — number of random shots to visualize.
- `--seed` — random seed for shot selection and noise injection.
- `--device` — inference device.
- `--batch-size` — inference batch size.
- `--save-npy` — save `input_shots.npy`, `pred_shots.npy`, and `target_shots.npy`.
- `--noise-kind` — override `preprocess.noise_kind`.
- `--snr-db` — override `preprocess.snr_db`.

#### Step-by-step inference flow

The inference script performs the following steps on the test shots:

1. **Load the raw volume** using `tools.array_io.load_volume`, which dispatches by file extension to SEG-Y, NPY, or MAT readers.
2. **Select the test split** using `inference.shot_split`. The same sequential FFID ordering as training is used, so the held-out test shot is selected consistently.
3. **Apply the same preprocessing** as training: spherical-divergence correction (if not skipped), normalization, and synthetic noise injection. Noise is injected with the CLI-overridden `noise_kind` and `snr_db`.
4. **Patchify** the noisy test shots into overlapping `(1, patch_trace, patch_time)` patches.
5. **Run the model forward** in batches on the selected device.
6. **Unpatchify** the outputs with overlap averaging (`sum / count`) to reconstruct full shots.
7. **Apply inverse transforms** to return the predictions, noisy inputs, and clean targets to the original amplitude domain. Metrics are computed on the normalized domain before the inverse transform, so the reported values align with training.
8. **Compute metrics** and **save visualizations**.

#### Metric groups

`metrics_summary.json` contains three groups of metrics:

- `noisy` — the noisy input compared against the clean target. This is the baseline.
- `denoised` — the model prediction compared against the clean target.
- `delta` — the difference `denoised - noisy`, showing how much the model improved (or degraded) each metric.

`metrics_per_shot.csv` contains the same metrics evaluated per shot, with columns prefixed by `noisy_`, `denoised_`, and `delta_`.

#### EB-WSE and FB-FRE

The binned diagnostics are enabled by `inference.binned_metrics.enabled`.

- **EB-WSE (Energy-Binned Weak Signal Evaluation)** computes normalized error (NE) and SNR inside reference-energy percentile bins. The default bins are `[5, 20]`, `[20, 40]`, `[40, 70]`, and `[70, 100]`, corresponding to very weak, weak, moderate, and strong signal regions. Output keys look like `eb_wse_very_weak_5_20_ne` and `eb_wse_very_weak_5_20_snr`.

- **FB-FRE (Frequency-Binned Fidelity and Recovery Evaluation)** estimates an effective frequency band from the reference spectrum, splits it into adaptive low/mid/high/very_high bands according to `band_ratios`, and computes NE and SNR per band. Output keys look like `fb_fre_low_ne`, `fb_fre_low_snr`, `fb_fre_low_energy_ratio`, and `fb_fre_low_frequency_range_hz`.

Both metrics are computed on the normalized domain and written as mean values into `metrics_summary.json` under the `noisy` and `denoised` groups. Their deltas are also reported in the `delta` group.

#### Output files

After inference, the output directory contains:

```
results/random_noise/random_noise_unet_base/inference/
├── inference.log
├── metrics_summary.json
├── metrics_per_shot.csv
├── visualizations/
│   └── shot_*.png
└── npy/                    # only when --save-npy is set
    ├── input_shots.npy
    ├── pred_shots.npy
    └── target_shots.npy
```

- `inference.log` — stdout and stderr from the inference run.
- `metrics_summary.json` — mean scalar metrics and binned metrics for `noisy`, `denoised`, and `delta` groups.
- `metrics_per_shot.csv` — per-shot scalar metrics.
- `visualizations/shot_*.png` — side-by-side panels of input, prediction, target, and residual for each visualized shot.
- `npy/` — optional NumPy arrays saved when `--save-npy` is passed.

### 4.6 Batch Sweeps

For systematic benchmarking, the repository provides shell launchers that sweep over noise kinds, SNR levels, and seeds without manual YAML edits.

#### `train_denoise_unet.sh`

`scripts/random_noise_suppression/train_denoise_unet.sh` rewrites a temporary copy of the base config for each combination of noise kind, SNR, and seed, then launches `torchrun`.

The editable block at the top of the script is:

```bash
CUDA_VISIBLE_DEVICES="0,1" # Physical GPU ids, comma-separated.
NPROC_PER_NODE=2           # Should match the number of visible GPUs.
N=3                        # Number of runs per (noise_kind, snr) pair.
START_SEED=42              # First seed; later runs use START_SEED+1, ...
NOISE_KIND_LIST=("gaussian" "poisson")
SNR_LIST=(-5 0 5)          # Synthetic noise SNR values (dB).
TORCHRUN_EXTRA=""          # Optional extra torchrun args, e.g. "--standalone".
```

The default sweep runs:

- 2 noise kinds × 3 SNR levels × 3 seeds = 18 training runs.

Each run gets a unique experiment name such as `random_noise_unet_base_gaussian_snr5_seed42`, so outputs never collide.

#### `inference_denoise_unet.sh`

`scripts/random_noise_suppression/inference_denoise_unet.sh` mirrors the same sweep for inference. It loops over noise kinds, SNRs, and seeds, runs the inference script for each trained checkpoint, and then aggregates the results across seeds.

The default configuration block is:

```bash
DEVICE="cuda:0"
NOISE_KIND_LIST=("gaussian" "poisson")
SNR_LIST=(-5 0 5)
N=3                         # Number of runs / seeds
START_SEED=42               # Seeds: START_SEED, START_SEED+1, ...
N_VIZ_SHOTS=5
SAVE_NPY=0                  # 1 = save .npy outputs, 0 = skip
CHECKPOINT_NAME="best.pt"   # e.g. "best.pt" or "epoch_0049.pt"
```

After the per-run inference loop finishes, the script aggregates the `metrics_summary.json` files from all seeds and writes a mean/standard-deviation summary to:

```
results/random_noise/random_noise_unet_base_<noise_kind>_snr<tag>_seed_stats/metrics_summary_mean_std.json
```

The `<tag>` is `neg5` for `-5` dB and `5` for `5` dB, matching the training script naming convention.

#### `run_all_random_noise_models.sh`

`scripts/random_noise_suppression/run_all_random_noise_models.sh` runs the full training and inference sweep for four model families sequentially:

```bash
MODEL_LIST=("unet" "dncnn" "res_unet" "atten_unet")
```

For each model, it looks for `scripts/random_noise_suppression/train_denoise_${model}.sh` and `scripts/random_noise_suppression/inference_denoise_${model}.sh`, runs the training sweep, and then runs the inference sweep. The script logs everything to `scripts/random_noise_suppression/run_all_random_noise_models.log`. If `STOP_ON_ERROR=1`, the script exits immediately when any stage fails.

This is a convenient way to produce a full benchmark across architectures, but it takes a long time because each stage runs sequentially.

#### Minimal manual loop for inference over SNRs

If you prefer to run a small inference sweep by hand, a minimal bash loop is:

```bash
CHECKPOINT=results/random_noise/random_noise_unet_base/checkpoints/best.pt
CONFIG=configs/random_noise_suppression/denoise_unet.yaml
for snr in -5 0 5; do
  python scripts/random_noise_suppression/inference_denoise_unet.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "results/random_noise/random_noise_unet_base/inference_snr${snr}" \
    --noise-kind gaussian \
    --snr-db "${snr}" \
    --n-viz-shots 5 \
    --device cuda:0
done
```

This writes each SNR level to its own output directory and is useful for quick comparisons without editing the shell sweep.


## Chapter 5: Extending to Other Tasks

The repository ships with four task families. They all use the same registry + factory backbone, but the way training pairs are generated differs.

### 5.1 Task comparison

| Task | Input Data | Entry Scripts | Config Directory | Key Differences |
|------|------------|---------------|------------------|-----------------|
| `random_noise_suppression` | Clean volume + synthetic noise | `scripts/random_noise_suppression/train_denoise_*.py`, `inference_denoise_*.py` | `configs/random_noise_suppression/` | Noise is injected with `add_noise`; metrics compare the denoised output to the clean target. |
| `ground_roll_attenuation` | Paired noisy / noise-label volumes | `scripts/ground_roll_attenuation/train_denoise_*.py`, `batch_evaluate.py` | `configs/ground_roll_attenuation/` | No synthetic noise injection; the model predicts the additive noise label; the `data` block uses `segy_pair` (or `npy_pair` / `mat_pair`). |
| `multiples_attenuation` | Paired noisy / noise-label volumes | `scripts/multiples_attenuation/train_denoise_*.py`, `batch_evaluate.py` | `configs/multiples_attenuation/` | Same shape as ground-roll attenuation; task-specific data and semantics. |
| `interpolation` | Single volume + trace masking | `scripts/interpolation/train_interpolation_unet.py`, `inference_interpolation.py` | `configs/interpolation/` | `mask_traces` simulates missing traces; the model reconstructs the full shot gather. |

### 5.2 random_noise_suppression

This is the task covered in Chapter 4.

### 5.3 ground_roll_attenuation

Ground-roll attenuation is trained on **paired volumes**: a noisy input volume and a corresponding noise-label volume (the additive noise component). The model learns to predict the noise map; the denoised estimate is `noisy_input - predicted_noise`.

Train a U-Net baseline:

```bash
python scripts/ground_roll_attenuation/train_denoise_unet.py \
  --config configs/ground_roll_attenuation/denoise_unet.yaml
```

After training, run the batch evaluator on the experiment directory tree:

```bash
python scripts/ground_roll_attenuation/batch_evaluate.py \
  --root_dir results/ground_roll_attenuation \
  --output results/ground_roll_attenuation/batch_evaluation.xlsx \
  --device cuda:0 \
  --batch_size 8
```

`batch_evaluate.py` scans each experiment directory, loads `checkpoints/best.pt`, runs inference on the held-out `test_set/`, and writes an Excel workbook with one sheet per noise level. The workbook compares raw-input metrics (noisy vs reference) and denoised metrics (model output vs reference).

> **Note:** `batch_evaluate.py` requires `openpyxl`: `pip install openpyxl`.

The ground-roll config uses a `data.segy_pair` block (NPY/MAT variants are `npy_pair` / `mat_pair`):

```yaml
data:
  segy_pair:
    input_path: /path/to/noisy.sgy
    target_path: /path/to/noise_label.sgy
    traces_per_shot: 201
    time_downsample: 1
```

The two volumes must have the same shape after loading.

### 5.4 multiples_attenuation

Multiples attenuation follows the same paired-volume setup as ground-roll attenuation; only the data and the physical meaning of the noise label differ.

Train a U-Net baseline:

```bash
python scripts/multiples_attenuation/train_denoise_unet.py \
  --config configs/multiples_attenuation/denoise_unet.yaml
```

Run the batch evaluator:

```bash
python scripts/multiples_attenuation/batch_evaluate.py \
  --root_dir results/multiples_attenuation \
  --output results/multiples_attenuation/batch_evaluation.xlsx \
  --device cuda:0 \
  --batch_size 8
```

### 5.5 interpolation

Interpolation trains a model to reconstruct missing traces. A single volume is loaded, masked along the trace axis, and the model learns to recover the original traces.

Train a U-Net baseline with uniform 50% missing traces:

```bash
python scripts/interpolation/train_interpolation_unet.py \
  --config configs/interpolation/interpolation_unet.yaml \
  --mask-mode uniform \
  --mask-ratio 0.5
```

The training script appends the masking parameters to the experiment name, so the output directory for the run above becomes `results/interp_unet_base_uniform_miss50/`.

Run inference with the masked checkpoint:

```bash
python scripts/interpolation/inference_interpolation.py \
  --config configs/interpolation/interpolation_unet.yaml \
  --checkpoint results/interp_unet_base_uniform_miss50/checkpoints/epoch_0049.pt \
  --output-dir results/interp_unet_base_uniform_miss50/inference \
  --n-viz-shots 5 \
  --device cuda:0
```

Interpolation-specific YAML fields:

- `preprocess.mask_mode` (or CLI `--mask-mode`): `uniform`, `random`, or `continuous`.
- `preprocess.mask_ratio` (or CLI `--mask-ratio`): fraction of traces to mask in `(0, 1)`.
- `preprocess.uniform_stride`: only used when `mask_mode` is `uniform`; keeps every `uniform_stride`-th trace. For example, `uniform_stride: 2` removes every other trace.
- `preprocess.spherical_power`: often enabled for interpolation (e.g., `1.2`) to compensate for spherical divergence before masking and normalization.

The interpolation task is the only one of the four that uses `mask_traces` instead of `add_noise` or paired noise labels. The other YAML blocks (`model`, `loss`, `metrics`, `optim`, `scheduler`, `train`, `log`) follow the same registry pattern as Chapter 4.

---

## Chapter 6: Customizing and Extending the Library

All pluggable components are added through the registry + factory pattern described in Chapter 2. This chapter shows the concrete steps for each component type.

### 6.1 Adding a New Model

Models live in `model/<task>/` and are registered with the `MODEL_REGISTRY`.

Steps:

1. Create `model/<task>/my_model.py`.
2. Inherit from `nn.Module`.
3. Decorate the class with `@register_model("my_model")`.
4. Add `from . import my_model  # noqa: F401` to `model/<task>/__init__.py` so the decorator runs at import time.
5. Reference the new model in the YAML config.

Minimal example:

```python
# model/random_noise_suppression/my_model.py
import torch.nn as nn
from ..registry import register_model


@register_model("my_model")
class MyModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv(x)
```

```python
# model/random_noise_suppression/__init__.py
from . import unet       # noqa: F401
from . import my_model     # noqa: F401
```

```yaml
# config snippet
model:
  type: my_model
  params:
    in_channels: 1
    out_channels: 1
    base_channels: 32
```

The same pattern applies to `model/ground_roll_attenuation/`, `model/multiples_attenuation/`, and `model/interpolation/`. Note that the top-level `model/__init__.py` does **not** import every task; each task subpackage has its own registry view, so scripts import `build_model` from the appropriate task (e.g., `from model.ground_roll_attenuation import build_model`).

### 6.2 Adding a New Loss

Losses are registered in `utils/losses.py`.

Steps:

1. Inherit from `BaseLoss`.
2. Implement `forward(self, pred, target=None, **extras)`.
3. Decorate with `@register_loss("my_loss")`.

The `extras` dict is passed by the training loop and can carry optional masks or weights.

Example:

```python
# utils/losses.py
@register_loss("my_loss")
class MyLoss(BaseLoss):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target=None, **extras):
        if target is None:
            raise ValueError("MyLoss requires target.")
        return self.weight * (pred - target).abs().mean()
```

YAML:

```yaml
loss:
  type: my_loss
  params:
    weight: 1.0
```

### 6.3 Adding a New Metric

Metrics are registered in `utils/metrics.py`.

Steps:

1. Inherit from `BaseMetric`.
2. Implement `__call__(self, pred, target)` returning a Python `float`.
3. Set `higher_is_better` appropriately.
4. Decorate with `@register_metric("my_metric")`.

Reduction modes:

- `reduction="per_sample"` (default): compute the metric independently for each sample in the leading batch dimension, then average across the batch. This matches the common seismic convention of reporting a mean per-shot SNR or PSNR.
- `reduction="global"`: pool all elements first, then apply any non-linear operation (e.g., `sqrt` or `log10`). This preserves textbook identities such as `RMSE == sqrt(MSE)` and `PSNR == 10*log10(peak^2 / MSE)`.

Example:

```python
# utils/metrics.py
@register_metric("my_metric")
class MyMetric(BaseMetric):
    """Mean absolute error (example custom metric)."""

    higher_is_better = False

    def __call__(self, pred, target):
        pred, target = _prepare(pred, target, "MyMetric")
        return float((pred - target).abs().mean())
```

YAML:

```yaml
metrics:
  - name: my_metric
    params: {}
```

### 6.4 Adding a New Dataset

Dataset classes live in `utils/datasets.py` and inherit from `BaseArrayDataset`.

Required overrides:

- `_build_index()`: scan `self.root` and populate `self._index` with `Path` objects.
- `_load_sample(path)`: return `(input_tensor, target_tensor_or_none)`. Both should be CPU tensors.

The dataset expects seismic volumes with the standard shape `(n_shots, n_traces, n_time)`.

Example:

```python
# utils/datasets.py
@register_dataset("my_dataset")
class MyDataset(BaseArrayDataset):
    """Dataset backed by custom NumPy files."""

    def _build_index(self):
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        self._index = sorted(self.root.rglob("*.npy"))
        if not self._index:
            raise FileNotFoundError(f"No .npy files found under {self.root}")

    def _load_sample(self, path):
        arr = np.load(path)
        x = torch.from_numpy(np.asarray(arr)).float()
        return x, None
```

YAML:

```yaml
data:
  type: my_dataset
  params:
    root: /path/to/data
  loader:
    batch_size: 8
    num_workers: 4
    pin_memory: true
```

### 6.5 Adding a New Preprocessing Step

New preprocessing functions should be added to `tools/preprocessing.py` as pure NumPy operations on `(n_shots, n_traces, n_time)` or `(n_traces, n_time)`.

Steps:

1. Implement the function and, if the step is optional, add it to the `skip` mechanism in the task-specific training script.
2. Add the configuration field to the YAML `preprocess` block.
3. Call the function from the task-specific script's preprocessing function (e.g., `_preprocess_shots` in `scripts/interpolation/train_interpolation_unet.py`).

Example:

```python
# tools/preprocessing.py
def scale_amplitude(shots, scale=1.0):
    """Scale amplitudes by a constant factor."""
    return shots * scale, {"scale": scale}
```

Then in the relevant training script:

```python
if "scale_amplitude" not in skip:
    shots, _ = scale_amplitude(shots, scale=float(prep.get("amplitude_scale", 1.0)))
```

YAML:

```yaml
preprocess:
  amplitude_scale: 1.0
  skip: []
```

When adding a step that changes the amplitude scale, remember to update `normalize_mode` and metric `data_range` values consistently.

---

## Chapter 7: Troubleshooting and Quick Reference

### 7.1 Troubleshooting and FAQ

**Checkpoint not found / path issues**

- Verify that the checkpoint file exists. For the random-noise example the default is `results/random_noise/<experiment.name>/checkpoints/best.pt`.
- For `ground_roll_attenuation` and `multiples_attenuation`, `batch_evaluate.py` requires both `checkpoints/best.pt` and a `test_set/` directory in every experiment directory it scans.
- If you train interpolation with `--mask-mode` / `--mask-ratio`, the experiment name is auto-suffixed (e.g., `interp_unet_base_uniform_miss50`), so the checkpoint path changes accordingly.

**`segyio` not installed or SEG-Y path wrong**

- Install the dependency: `pip install segyio`.
- Confirm the file exists: `ls /path/to/volume.sgy`.
- Check that `traces_per_shot` and `time_downsample` in the config match the actual file geometry.
- For paired tasks, the noisy input and the noise label must have the same trace count, sample count, and FFID ordering.

**Out-of-memory**

- Reduce `data.loader.batch_size` or `inference.batch_size`.
- Reduce `preprocess.patch_trace` or `preprocess.patch_time`.
- Reduce `inference.n_viz_shots` if visualization is slow.
- Use a smaller model (`base_channels`, `depth`) for memory-constrained GPUs.

**SSIM / PSNR `data_range` mismatch with `normalize_mode`**

- `max_abs` normalizes to `[-1, 1]`: SSIM needs `data_range: 2.0`, PSNR needs `data_range: 1.0`.
- `minmax` normalizes to `[0, 1]`: both SSIM and PSNR use `data_range: 1.0`.
- `mean_std` is unbounded; set the ranges from the actual target volume or keep them in the metric params.

**`shot_split` inconsistency between training and inference**

- The `inference.shot_split` block must match `data.shot_split` from training so the same test shot is selected.
- If training used a patch-level split (`test_ratio` instead of `shot_split`), do not add `inference.shot_split` at inference time.
- The split is based on sequential FFIDs, not arbitrary shot indices.

**Model not registered: missing import or decorator typo**

- Check the registry contents: `from model.<task> import MODEL_REGISTRY; print(sorted(MODEL_REGISTRY))`.
- Verify that `model/<task>/__init__.py` contains `from . import my_model  # noqa: F401`.
- Verify the decorator name exactly matches `model.type` in the YAML config (case-sensitive).
- Verify the script imports `build_model` from the correct task subpackage (e.g., `from model.ground_roll_attenuation import build_model`).

### 7.2 Quick Reference Cards

#### CLI command cheat sheet

| Task | Train | Inference |
|------|-------|-----------|
| `random_noise_suppression` | `python scripts/random_noise_suppression/train_denoise_unet.py --config configs/random_noise_suppression/denoise_unet.yaml` | `python scripts/random_noise_suppression/inference_denoise_unet.py --config configs/random_noise_suppression/denoise_unet.yaml --checkpoint results/random_noise/<exp>/checkpoints/best.pt` |
| `ground_roll_attenuation` | `python scripts/ground_roll_attenuation/train_denoise_unet.py --config configs/ground_roll_attenuation/denoise_unet.yaml` | `python scripts/ground_roll_attenuation/batch_evaluate.py --root_dir results/ground_roll_attenuation --output results/ground_roll_attenuation/batch_evaluation.xlsx` |
| `multiples_attenuation` | `python scripts/multiples_attenuation/train_denoise_unet.py --config configs/multiples_attenuation/denoise_unet.yaml` | `python scripts/multiples_attenuation/batch_evaluate.py --root_dir results/multiples_attenuation --output results/multiples_attenuation/batch_evaluation.xlsx` |
| `interpolation` | `python scripts/interpolation/train_interpolation_unet.py --config configs/interpolation/interpolation_unet.yaml --mask-mode uniform --mask-ratio 0.5` | `python scripts/interpolation/inference_interpolation.py --config configs/interpolation/interpolation_unet.yaml --checkpoint results/interp_unet_base_uniform_miss50/checkpoints/epoch_0049.pt` |

#### YAML top-level keys and common fields

| Key | Common fields |
|-----|---------------|
| `experiment` | `name`, `output_dir`, `seed`, `device` |
| `data` | Source (`segy`, `npy`, `mat`, `segy_pair`, `npy_pair`, `mat_pair`), `shot_split`, `loader` (`batch_size`, `num_workers`, `pin_memory`) |
| `preprocess` | `dt`, `t0`, `spherical_power`, `normalize_mode`, `normalize_scope`, `clip_percentile`, `patch_time`, `patch_trace`, `patch_overlap`, `max_shots`, `skip`, `noise_kind`, `snr_db`, `mask_mode`, `mask_ratio`, `uniform_stride` |
| `model` | `type`, `params` |
| `loss` | `type`, `params` |
| `metrics` | List of `{name, params}` |
| `optim` | `type`, `params` |
| `scheduler` | `type`, `params` |
| `train` | `epochs`, `grad_clip`, `log_step`, `log_interval`, `eval_interval`, `ckpt_interval`, `vis_interval`, `resume` |
| `log` | `log_dir`, `plot_interval` |
| `inference` | `data`, `shot_split`, `checkpoint`, `output_dir`, `n_viz_shots`, `device`, `batch_size`, `save_npy`, `binned_metrics` |

#### Registry decorator / factory / base class cheat sheet

| Kind | Decorator | Base class | Factory | Registry |
|------|-----------|------------|---------|----------|
| Model | `@register_model("name")` | `nn.Module` | `build_model(cfg)` | `MODEL_REGISTRY` |
| Loss | `@register_loss("name")` | `BaseLoss` | `build_loss(cfg)` | `LOSS_REGISTRY` |
| Metric | `@register_metric("name")` | `BaseMetric` | `build_metrics(cfg_list)` | `METRIC_REGISTRY` |
| Dataset | `@register_dataset("name")` | `BaseArrayDataset` | `build_dataset(cfg)` / `build_dataloader(cfg)` | `DATASET_REGISTRY` |

#### Metric arguments cheat sheet

| Metric | Arguments | Notes |
|--------|-----------|-------|
| `mse` | — | Global mean over all elements. |
| `mae` | — | Global mean over all elements. |
| `rmse` | `reduction: per_sample \| global` | `global` preserves `RMSE == sqrt(MSE)`. |
| `snr` | `reduction`, `eps`, `min_signal_energy` | Signal-to-noise ratio in dB. |
| `psnr` | `data_range`, `reduction`, `eps` | Peak amplitude of the reference signal. |
| `ssim` | `data_range`, `window_size`, `sigma`, `k1`, `k2` | Peak-to-peak range of the reference signal. |


