# Seismic Benchmark Tutorial

A hands-on guide to the `seismic-benchmark-code` repository.

## Table of Contents

- [Chapter 1: Introduction](#chapter-1-introduction)
- [Chapter 2: Project Structure and Core Concepts](#chapter-2-project-structure-and-core-concepts)
- [Chapter 3: Quick Start](#chapter-3-quick-start)
- [Chapter 4: Complete End-to-End Example — Data and Preprocessing](#chapter-4-complete-end-to-end-example--data-and-preprocessing)

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

For the random-noise suppression task, the registered models include `unet`, `dncnn`, `res_unet`, `atten_unet`, and `SCRN`. You can switch models by changing only `type` and, if necessary, the model-specific `params`.

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
  checkpoint: /data/liuqi/code/Seismic-bench/seismic-benchmark-code-main/result/checkpoint/best.pt
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

