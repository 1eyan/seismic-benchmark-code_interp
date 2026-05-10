# Coherent Noise Attenuation Benchmark

## Task Overview

Suppress coherent ground-roll noise from pre-stack seismic shot gathers using four deep-learning architectures: **UNet**, **ResUNet**, **DnCNN**, and **Attention UNet**. The 9-shot SEG-C3 dataset is synthetic, generated via forward modeling on the official SEG C3 velocity model — reflection signals are modeled with the acoustic wave equation, while ground-roll noise is modeled with the elastic wave equation to capture its dispersive, low-velocity character. Given a noisy shot gather, the model predicts the additive noise component; the denoised signal is obtained by subtracting the prediction from the input. This is a **paired regression** task trained with a noise-label objective (the ground-truth noise component).

Supervision signal: `denoised = noisy_input - predicted_noise`, evaluated against `clean_reference = noisy_input - label_noise`.

## Dataset

### Source

SEG-C3 pre-stack data, 9 regular shot gathers (`shots1-9`), stored in SEG-Y format.

### Geometry

| Property | Value |
|----------|-------|
| Traces per shot | 201 |
| Time samples | 625 |
| Sampling interval (dt) | 2 ms |
| Shot shape | `(n_shots=9, n_traces=201, n_time=625)` |

### Noise Intensity Levels

Ground-roll noise at five intensity levels injected into the clean signal, producing paired `(noisy, noise_label)` shot gathers:

| Noise level | Noisy file | Noise-label file | SNR (dB) | PSNR (dB) | SSIM | MAE | MSE | RMSE |
|-------------|-----------|-----------------|----------|-----------|------|-----|-----|------|
| 1.0 | `SEGC3_shots1_9_noisy_1.0.sgy` | `SEGC3_shots1_9_noise_1.0.sgy` | 2.71 | 15.81 | 0.9480 | 0.030624 | 0.026232 | 0.161964 |
| 3.0 | `SEGC3_shots1_9_noisy_3.0.sgy` | `SEGC3_shots1_9_noise_3.0.sgy` | -6.83 | 6.27 | 0.9418 | 0.091871 | 0.236091 | 0.485892 |
| 5.0 | `SEGC3_shots1_9_noisy_5.0.sgy` | `SEGC3_shots1_9_noise_5.0.sgy` | -11.27 | 1.83 | 0.9402 | 0.153118 | 0.655809 | 0.809820 |
| 7.0 | `SEGC3_shots1_9_noisy_7.0.sgy` | `SEGC3_shots1_9_noise_7.0.sgy` | -14.19 | -1.09 | 0.9395 | 0.214366 | 1.285386 | 1.133749 |
| 9.0 | `SEGC3_shots1_9_noisy_9.0.sgy` | `SEGC3_shots1_9_noise_9.0.sgy` | -16.37 | -3.27 | 0.9390 | 0.275613 | 2.124822 | 1.457677 |

Initial metrics computed on the full 9-shot dataset in the original amplitude domain (noisy input vs. clean reference `input - noise_label`).

## Data Preprocessing

### Normalization

- **Mode**: `max_abs` — each dataset scaled to `[-1, 1]` by dividing by `max(|x|)`.
- **Scope**: `global` — statistics are computed once over the full noisy dataset; the same scalars are applied to the noise-label target so both stay in a consistent amplitude range.
- **Clipping**: None (optional percentile clipping available but not active).

### Patchify

Overlapping 2D patches of size `(trace=128, time=256)` extracted with a uniform sliding grid, 50% overlap. Each shot gather produces many patches; all 9 shots are patchified and pooled.

Output shape per patch: `(1, 128, 256)` — channel-last for 2D ConvNet consumption.

### Train / Validation / Test Split

**Shot-level (FFID) sequential split** — 7 : 1 : 1.

| Split | FFIDs | Shots | Purpose |
|-------|-------|-------|---------|
| Train | 1–7 | 7 | Model training |
| Validation | 8 | 1 | Early stopping, best-checkpoint selection |
| Test | 9 | 1 | Held-out final evaluation |

The split boundary is at the shot (FFID) level, so no patches from the same shot appear in multiple splits. The raw test-set shots are saved as `.npy` alongside the checkpoint for downstream inference.

## Models

Four architectures are evaluated. All models operate on `(1, H, W)` single-channel 2D patches. The output is the predicted noise residual of the same shape.

| Model | Registry Key | Base Channels | Depth | Key Parameters | Reference |
|-------|-------------|--------------|-------|----------------|-----------|
| **UNet** | `unet` | 32 | 4 | DoubleConv encoder/decoder, MaxPool ×2 down, ConvTranspose ×2 up, skip connections | Ronneberger et al., 2015 |
| **ResUNet** | `res_unet` | 32 | 4 | Same U-Net topology; residual blocks (`Conv→BN→ReLU→Conv→BN + shortcut`) replace plain double-conv blocks | He et al., 2016 (residual block); Zhang et al., 2018 (ResUNet) |
| **DnCNN** | `dncnn` | 64 | 17 | Flat Conv→BN→ReLU stack, residual output `x - F(x)`, orthogonal weight init | Zhang et al., 2017 (IEEE TIP) |
| **Attention UNet** | `atten_unet` | 32 | 4 | U-Net + additive attention gates (`W_g·g + W_x·x → sigmoid → gate`) on skip connections | Oktay et al., 2018 (MIDL) |

### Architecture Details

**UNet** — Classic encoder-decoder. Each stage is two `Conv2d(3×3)→BN→ReLU` blocks. Downsampling: `MaxPool2d(2×2)`. Upsampling: `ConvTranspose2d(2×2)`. Skip connections concatenate encoder features onto decoder inputs. Final layer: `Conv2d(1×1)`.

**ResUNet** — Identical U-Net skeleton with `_DoubleConv` blocks replaced by residual blocks that add an identity (or 1×1-projected) shortcut before the final ReLU activation. Improves gradient flow through deeper encoder/decoder stages.

**DnCNN** — No encoder-decoder; a flat stack of 17 convolutional layers with BatchNorm and ReLU. The first layer uses bias, subsequent layers are bias-free following the original paper. The model learns the noise residual: `output = x - net(x)`. Deeper than the U-Nets but has no spatial downsampling, preserving full resolution throughout.

**Attention UNet** — Same encoder-decoder as UNet, with additive attention gates inserted between each upsampled feature map and its corresponding skip connection. The gate computes: `α = σ(ψ(ReLU(W_g·g + W_x·x)))` and gates the skip: `skip' = skip ⊙ α`. The gating bottleneck uses `F_int = max(c // 2, 8)` channels.

## Training Configuration

### Loss Function

Mean Squared Error (MSE) between predicted noise and label noise:

`L = (1/N) Σ (pred_noise - label_noise)²`

### Optimizer & Scheduler

| Component | Type | Parameters |
|-----------|------|------------|
| Optimizer | AdamW | `lr=1e-3`, `weight_decay=1e-4` |
| Scheduler | Cosine annealing | `min_lr=1e-6`, over full epoch budget |

### Per-Model Batch Size

| Model | Batch Size | Notes |
|-------|-----------|-------|
| UNet | 196 | Lightweight; fits more patches per GPU step |
| ResUNet | 196 | Residual blocks add memory overhead |
| DnCNN | 196 | 17-layer deep stack; largest memory footprint |
| Attention UNet | 196 | Similar overhead to baseline UNet despite attention gates |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Gradient clipping | 1.0 (max norm) |
| Evaluation interval | Every epoch |
| Checkpoint interval | Every 20 epochs |
| Visualization interval | Every 5 epochs |
| Best checkpoint | Minimum validation loss |

### Reproducibility

- **Seeds**: 3 independent runs per model per noise level (seed = 42, 43, 44).
- **RNG**: Python `random`, NumPy, and PyTorch RNGs are all seeded from `experiment.seed`.
- **Data split**: Deterministic by FFID ordering (same split for all seeds and noise levels).
- **Distributed training**: 2 × NVIDIA RTX 4090 (24 GB), `torchrun` DDP, one process per GPU.
- **Output directory**: `<output_dir>/<experiment_name>_level<L>_seed<S>/` — isolated per run.

## Evaluation Metrics

All metrics are computed in the **normalized domain** on the denoised signal estimate (`input - pred_noise`) vs. the reference signal (`input - label_noise`):

| Metric | Reduction | Notes |
|--------|-----------|-------|
| **SNR** | per-sample mean | Signal-to-Noise Ratio; `+∞` when noise-free |
| **PSNR** | per-sample mean | Peak `data_range = 1.0` (max-abs normalized, peak amplitude = 1) |
| **SSIM** | N/A (uniform weight) | `data_range = 2.0` (signal spans `[-1, 1]`), `window_size = 11`, `σ = 1.5` |
| **MAE** | mean | Mean Absolute Error |
| **MSE** | mean | Mean Squared Error |
| **RMSE** | per-sample mean | Root Mean Squared Error |

For SNR and PSNR, `per_sample` reduction computes the metric independently for each patch (or shot) then averages them, matching the per-shot evaluation convention in seismic processing.

## Results

| Model | Noise 1.0 | Noise 3.0 | Noise 5.0 | Noise 7.0 | Noise 9.0 |
|-------|-----------|-----------|-----------|-----------|-----------|
| UNet | — | — | — | — | — |
| ResUNet | — | — | — | — | — |
| DnCNN | — | — | — | — | — |
| Attention UNet | — | — | — | — | — |

*Results pending — training in progress.*

## Quick Start

All four models use the same nested-loop launcher pattern: edit the config block at the top of each `.sh` to select noise levels, seeds, GPUs, and master port, then run a single command.

```bash
# UNet — full grid sweep (5 noise levels × 3 seeds)
bash scripts/coherent_noise_attenuation/train_denoise_unet.sh

# ResUNet — full grid sweep
bash scripts/coherent_noise_attenuation/train_denoise_res_unet.sh

# DnCNN — full grid sweep
bash scripts/coherent_noise_attenuation/train_denoise_dncnn.sh

# Attention UNet — full grid sweep
bash scripts/coherent_noise_attenuation/train_denoise_atten_unet.sh
```

Or launch a single model / noise level / seed directly via `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
    scripts/coherent_noise_attenuation/train_denoise_unet.py \
    --config configs/coherent_noise_attenuation/denoise_unet.yaml
```

Edit the configuration block at the top of each `.sh` script to select noise levels (`NOISE_LEVELS`), seed count (`N_SEEDS`), starting seed (`START_SEED`), GPUs (`CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`), and base port (`MASTER_PORT`).
