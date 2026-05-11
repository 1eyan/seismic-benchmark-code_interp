# Conditional DDPM (DDPM-2c) for Ground-Roll Attenuation

Reproduction of the two-component conditional denoising diffusion probabilistic model for simultaneous signal and ground-roll estimation.

**Reference**: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020; Nichol & Dhariwal, "Improved DDPM", ICML 2021.

## Code Entry Point

```bash
bash scripts/coherent_noise_attenuation/train_denoise_ddpm.sh
```

Key source files:
- `model/coherent_noise_attenuation/ddpm.py` — DDPM U-Net + noise scheduler
- `scripts/coherent_noise_attenuation/train_denoise_ddpm.py` — DDP training loop
- `configs/coherent_noise_attenuation/denoise_ddpm.yaml` — Hyperparameters

## Architecture

### DDPM-2c U-Net

- **Input** (3 channels): concatenation of condition `y` (noisy data), noised signal `x_t`, and noised ground-roll `z_t`
- **Output** (2 channels): predicted noise for signal `ε_x` and ground-roll `ε_z`
- **Encoder**: Conv2d(3×3) → 5 ResNet blocks with downsampling
- **Bottleneck**: Self-attention block for global feature learning
- **Decoder**: 5 ResNet blocks with upsampling + skip connections from encoder
- **Time embedding**: Sinusoidal positional encoding injected into every ResNet block via `SiLU → Linear`

### ResNet Block

`GroupNorm → SiLU → Conv3×3 → GroupNorm → SiLU → Conv3×3 + time_proj + residual`

### Noise Scheduler

Linear β schedule: `β_start=1e-4, β_end=0.02, T=1000`

## Training: Forward Diffusion

At each training step, sample `t ~ Uniform(1, T)` and noise `ε_x, ε_z ~ N(0, I)`:

```
x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε_sig
z_t = √ᾱ_t * z_0 + √(1-ᾱ_t) * ε_gr
```

The model takes `[y, x_t, z_t]` and predicts `[ε_sig, ε_gr]`.

## Loss Function

L1 loss on both noise components:
```
Loss = E[ |ε_sig - ε_θ,sig(x_t, z_t, y, t)| + |ε_gr - ε_θ,gr(x_t, z_t, y, t)| ]
```

## Sampling: Reverse Process

Start from `x_T, z_T ~ N(0, I)`. At each step t → t-1:

```
x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ,sig) + σ_t * ε    (ε=0 when t=1)
z_{t-1} = 1/√α_t * (z_t - (1-α_t)/√(1-ᾱ_t) * ε_θ,gr) + σ_t * ε
```

Use standard DDPM sampling (Algorithm 2 in Ho et al.); do NOT use DDIM acceleration.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=1e-4) |
| Batch size | 196 (effective, across 4 GPUs) |
| Timesteps (T) | 1000 |
| β schedule | linear, [1e-4, 0.02] |
| Normalization | max_abs, global |
| Patching | 128×256, 50% overlap |

## Implementation Notes

- `DDPMNoiseScheduler` precomputes all α/β tensors and provides `.to(device)` for device transfer.
- Evaluation uses `num_steps=50` for fast approximate sampling during training monitoring.
- The scheduler's `.state_dict()` / `.load_state_dict()` serialize precomputed tensors for checkpointing.
- All evaluation must run on all DDP ranks (not just rank 0) to avoid NCCL deadlock.
- Architecture details: `base_channels=64, channel_mults=(1,2,4,8,8), time_emb_dim=256, num_res_blocks=1`.
