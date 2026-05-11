# Physics-Constrained Deep Learning for Ground-Roll Attenuation

Reproduction of the physics-constrained CNN framework that separates signal and ground-roll using physical prior knowledge (frequency, velocity) and an f-k domain classifier.

## Code Entry Point

```bash
bash scripts/coherent_noise_attenuation/train_denoise_physics.sh
```

Key source files:
- `model/coherent_noise_attenuation/physics_unet.py` — Physics-constrained CNN models
- `scripts/coherent_noise_attenuation/train_denoise_physics.py` — DDP training loop
- `configs/coherent_noise_attenuation/denoise_physics.yaml` — Hyperparameters

## Architecture

The framework consists of three CNNs and one f-k domain classifier.

### Signal/Noise Separation Networks (CNN₁ & CNN₂)

- **CNN₁**: Estimates clean signal `X` from noisy input `Z`
- **CNN₂**: Estimates ground-roll `Y` from noisy input `Z`
- **Architecture**: Encoder-decoder with a single skip connection (first encoder → last decoder)
- **Asymmetric kernel design** (t×x):
  - Initial conv: 7×21 (captures longer time-axis dependencies)
  - Internal conv: 3×9
- **Downsampling**: Conv3×9 + MaxPool2d
- **Upsampling**: Upsample + Conv3×9
- **Output**: CNN₁ has 3 conv layers after decoder; CNN₂ has 2 conv layers

### Ground-Roll Mapping Constraint (CNN₃, SW Constraint)

- **Architecture**: Identical to CNN₂
- **Purpose**: Takes CNN₁ output (estimated signal `X`) and predicts ground-roll `Y_recover`
- Enforces the physical prior that shallow reflections and ground-roll are spatially correlated across shot gathers

### f-k Domain Classifier

- **Architecture**: Conv3×3 → 3× (Conv3×3 + MaxPool) → Dropout → 3× FC layers
- **Input**: Real and imaginary parts of 2D FFT (f-k spectrum)
- **Output**: Probability — 1 = signal, 0 = ground-roll
- **Must be pre-trained** before training the main separation networks

## Loss Function

### Final Joint Loss (Equation 8)

```
L_total = L_signal + L_ground_roll
        + λ₁·L_signal_class + λ₂·L_noise_class
        + λ₃·L_recover_data + λ₄·L_recover_noise_class
```

Component definitions:
- `L_signal = MSE(X_true, X_pred)` — signal fidelity
- `L_ground_roll = MSE(Y_true, Y_pred)` — noise fidelity
- `L_signal_class` — cross-entropy: f-k classifier on X (target: 1)
- `L_noise_class` — cross-entropy: f-k classifier on Y (target: 0)
- `L_recover_data = MSE(Y_recover, Y_pred)` — SW constraint consistency
- `L_recover_noise_class` — cross-entropy: f-k classifier on Y_recover

### Frequency Constraints

- `L_low`: Minimize low-frequency energy in estimated signal X (suppress residual ground-roll)
- `L_high`: Minimize high-frequency energy in estimated ground-roll Y (preserve reflection signals)

## Training Procedure

1. **Pre-train f-k classifier** on labeled f-k domain data
2. **Train CNN₁ + CNN₂ + CNN₃** with the joint loss
3. The f-k classifier computes losses by: t-x output → differentiable 2D FFT → extract real/imag → feed to classifier
4. All FFT operations must be differentiable (`torch.fft.rfft2`) for end-to-end gradient flow

## Implementation Notes

- Asymmetric rectangular kernels (e.g., 7×21, 3×9) are intentional — patches are typically longer in the time dimension, requiring larger temporal receptive fields.
- CNN₁ and CNN₂ share no weights but have coordinated architectures.
- The f-k classifier operates on the complex FFT (real + imaginary channels), not magnitude-only.
