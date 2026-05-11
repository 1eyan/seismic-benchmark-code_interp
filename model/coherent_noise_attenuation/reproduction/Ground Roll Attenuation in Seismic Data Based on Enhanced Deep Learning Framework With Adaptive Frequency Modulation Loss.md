# Enhanced U-Net with Adaptive Frequency Modulation (AFM) Loss

Reproduction of the enhanced deep learning framework combining residual U-Net, attention gates, and a hybrid time-frequency domain loss for ground-roll attenuation.

## Code Entry Point

```bash
bash scripts/coherent_noise_attenuation/train_denoise_enhanced_unet.sh
```

Key source files:
- `model/coherent_noise_attenuation/enhanced_unet.py` — Model + AFM loss
- `scripts/coherent_noise_attenuation/train_denoise_enhanced_unet.py` — DDP training loop
- `configs/coherent_noise_attenuation/denoise_enhanced_unet.yaml` — Hyperparameters

## Architecture

### Enhanced U-Net Backbone

- **Residual blocks** in both encoder and decoder replace standard double-conv layers (ResUNet-style)
- **Attention gates** on skip connections (Attention U-Net style): gating signal from decoder upsampled features modulates encoder skip features before concatenation
- Encoder: ResBlock → MaxPool at each level (depth=4, base_channels=32)
- Bottleneck: ResBlock (channel doubling)
- Decoder: ConvTranspose2d → AttentionGate → concat(gated_skip, upsampled) → ResBlock

### Input / Output

- **Input**: Noisy shot gather (time-space domain, normalized to [-1, 1])
- **Output**: Clean reflection signal (directly predicted, not noise residual)

## Loss Function

### Hybrid Loss: MSE + AFM

```
Loss = L_MSE + λ * L_AFM    (λ = 0.001)
```

**MSE (time-space domain)**:
```
L_MSE = ||y_true - y_pred||²
```
Ensures pixel-level waveform fidelity in the time-space domain.

**AFM (adaptive frequency modulation, f-x domain)**:
- Computes 2D real FFT (`torch.fft.rfft2`) of both prediction and target
- Adaptive weighting: `w = (1 / |Y_true|) * (1 + |Δ| / |Y_true|)` where Δ = |Y_pred| - |Y_true|
- Weights are self-normalized to O(1) scale
- Emphasizes frequency components where signal energy is weak (prone to being overwhelmed by ground-roll)
- All operations use differentiable PyTorch FFT ops for end-to-end gradient flow

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=1e-4) |
| Scheduler | Cosine annealing (min_lr=1e-6) |
| Epochs | 200 |
| Loss | hybrid_mse_afm (λ=0.001, eps=1e-3) |
| Normalization | max_abs, global, range [-1, 1] |
| Patching | 128×256, 50% overlap |
| Gradient clipping | 1.0 (max norm) |

## Implementation Notes

- `_ResBlock` does NOT apply ReLU after the skip addition — this allows bipolar feature representation needed for clean-signal prediction (range [-1, 1]).
- AFM loss uses `torch.clamp(targ_mag, min=eps)` as a hard energy floor to prevent division-by-zero explosion at high frequencies.
- AFM weights are self-normalized (`weight / weight.mean().detach()`) to keep the AFM term O(1) regardless of absolute FFT magnitude scale.
- λ is set low (0.001) to keep AFM as a light regularizer; the bulk of optimization is driven by MSE.
