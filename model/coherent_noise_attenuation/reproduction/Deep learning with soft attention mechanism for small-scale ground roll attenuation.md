# Soft Attention Network (SANet) for Ground-Roll Attenuation

Reproduction of the multi-branch soft-attention residual network for small-scale ground-roll suppression.

## Code Entry Point

```bash
bash scripts/coherent_noise_attenuation/train_denoise_sanet.sh
```

Key source files:
- `model/coherent_noise_attenuation/sanet.py` — SANet model
- `scripts/coherent_noise_attenuation/train_denoise_sanet.py` — DDP training loop
- `configs/coherent_noise_attenuation/denoise_sanet.yaml` — Hyperparameters

## Architecture

### Multi-Branch Attention Block

- Multiple parallel convolutional branches with **different kernel sizes** extract features at multiple spatial scales
- Each branch captures different aspects of seismic waveforms (broad dispersive ground-roll vs. fine reflection structure)

### Soft Attention Mechanism

- After multi-branch feature extraction, a soft attention module dynamically weights and fuses branch outputs
- Attention weights are computed from spatial feature relationships, assigning higher weight to ground-roll-correlated features
- This focuses the network on noise suppression while reducing signal leakage in overlapping low-frequency bands

## Learning Strategy

### Noise-Target Residual Learning

The model predicts **ground-roll noise** (not clean signal):

```
n = y - x           (n: noise label, y: noisy input, x: clean reference)
F_SANet(y) ≈ n      (model learns noisy → noise mapping)
x_hat = y - n_hat   (denoised output)
```

Fitting the noise component is easier than fitting the clean signal because ground-roll has a more consistent structure (low-frequency, dispersive) compared to complex reflection events.

## Loss Function

Uses residual learning with MSE or MAE between predicted noise and true noise label. The model is optimized to minimize the difference between "predicted residual" and "true ground-roll noise."

## Implementation Notes

- **Small-scale patching**: Input must be 2D patches (default 128×256). The multi-branch kernels and attention receptive fields are designed for local feature optimization on small patches.
- **Attention normalization**: Use Softmax to ensure attention weights across branches are normalized and stable in local spatial regions.
- Architecture details follow the standard U-Net encoder-decoder with the multi-branch attention block replacing the standard convolution blocks.
