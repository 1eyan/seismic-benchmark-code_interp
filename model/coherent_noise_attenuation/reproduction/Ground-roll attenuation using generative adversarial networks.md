# Pix2Pix cGAN for Ground-Roll Attenuation

Reproduction of the conditional GAN (Pix2Pix) approach for seismic ground-roll suppression.

**Reference**: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks", CVPR 2017.

## Code Entry Point

```bash
bash scripts/coherent_noise_attenuation/train_denoise_pix2pix.sh
```

Key source files:
- `model/coherent_noise_attenuation/pix2pix.py` — Generator + PatchGAN discriminator
- `scripts/coherent_noise_attenuation/train_denoise_pix2pix.py` — DDP training loop
- `configs/coherent_noise_attenuation/denoise_pix2pix.yaml` — Hyperparameters

## Architecture

### Generator (U-Net)

Encoder: `C64 → C128 → C256 → C512 → C512 → C512 → C512`
- `C(k)`: Conv2d(4×4, stride=2) → BatchNorm → LeakyReLU(0.2)
- First layer `C64` has no BatchNorm
- All activations: LeakyReLU(0.2)

Decoder: `CD512 → CD512 → C512 → C512 → C256 → C128 → C64`
- `CD(k)`: ConvTranspose2d(4×4, stride=2) → BatchNorm → Dropout(0.5) → ReLU
- First two decoder blocks have 50% dropout
- Skip connections from encoder layer i to decoder layer (n-i)
- Output: Conv2d(7×7) → Tanh

### Discriminator (PatchGAN)

`C64 → C128 → C256 → C512 → Conv2d → 1-channel output`
- All activations: LeakyReLU(0.2)
- First layer has no BatchNorm
- Output: patch-wise real/fake logits (use `BCEWithLogitsLoss`)

## Loss Function

```
L_total = L_cGAN(G, D) + λ * L_L1(G)    (λ = 100)
```

- **Adversarial**: `BCEWithLogitsLoss` — discriminator distinguishes real vs. fake pairs
- **L1**: `F.l1_loss(fake_y, y)` — encourages low-frequency correctness
- Discriminator is fed concatenation of condition (noisy) and target (noise label)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.0002, betas=[0.5, 0.999]) |
| GAN weight | 1 |
| L1 weight (λ) | 100 |
| Normalization | max_abs, global, range [-1, 1] |
| Patching | 128×256, 50% overlap |
| Gradient clipping | 1.0 (max norm) |

## Implementation Notes

- The discriminator is called twice per iteration (real + fake). To avoid DDP inplace conflicts, all ReLU/LeakyReLU activations use `inplace=False`.
- The generator predicts the noise component (not clean signal). Denoised output = input - predicted_noise.
- The discriminator uses `BCEWithLogitsLoss` (no Sigmoid in the model).
