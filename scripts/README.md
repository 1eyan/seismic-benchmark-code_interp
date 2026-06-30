# scripts/

## Purpose

Entry-point scripts for training, evaluation, and data preparation. Scripts only parse CLI, load configs, and wire up components from `model/` and `utils/`; they do not implement algorithms.

## Contents

Task-specific training and inference scripts; each task lives in its own sub-directory:

- `interpolation/` — Interpolation training (`train_interpolation_unet.py`, `train_paired_unet.py`) and inference (`inference_interpolation.py`).
- `random_noise_suppression/` — Training (`train_denoise_*.py`) and inference (`inference_denoise_*.py`) for UNet, DnCNN, ResUNet, Attention UNet, and SCRN. Inference scripts output scalar metrics plus EB-WSE/FB-FRE diagnostics.
- `ground_roll_attenuation/` — Training scripts for various denoising architectures.
- `multiples_attenuation/` — Training scripts for multiple attenuation.
- `first_break_picking/` — Training scripts for first-break picking.

## Constraints

- Scripts must **not** define model structures or loss functions; everything is imported from `model/` and `utils/`.
- Each script must accept `--config <yaml>` and dump the effective final config into the experiment output directory.
- The agent must not run these scripts; the user executes them manually.

## How to run a training script

Each training script follows the same pattern: load a YAML config, build the model/loss/dataset from the registries, and run the training loop.

```bash
# Example: random-noise suppression UNet
python scripts/random_noise_suppression/train_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml

# Resume from a checkpoint
python scripts/random_noise_suppression/train_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml \
  --resume results/<exp>/checkpoints/epoch_0010.pt
```

Expected outputs under `results/<experiment.name>/`:

- `checkpoints/epoch_*.pt`, and optionally `best.pt`
- `logs/train_log.txt`, `logs/loss_history.csv`, `logs/metrics_history.csv`
- `logs/loss_curve.png`, `logs/metrics_curve.png`
- `visualizations/epoch_*.png`
- `config_used.yaml`

## How to run an inference script

Inference scripts load a trained checkpoint and a held-out volume, run full-shot reconstruction, and save metrics and visualizations.

```bash
# Random-noise suppression inference (UNet)
python scripts/random_noise_suppression/inference_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml \
  --checkpoint results/random_noise/random_noise_unet_base/checkpoints/best.pt \
  --output-dir results/random_noise/random_noise_unet_base/inference \
  --noise-kind gaussian \
  --snr-db 5 \
  --n-viz-shots 5
```

Inference outputs under `--output-dir`:

- `metrics_summary.json` — mean scalar metrics plus EB-WSE and FB-FRE diagnostics.
- `metrics_per_shot.csv` — per-shot scalar metrics (no binned columns).
- `visualizations/` — sample PNGs.
- `npy/` — optional saved arrays when `--save-npy` is set.

## How to add a new entry script

1. Create `scripts/<new_entry>.py` that:
   - Parses CLI args (at minimum `--config`).
   - Calls `utils.load_config` and the appropriate `build_*` factories.
   - Contains **no** algorithmic logic of its own.
2. Document the command and expected outputs in this README.
3. Keep the script idempotent: safe to re-run against the same `--config`.
