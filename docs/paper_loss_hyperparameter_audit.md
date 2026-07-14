# Paper-Aligned Loss Functions & Training Hyperparameters Audit

Audit of paper-specific training objectives and hyperparameters for five
2D seismic interpolation reproduction models. Each model now has a dedicated
paper-aligned config that overrides the previous shared defaults.

## Summary Alignment Table

| Parameter       | Chai2020          | Wang2019          | Yoon2021          | Yuan2022 BTN       | Guo2023 MST        |
|-----------------|-------------------|-------------------|-------------------|--------------------|--------------------|
| Loss            | MSE               | MSE               | MSE               | NormalizedObservedL1 | MSE              |
| Optimizer       | Adam              | Adam              | Adam              | Adam               | AdamW              |
| LR              | 1e-4              | 1e-3              | 1e-3              | 1e-4               | 1e-4               |
| Scheduler       | cosine→1e-6       | step(30, 0.5)     | step(20, 0.5)     | cosine→1e-6        | cosine→1e-6        |
| Batch Size      | 128               | 192               | 128               | 64 / 32            | 32                 |
| Epochs          | 50                | 100               | 80                | 100                | 80                 |
| Patch Size      | 112×112           | 256×128           | 256×128           | 256×128            | 96×96              |
| Mask Mode       | continuous        | continuous        | uniform           | random / continuous| random             |
| Mask Ratio      | 0.5               | 0.5               | 0.5               | 0.5                | 0.5                |
| Self-Supervised | No                | No                | No                | **Yes**            | No                 |
| Grad Clip       | None              | None              | 1.0               | None               | None               |
| Weight Decay    | 0 (Adam)          | 0 (Adam)          | 0 (Adam)          | 0 (Adam)           | 1e-4               |

## Classification Legend

- **paper-explicit**: Explicitly stated in the paper (value, formula, or table)
- **paper-inferred**: Derived from paper context but not explicitly stated
- **reproduction-assumption**: Best-guess default; paper is silent or inaccessible
- **repository-adaptation**: Added for benchmark framework integration
- **unresolved**: Known gap; requires paper access or author clarification

## Model-by-Model Details

### Chai2020 (chai2020_unet)

- **Config**: `configs/interpolation/chai2020_unet_paper.yaml`
- **Architecture**: Paper-explicit (50-layer U-Net, 5×5 kernels, [64,128,256,512,1024] channels)
- **Training**: Predominantly paper-inferred. The paper's primary contribution is
  the architecture study; training recipes are sparse.
- **Key changes from baseline**: Adam (not AdamW), 1e-4 LR, 112×112 patches, 50 epochs

### Wang2019 (wang2019_resnet)

- **Config**: `configs/interpolation/wang2019_resnet_paper.yaml`
- **Architecture**: Paper-explicit (8-layer residual CNN). All other details are
  reproduction-assumptions.
- **Training**: Almost entirely reproduction-assumptions. The 2019-era paper
  predates widespread use of AdamW, cosine schedules, etc.

### Yoon2021 (yoon2021_dbilstm)

- **Config**: `configs/interpolation/yoon2021_dbilstm_paper.yaml`
- **Architecture**: Paper-explicit (3 BiLSTM, hidden [64,128,128], dropout 0.2)
- **Training**: Reproduction-assumptions. The paper used Keras; optimizer and
  scheduler choices are not documented.

### Yuan2022 BTN (yuan2022_btn)

- **Configs**:
  - `yuan2022_btn_random.yaml` — Random mask + NormalizedObservedL1
  - `yuan2022_btn_spectrum.yaml` — Continuous mask + NormalizedObservedL1 + FK Spectrum Suppression
  - `yuan2022_btn_mix.yaml` — Random mask + NormalizedObservedL1 (mix-training simplified)
- **Architecture**: Paper-explicit (blind-trace constraint, two half-plane branches, 1×1 fusion)
- **Training**: Self-supervised (paper-explicit). Loss is computed ONLY on observed
  positions. This required new infrastructure:
  - `NormalizedObservedL1` loss (mask-aware)
  - `FKSpectrumSuppressionLoss` (BTN-SS variant)
  - `WeightedCompositeLoss` (multi-term combiner)
  - `return_mask` option in data pipeline
  - Extended `train_one_epoch`/`evaluate` for mask extras
- **Unresolved**: Full mix-training data sampler (cycles through multiple decimation
  patterns) is NOT yet implemented. Current BTN-Mix config uses a single random pattern.

### Guo2023 MST (guo2023_mst)

- **Config**: `configs/interpolation/guo2023_mst_paper.yaml`
- **Architecture**: Multi-scale self-attention is paper-explicit. All architecture
  details are reproduction-assumptions (paper behind paywall).
- **Training**: All parameters are reproduction-assumptions.
- **Unresolved**: Horizontal flip and spatial downsampling data augmentation are
  NOT yet implemented.

## Code Changes Summary

### New Files
| File | Purpose |
|------|---------|
| `utils/spectrum_loss.py` | F-K domain spectrum suppression loss + registration |
| `configs/interpolation/chai2020_unet_paper.yaml` | Paper-aligned Chai2020 config |
| `configs/interpolation/wang2019_resnet_paper.yaml` | Paper-aligned Wang2019 config |
| `configs/interpolation/yoon2021_dbilstm_paper.yaml` | Paper-aligned Yoon2021 config |
| `configs/interpolation/yuan2022_btn_random.yaml` | BTN self-supervised (random mask) |
| `configs/interpolation/yuan2022_btn_spectrum.yaml` | BTN-SS with spectrum suppression |
| `configs/interpolation/yuan2022_btn_mix.yaml` | BTN-Mix (simplified) |
| `configs/interpolation/guo2023_mst_paper.yaml` | Paper-aligned Guo2023 config |
| `tests/test_paper_aligned_losses.py` | New loss function tests |
| `tests/test_paper_training_configs.py` | Config validation + regression tests |
| `docs/paper_loss_hyperparameter_audit.md` | This document |

### Modified Files
| File | Change |
|------|--------|
| `utils/losses.py` | Added NormalizedObservedL1, MaskedL1, MaskedMSE, WeightedCompositeLoss |
| `utils/train_utils.py` | Extended train_one_epoch/evaluate for mask extras; generalized build_loaders/build_shot_split_loaders for variable-length returns |
| `utils/__init__.py` | Added spectrum_loss import for registration |
| `scripts/interpolation/train_interpolation_unet.py` | Added return_mask support in _patchify_pairs |
| `scripts/interpolation/train_interpolation_patch_transformer.py` | Added return_mask support in _patchify_pairs |

### Backward Compatibility
- All existing configs continue to work unchanged
- `train_one_epoch` accepts both 2-tuples `(x, y)` and 3-tuples `(x, y, mask)`
- `_patchify_pairs` returns 2 tensors by default; 3 when `return_mask: true`
- No model architectures were modified
- No existing supervised defaults were changed
