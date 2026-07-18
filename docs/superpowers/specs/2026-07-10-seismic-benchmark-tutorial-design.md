# Tutorial Document Design: seismic-benchmark-code End-to-End User Guide

## Metadata

- **Date**: 2026-07-10
- **Topic**: Comprehensive user tutorial for the seismic-benchmark-code repository
- **Target audience**: Beginners and experienced deep-learning practitioners
- **Approach**: Hybrid structure (general concepts → worked example → extensions → quick reference)
- **Status**: Awaiting implementation

## Goal

Write a single, self-contained Markdown tutorial document that teaches users how to use the `seismic-benchmark-code` repository end-to-end:

1. Environment setup and project orientation
2. Data preparation and preprocessing
3. Training a model from a YAML config
4. Running inference / evaluation
5. Adding custom models, losses, metrics, and datasets
6. Troubleshooting common problems

The document must be detailed enough for a beginner to follow, while also providing quick-reference sections for experienced users.

## Document Structure (Approved)

### 1. Introduction

- 1.1 What this repository is: a PyTorch benchmark template for seismic data processing using a registry + factory pattern.
- 1.2 Who this tutorial is for: both newcomers and experienced practitioners.
- 1.3 Prerequisites: basic PyTorch, familiarity with YAML, optional seismology background.
- 1.4 Dependencies: torch, numpy, matplotlib, pyyaml, segyio, scipy. Note that there is no centralized requirements.txt yet.

### 2. Project Structure and Core Concepts

- 2.1 Directory overview:
  - `model/` — neural network definitions and registry
  - `utils/` — datasets, losses, metrics, training loops, logging
  - `tools/` — data I/O and preprocessing helpers
  - `scripts/` — CLI entry points
  - `configs/` — YAML experiment configs
  - `results/` — outputs (gitignored)
- 2.2 Registry + factory pattern:
  - `MODEL_REGISTRY`, `DATASET_REGISTRY`, `LOSS_REGISTRY`, `METRIC_REGISTRY`
  - How `{ type: ..., params: {...} }` in YAML maps to Python decorators and factories
  - Minimal pseudocode example of a registered component
- 2.3 Component-agnostic training script (`scripts/train.py`) and task-specific scripts.

### 3. Quick Start

- 3.1 Data placement: SEG C3 `SEG_45Shot_shots1-9.sgy` location and alternatives (NPY/MAT).
- 3.2 Train a random-noise suppression model in one command.
- 3.3 Run inference in one command.
- 3.4 Expected output tree after training and inference.

### 4. Complete End-to-End Example: Random Noise Suppression

- 4.1 Data format and loading:
  - SEG-Y via `tools/segy_read.py`
  - NPY/MAT switching in YAML
  - Shape convention `(n_shots, n_traces, n_time)`
- 4.2 Preprocessing pipeline:
  - Spherical divergence correction (skipped in this example)
  - Normalization (`max_abs` per shot)
  - Synthetic noise injection (`gaussian` / `poisson`)
  - Patching (`patchify_uniform` with overlap)
  - Shot-level split (`shot_split: {train: 7, val: 1, test: 1}`)
- 4.3 YAML config walkthrough:
  - `experiment`
  - `data`
  - `preprocess`
  - `model`
  - `loss`
  - `metrics`
  - `optim` / `scheduler` / `train` / `log`
  - `inference`
  - `binned_metrics` (EB-WSE / FB-FRE)
- 4.4 Training in detail:
  - Single-GPU command
  - Output directory structure
  - Logs and curves
  - Resume from checkpoint
  - Multi-GPU with `torchrun`
- 4.5 Inference and evaluation:
  - Command with overrides
  - Step-by-step inference flow
  - Metric groups (`noisy`, `denoised`, `delta`)
  - Binned metrics explanation
  - Output files
- 4.6 Batch sweeps:
  - Shell scripts `train_denoise_unet.sh` / `inference_denoise_unet.sh`
  - `run_all_random_noise_models.sh`

### 5. Extending to Other Tasks

- Comparison table of the four supported tasks:
  - random_noise_suppression
  - ground_roll_attenuation
  - multiples_attenuation
  - interpolation
- For each task: input characteristics, entry scripts, config directory, key YAML differences.
- Example launch commands for each.

### 6. Customizing and Extending the Library

- 6.1 Adding a new model:
  - File location under `model/<task>/`
  - Inherit `nn.Module`
  - Use `@register_model("my_model")`
  - Import in `model/<task>/__init__.py`
  - YAML usage
  - Complete minimal code example
- 6.2 Adding a new loss:
  - Inherit `BaseLoss` in `utils/losses.py`
  - Use `@register_loss("my_loss")`
  - Example + YAML usage
- 6.3 Adding a new metric:
  - Inherit `BaseMetric` in `utils/metrics.py`
  - Use `@register_metric("my_metric")`
  - Explain `reduction` convention
- 6.4 Adding a new dataset:
  - Inherit `BaseArrayDataset`
  - Shape convention
- 6.5 Adding a new preprocessing step:
  - Implement in `tools/preprocessing.py`
  - Integrate into pipeline if needed

### 7. Troubleshooting and FAQ

- Checkpoint not found / path issues
- segyio not installed or SEG-Y path problems
- Out-of-memory: reduce batch size or patch size
- SSIM / PSNR `data_range` mismatch with `normalize_mode`
- `shot_split` inconsistency between training and inference
- Model not registered: missing import or decorator typo

### 8. Quick Reference Cards

- 8.1 CLI command cheat sheet
- 8.2 YAML top-level keys and common fields
- 8.3 Registry decorator / factory / base class cheat sheet
- 8.4 Metric arguments cheat sheet

## Quality Gates

1. **Spec self-review**: after writing the spec, scan for placeholders, contradictions, and ambiguity.
2. **User review**: user must approve the spec before implementation.
3. **Implementation plan**: invoke `writing-plans` skill to produce a concrete writing plan.
4. **Beginner critique**: after the first draft is written, spawn a critic subagent that role-plays a beginner, reviews the tutorial for clarity and gaps, and produces actionable feedback.
5. **Revision**: incorporate critic feedback and produce the final document.

## Success Criteria

- A user with basic PyTorch knowledge can follow the tutorial to train and evaluate a random-noise suppression model without reading source code.
- An experienced user can find the CLI / YAML / registry reference in under 60 seconds.
- The document contains no broken commands, no outdated paths, and no ambiguous instructions.
- The beginner critique identifies no critical blockers.

## Files to Create

- `docs/tutorial.md` (final tutorial document)
- `docs/superpowers/specs/2026-07-10-seismic-benchmark-tutorial-design.md` (this spec)
- Optional: update `README.md` to link to the new tutorial (decided during implementation)

## Notes

- Keep all text in English per project rules.
- Do not include design rationale inside code blocks; concise docstrings / comments only.
- Concrete commands must be copy-pasteable and use the actual paths from the repository.
