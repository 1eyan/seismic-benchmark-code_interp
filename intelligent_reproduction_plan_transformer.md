# Intelligent Reproduction Plan - Transformer

> Scope: transformer, Swin-style attention, self-attention, frequency-domain attention, and attention-heavy CNN baselines for seismic interpolation / reconstruction.

## Source Basis

- Primary survey table: `../插值 .csv`.
- Structured metadata: `../outputs/interpolation_json_format/插值_v2_fillbycodexjson.json`.
- Parent plan: `memory/intelligent_reproduction_plan.md`.
- Selection rule: include methods that test whether attention / transformer structure improves seismic interpolation beyond U-Net, 3D CNN, and 5D CNN baselines.

## Transformer-Side Target Set

| Priority | Paper | Year | Survey Row | Method Family | Reproduction Status | Role In Benchmark |
|---:|---|---:|---:|---|---|---|
| T1 | Seismic Data Interpolation Based on Multi-Scale Transformer | 2023 | 60 | Multi-scale Transformer | First-wave target | Main transformer baseline against 5D Flow-Matching Transformer. |
| T2 | Consecutively Missing Seismic Data Reconstruction Via Wavelet-Based Swin Residual Network | 2023 | 58 | Swin residual + wavelet | Second-wave target | Consecutive-missing transformer-style local-window baseline. |
| T3 | FDSANet: Seismic Data Reconstruction Based on a Frequency-Domain Self-Attention Network | 2025 | 66 | U-Net + frequency-domain self-attention | Second-wave target | Frequency-domain attention baseline; useful for spectral fidelity comparison. |
| T4 | Seismic Data Reconstruction Based on Multiscale Attention Deep Learning | 2023 | 16 | multiscale enhanced attention CNN | Optional target | Attention-CNN bridge baseline if full transformer methods underperform or are too costly. |
| T5 | Reconstruction and Compensation of Missing Trace and Attenuation Seismic Data: Integrating Nonlinear Activation-Free Network With Attention Mechanisms | 2025 | 40 | NAFNet / U-Net3+ / CBAM | Optional target | Attention restoration baseline for missing trace plus attenuation settings. |

## Recommended First-Wave Transformer Reproduction

Start with T1 only, then add T2 and T3 after the shared 2D protocol is stable.

- T1, row 60:
  - Paper setup in survey: BP2007 public dataset for train/validation, synthetic test data, and field seismic data.
  - Preprocessing in survey: spatial x2 and x4 downsampling augmentation; 96x96 patches; 50% random trace dropout; normalization to [0, 1]; AdamW; batch size 32; 80 epochs; learning rate 1e-4 to 1e-6.
  - Minimum viable reproduction:
    - Input/output: `(B, 1, 96, 96)` masked patch to reconstructed patch.
    - Missing modes: random 50%, regular 50%, consecutive 10/20/30 traces.
    - Model: multi-scale patch embedding + local/global transformer blocks + reconstruction head.
    - Loss: MSE first, then add L1/SSIM hybrid only if the paper-aligned setting needs it.
  - Paper-aligned upgrade:
    - Add x2/x4 downsample augmentation.
    - Match AdamW schedule and 80-epoch budget.
    - Add the original multi-scale branch widths if recoverable from the paper.

## Phase Plan

### Phase T0 - Attention Baseline Contract

- Reuse the shared 2D artifact format from `memory/intelligent_reproduction_plan.md`.
- Fix the first transformer patch size to 96x96 so row 60 can be compared cleanly.
- Add a config field for positional encoding type:
  - `absolute_2d`
  - `relative_window`
  - `none`
- Add shape tests for every transformer block because silent spatial reshaping errors are common.

### Phase T1 - Multi-Scale Transformer

- Files:
  - `model/transformer_multiscale.py`
  - `configs/repro/row60_multiscale_transformer.yaml`
  - `configs/repro/row60_multiscale_transformer_debug.yaml`
- Architecture contract:
  - Keep input/output shape identical.
  - Keep tokenization reversible enough that output is aligned pixel-wise with the target.
  - Do not mix trace/time axes without an explicit permutation utility.
- Evaluation:
  - Run the same masks as U-Net row 32.
  - Report parameter count and inference time.
  - Save attention-friendly visual diagnostics: residual map and frequency residual.

### Phase T2 - Swin / Window Attention Consecutive-Missing Baseline

- Files:
  - `model/transformer_swin_wavelet.py`
  - `configs/repro/row58_swin_wavelet_transformer.yaml`
- Implementation:
  - Start with a U-Net-like encoder/decoder and insert shifted-window attention blocks.
  - Add wavelet preprocessing or wavelet loss only after the window-attention path is verified.
  - Consecutive missing traces are mandatory for this target.
- Evaluation:
  - Prioritize consecutive 10/20/30-trace masks.
  - Compare directly against row 2 nested U-Net and row 13 regeneration-constrained self-supervision.

### Phase T3 - Frequency-Domain Self-Attention

- Files:
  - `model/frequency_attention_unet.py`
  - `configs/repro/row66_fdsanet_transformer.yaml`
- Implementation:
  - Implement the spatial U-Net path first.
  - Add FFT-domain attention in the bottleneck as a named module.
  - Keep FFT normalization and inverse transform settings in YAML.
- Evaluation:
  - Add frequency-domain residual plots.
  - Report whether spectral continuity improves under regular and consecutive missing masks.

## Transformer Config Suffix Rule

All transformer-side configs must end with `_transformer.yaml`, even if the architecture is attention-CNN rather than a pure transformer:

- `row60_multiscale_transformer.yaml`
- `row58_swin_wavelet_transformer.yaml`
- `row66_fdsanet_transformer.yaml`
- `row16_msea_attention_transformer.yaml`
- `row40_naf_attention_transformer.yaml`

Debug configs add `_debug` before the suffix:

- `row60_multiscale_debug_transformer.yaml`

## Transformer Coding Rules

- Never flatten `(trace, time)` into one token axis without storing the original grid shape.
- Positional encoding must be configurable and documented in the method manifest.
- Window size, patch size, stride, heads, depth, and embedding dimension must live in YAML.
- Attention maps are optional, but residual and mask visualizations are mandatory.
- The method manifest must state whether the method is a pure transformer, Swin/window transformer, or attention-augmented CNN.

## First Transformer Milestone

- Implement T1 row 60 as `minimum_viable`.
- Run a smoke test on existing SEG-C3-45 2D patches.
- Produce one comparison table against the current U-Net baseline:
  - random 50%
  - regular 50%
  - consecutive 20 traces
  - SNR, PSNR, SSIM, MSE, inference time

