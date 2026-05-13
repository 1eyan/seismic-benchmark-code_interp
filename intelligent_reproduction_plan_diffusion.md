# Intelligent Reproduction Plan - Diffusion

> Scope: diffusion, score-based, classifier-guided, diffusion-driven DIP, and future flow-matching baselines for seismic interpolation / reconstruction.

## Source Basis

- Primary survey table: `../插值 .csv`.
- Structured metadata: `../outputs/interpolation_json_format/插值_v2_fillbycodexjson.json`.
- Parent plan: `memory/intelligent_reproduction_plan.md`.
- Selection rule: include methods that clarify the value of generative priors, data consistency, and sampling cost before introducing the planned 5D Flow-Matching Transformer.

## Diffusion-Side Target Set

| Priority | Paper | Year | Survey Row | Method Family | Reproduction Status | Role In Benchmark |
|---:|---|---:|---:|---|---|---|
| D1 | Seismic Data Reconstruction Based on Conditional Constraint Diffusion Model | 2024 | 63 | DDPM / conditional constraint | First-wave target | Main diffusion baseline for flow-matching comparison. |
| D2 | CDDIP: Constrained Diffusion-Driven Deep Image Prior for Seismic Data Reconstruction | 2025 | 67 | diffusion + DIP + data consistency | First-wave target after D1 | Strong recent hybrid; important for data-consistency and sampling-cost comparison. |
| D3 | Reconstructing Regularly Missing Seismic Traces With a Classifier-Guided Diffusion Model | 2024 | 19 | classifier-guided DDPM | Second-wave target | Regular-missing specialized diffusion baseline. |
| D4 | Seismic trace interpolation via score-based diffusion model with wavelet convolution | 2025 | 20 | score-based diffusion + wavelet convolution | Optional target | Score-model variant; metadata is incomplete in JSON, so verify before implementation. |

## Recommended First-Wave Diffusion Reproduction

Start with D1, then D2. D3 and D4 are deferred until D1/D2 establish a stable sampling/evaluation contract.

- D1, row 63:
  - Paper setup in survey: SEG C3 synthetic dataset and Mobil AVO Viking Graben Line 12 field data.
  - Preprocessing in survey: SEG C3 selects 1800 patches split 1260/360/180; field selects 1000 patches split 700/200/100; synthetic small-gap missing around 28%; field small-gap missing around 20%; diffusion `T=1000`, sampling `T/U=250/10`.
  - Minimum viable reproduction:
    - Input/output: `(B, 1, 128, 128)` masked patch, mask, noisy state, reconstructed clean patch.
    - Conditioning: concatenate masked patch and mask to the denoiser input.
    - Noise schedule: linear or cosine schedule in YAML.
    - Loss: diffusion noise-prediction MSE on all pixels first; add missing-region weighting only after the base model works.
  - Paper-aligned upgrade:
    - Match patch counts and train/val/test split.
    - Implement conditional constraint during sampling.
    - Match the original sampling skip schedule if specified by the paper.

- D2, row 67:
  - Paper setup in survey: poststack 128x128 patches from TGS salt, SEAM I, F3, 1994 BP, AGL Elastic Marmousi, Kerry 3D; prestack 128x128 patches from Alaska, AvoMobil, BP94, SEG-C3, synthetic cross-spread, SEAM I/II, Stratton 3D.
  - Preprocessing in survey: 90/10 split; cosine noise schedule; `T=1000`; diffusion training 5000 epochs; inference uses 25 diffusion timesteps plus decreasing DIP steps.
  - Minimum viable reproduction:
    - Use the D1 pretrained diffusion prior or a shared small diffusion prior.
    - Implement DIP solver as a separate module, not inside the sampler.
    - Enforce data consistency with the observed mask at every outer iteration.
  - Paper-aligned upgrade:
    - Separate poststack and prestack settings.
    - Add 25-step inference schedule and decreasing DIP-step policy.
    - Report sampling time with and without DIP refinement.

## Phase Plan

### Phase D0 - Shared Diffusion Infrastructure

- Add common modules only after D1 is started:
  - `model/diffusion/schedules.py`
  - `model/diffusion/denoisers.py`
  - `model/diffusion/samplers.py`
  - `model/diffusion/data_consistency.py`
- Keep schedule, objective, timestep count, conditioning mode, and sampler steps in YAML.
- Add unit/shape checks for:
  - forward noising
  - timestep embedding
  - denoiser input channel count
  - mask broadcasting
  - observed-trace data consistency

### Phase D1 - Conditional Constraint Diffusion

- Files:
  - `model/diffusion/conditional_unet.py`
  - `model/diffusion/ddpm.py`
  - `configs/repro/row63_conditional_constraint_diffusion.yaml`
  - `configs/repro/row63_conditional_constraint_debug_diffusion.yaml`
- Architecture contract:
  - Use a U-Net denoiser first; transformer denoisers are deferred to the future 5D Flow-Matching Transformer.
  - Denoiser input channels must be explicit, e.g. noisy image, masked image, mask.
  - The sampler must accept an externally supplied mask and observed data.
- Evaluation:
  - Report reconstruction metrics and sampling cost.
  - Evaluate random, regular, and consecutive masks even if the paper emphasizes small-gap missing.

### Phase D2 - CDDIP

- Files:
  - `model/diffusion/cddip.py`
  - `model/dip/attention_unet_dip.py`
  - `configs/repro/row67_cddip_diffusion.yaml`
  - `configs/repro/row67_cddip_debug_diffusion.yaml`
- Architecture contract:
  - Keep diffusion prior, DIP network, and data-consistency update as separate callable modules.
  - The DIP solver may optimize per sample at inference, but the number of optimization steps must be recorded.
  - Observed samples must remain fixed after each data-consistency projection.
- Evaluation:
  - Report inference time per sample because DIP refinement can dominate runtime.
  - Include ablation: diffusion only, DIP only, diffusion + DIP.

### Phase D3 - Classifier-Guided Diffusion

- Files:
  - `model/diffusion/classifier_guided.py`
  - `configs/repro/row19_classifier_guided_diffusion.yaml`
- Implementation:
  - Add missing-pattern class conditioning only after D1 works.
  - Keep class labels in the dataset manifest.
  - Compare with D1 under regular missing masks.

### Phase D4 - Score-Based Wavelet Diffusion

- Files:
  - `model/diffusion/score_wavelet.py`
  - `configs/repro/row20_score_wavelet_diffusion.yaml`
- Implementation:
  - Metadata for row 20 is incomplete in JSON; verify authors/DOI/source before promotion to official target.
  - Implement wavelet convolution as an optional denoiser block controlled by YAML.

## Diffusion Config Suffix Rule

All diffusion-side configs must end with `_diffusion.yaml`, including score-based and DIP-hybrid methods:

- `row63_conditional_constraint_diffusion.yaml`
- `row67_cddip_diffusion.yaml`
- `row19_classifier_guided_diffusion.yaml`
- `row20_score_wavelet_diffusion.yaml`

Debug configs add `_debug` before the suffix:

- `row63_conditional_constraint_debug_diffusion.yaml`

## Diffusion Coding Rules

- Never report diffusion quality without inference time and sampling step count.
- Noise schedule, objective type, `T`, sampler steps, and conditioning channels must be in YAML.
- The mask convention must be explicit at every sampler/data-consistency boundary.
- Data consistency must be implemented as a named function or module, not inline inside a long sampling loop.
- The method manifest must state whether the run is DDPM, score-based, classifier-guided, DIP-hybrid, or flow-matching.

## First Diffusion Milestone

- Implement D1 row 63 as `minimum_viable`.
- Train a small debug diffusion model on existing 128x128 2D patches.
- Produce one comparison table against U-Net and Multi-Scale Transformer:
  - random 50%
  - regular 50%
  - consecutive 20 traces
  - SNR, PSNR, SSIM, MSE, inference steps, inference time

