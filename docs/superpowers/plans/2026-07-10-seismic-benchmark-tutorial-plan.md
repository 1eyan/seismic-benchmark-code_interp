# Seismic-Benchmark-Code Tutorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write a comprehensive, beginner-friendly yet reference-rich Markdown tutorial at `docs/tutorial.md` that covers preprocessing, training, evaluation/inference, and extending the seismic-benchmark-code repository with new models.

**Architecture:** A single document organized as: Introduction → Project Structure → Quick Start → Complete random_noise_suppression Example → Other Tasks → Custom Extensions → Troubleshooting → Quick Reference. Content is derived directly from existing source files, configs, and READMEs.

**Tech Stack:** Markdown, repository-specific file paths and commands.

## Global Constraints

- All text must be in English per project rules.
- Concrete commands must be copy-pasteable and use actual repository paths.
- No design rationale inside code blocks; concise comments only.
- Do not auto-run scripts; only provide commands in plain text.
- The final document must be self-contained: a beginner can follow it without reading source code.
- Include a beginner-critique review step before final delivery.

---

### Task 1: Gather Accurate Source Material

**Files:**
- Read: `README.md`
- Read: `scripts/README.md`
- Read: `scripts/random_noise_suppression/README_models.md`
- Read: `configs/random_noise_suppression/denoise_unet.yaml`
- Read: `model/registry.py`
- Read: `utils/metrics.py` (first 120 lines)
- Read: `utils/losses.py` (first 120 lines)
- Read: `utils/datasets.py` (first 120 lines)
- Read: `tools/preprocessing.py` (function signatures and docstrings)
- Read: `tools/patching.py` (function signatures and docstrings)
- Read: `tools/segy_read.py` (function signatures and docstrings)
- Read: `scripts/random_noise_suppression/train_denoise_unet.py` (argparse section and main flow)
- Read: `scripts/random_noise_suppression/inference_denoise_unet.py` (argparse section and main flow)

**Interfaces:**
- Consumes: approved spec at `docs/superpowers/specs/2026-07-10-seismic-benchmark-tutorial-design.md`
- Produces: a verified list of accurate commands, config values, file paths, and registry APIs to use in the tutorial

- [ ] **Step 1: Read top-level documentation**

  Read `README.md`, `scripts/README.md`, and `scripts/random_noise_suppression/README_models.md`. Extract all concrete commands, directory structures, and model names.

- [ ] **Step 2: Read config and registry files**

  Read `configs/random_noise_suppression/denoise_unet.yaml` and `model/registry.py`. Record every top-level YAML key and the registry decorator signatures.

- [ ] **Step 3: Read utility modules**

  Read `utils/metrics.py`, `utils/losses.py`, and `utils/datasets.py` only far enough to capture the base classes and decorator names (`BaseMetric`, `BaseLoss`, `BaseArrayDataset`, `@register_metric`, `@register_loss`, `@register_dataset`).

- [ ] **Step 4: Read tools modules**

  Read `tools/preprocessing.py`, `tools/patching.py`, and `tools/segy_read.py`. Capture the exact function names, parameter names, and default values needed for the preprocessing section.

- [ ] **Step 5: Read entry script interfaces**

  Read the argparse sections of `scripts/random_noise_suppression/train_denoise_unet.py` and `scripts/random_noise_suppression/inference_denoise_unet.py`. Capture every CLI flag, default value, and help text relevant to the tutorial.

- [ ] **Step 6: Verify example commands**

  Run these verification commands (read-only checks):

  ```bash
  ls configs/random_noise_suppression/denoise_unet.yaml
  ls scripts/random_noise_suppression/train_denoise_unet.py
  ls scripts/random_noise_suppression/inference_denoise_unet.py
  ```

  Expected: all three files exist.

- [ ] **Step 7: Commit notes**

  Create a scratch file `docs/superpowers/plans/2026-07-10-tutorial-source-notes.md` containing the verified facts. Do not commit yet; this is a working note.

---

### Task 2: Write Chapters 1–2 (Introduction, Structure, Quick Start)

**Files:**
- Create: `docs/tutorial.md`

**Interfaces:**
- Consumes: verified facts from Task 1
- Produces: `docs/tutorial.md` with sections 1–2 completed

- [ ] **Step 1: Write Section 1 — Introduction**

  Include:
  - What the repository is.
  - Target audience statement.
  - Prerequisites.
  - Dependency list (`torch`, `numpy`, `matplotlib`, `pyyaml`, `segyio`, `scipy`) and note about no `requirements.txt`.

- [ ] **Step 2: Write Section 2 — Project Structure and Core Concepts**

  Include:
  - Directory table: `model/`, `utils/`, `tools/`, `scripts/`, `configs/`, `results/`.
  - Registry + factory explanation with a minimal pseudocode example.
  - Mention `scripts/train.py` is component-agnostic.

- [ ] **Step 3: Write Section 3 — Quick Start**

  Include:
  - Data path: `data/SEG_45Shot_shots1-9.sgy`.
  - Training command:
    ```bash
    python scripts/random_noise_suppression/train_denoise_unet.py \
      --config configs/random_noise_suppression/denoise_unet.yaml
    ```
  - Inference command:
    ```bash
    python scripts/random_noise_suppression/inference_denoise_unet.py \
      --config configs/random_noise_suppression/denoise_unet.yaml
    ```
  - Expected output tree after training and inference.

- [ ] **Step 4: Self-review sections 1–2**

  Check: no broken commands, all paths match the repository, no placeholders.

- [ ] **Step 5: Commit progress**

  ```bash
  git add docs/tutorial.md
  git commit -m "docs(tutorial): add introduction, structure, and quick start"
  ```

---

### Task 3: Write Section 3.1–3.2 (Data and Preprocessing)

**Files:**
- Modify: `docs/tutorial.md`

**Interfaces:**
- Consumes: verified facts from Task 1
- Produces: `docs/tutorial.md` with data format and preprocessing subsections

- [ ] **Step 1: Write 3.1 Data Format and Loading**

  Include:
  - SEG-Y reading via `tools/segy_read.py` (`read_regular_shots`).
  - Switching to NPY/MAT in YAML by commenting blocks.
  - Shape convention `(n_shots, n_traces, n_time)`.
  - SEG C3 geometry: 201 traces/shot, `dt=0.008s`.

- [ ] **Step 2: Write 3.2 Preprocessing Pipeline**

  Include:
  - Spherical divergence correction (skipped in this example via `skip` list).
  - Normalization: `max_abs` per shot.
  - Synthetic noise injection: `gaussian` / `poisson`, `snr_db`.
  - Patching: `patchify_uniform` with `patch_size=(128, 256)` and `overlap=0.5`.
  - Shot-level split: `shot_split: {train: 7, val: 1, test: 1}`.

- [ ] **Step 3: Verify preprocessing order**

  Cross-check against `tools/preprocessing.py` and `configs/random_noise_suppression/denoise_unet.yaml`.

- [ ] **Step 4: Commit progress**

  ```bash
  git add docs/tutorial.md
  git commit -m "docs(tutorial): add data format and preprocessing sections"
  ```

---

### Task 4: Write Section 3.3 (YAML Config Walkthrough)

**Files:**
- Modify: `docs/tutorial.md`

**Interfaces:**
- Consumes: `configs/random_noise_suppression/denoise_unet.yaml`
- Produces: `docs/tutorial.md` with complete YAML walkthrough

- [ ] **Step 1: Write 3.3 YAML Config Walkthrough**

  Walk through each top-level block of `configs/random_noise_suppression/denoise_unet.yaml`:
  - `experiment:` name, output_dir, seed, device.
  - `data:` format selection, shot_split vs test_ratio, loader params.
  - `preprocess:` dt, noise_kind, snr_db, normalize_mode, normalize_scope, patch_*, skip.
  - `model:` type mapping to registered model, params.
  - `loss:` type and params.
  - `metrics:` each metric with concrete params, including `data_range` for SSIM/PSNR.
  - `optim:` / `scheduler:` / `train:` / `log:`.
  - `inference:` checkpoint, output_dir, n_viz_shots, device, batch_size.
  - `binned_metrics:` enabled, eb_wse, fb_fre with their sub-options.

- [ ] **Step 2: Add callout boxes for common mistakes**

  For example:
  - `data_range` must match `normalize_mode`.
  - `inference.shot_split` must match `data.shot_split`.

- [ ] **Step 3: Commit progress**

  ```bash
  git add docs/tutorial.md
  git commit -m "docs(tutorial): add YAML config walkthrough"
  ```

---

### Task 5: Write Sections 3.4–3.5 (Training, Inference, Evaluation)

**Files:**
- Modify: `docs/tutorial.md`

**Interfaces:**
- Consumes: argparse facts from Task 1, YAML facts from Task 4
- Produces: `docs/tutorial.md` with training and inference sections

- [ ] **Step 1: Write 3.4 Training in Detail**

  Include:
  - Single-GPU command.
  - Output directory structure.
  - Log files: `train_log.txt`, `loss_history.csv`, `metrics_history.csv`.
  - Curve images: `loss_curve.png`, `metrics_curve.png`.
  - Resume command with `--resume`.
  - Multi-GPU command with `torchrun --nproc_per_node=2`.

- [ ] **Step 2: Write 3.5 Inference and Evaluation**

  Include:
  - Command with explicit overrides for checkpoint, noise-kind, snr-db, output-dir.
  - Step-by-step inference flow.
  - Metric groups: `noisy`, `denoised`, `delta`.
  - EB-WSE / FB-FRE explanation and JSON output location.
  - Output files list.

- [ ] **Step 3: Commit progress**

  ```bash
  git add docs/tutorial.md
  git commit -m "docs(tutorial): add training and inference sections"
  ```

---

### Task 6: Write Section 3.6 (Batch Sweeps)

**Files:**
- Modify: `docs/tutorial.md`
- Read: `scripts/random_noise_suppression/train_denoise_unet.sh`
- Read: `scripts/random_noise_suppression/inference_denoise_unet.sh`
- Read: `scripts/random_noise_suppression/run_all_random_noise_models.sh`

**Interfaces:**
- Consumes: shell script contents
- Produces: `docs/tutorial.md` with batch sweep explanation

- [ ] **Step 1: Explain shell sweep scripts**

  Describe how `train_denoise_unet.sh` and `inference_denoise_unet.sh` rewrite temporary YAML configs to sweep noise_kind, SNR, and seed.

- [ ] **Step 2: Explain multi-model runner**

  Describe `run_all_random_noise_models.sh` and which models it covers.

- [ ] **Step 3: Provide example custom sweep snippet**

  Show a minimal loop example for running inference over multiple SNRs manually.

- [ ] **Step 4: Commit progress**

  ```bash
  git add docs/tutorial.md
  git commit -m "docs(tutorial): add batch sweep section"
  ```

---

### Task 7: Write Section 4 (Other Tasks)

**Files:**
- Modify: `docs/tutorial.md`
- Read: `configs/ground_roll_attenuation/denoise_unet.yaml` (first 80 lines)
- Read: `configs/multiples_attenuation/denoise_unet.yaml` (first 80 lines)
- Read: `configs/interpolation/interpolation_unet.yaml` (first 80 lines)

**Interfaces:**
- Consumes: task-specific config facts
- Produces: `docs/tutorial.md` with other-task comparison table and commands

- [ ] **Step 1: Build comparison table**

  Columns: Task, Input Data, Entry Scripts, Config Directory, Key Differences.

- [ ] **Step 2: Add launch commands for each task**

  Provide training and inference commands for ground_roll_attenuation, multiples_attenuation, and interpolation.

- [ ] **Step 3: Highlight task-specific YAML fields**

  For example, interpolation uses `mask_traces` configuration.

- [ ] **Step 4: Commit progress**

  ```bash
  git add docs/tutorial.md
  git commit -m "docs(tutorial): add other tasks section"
  ```

---

### Task 8: Write Section 5 (Custom Extensions)

**Files:**
- Modify: `docs/tutorial.md`
- Read: `model/random_noise_suppression/unet.py` (as an example registered model)
- Read: `model/random_noise_suppression/__init__.py`

**Interfaces:**
- Consumes: registry and base-class facts from Task 1, example model from Task 8
- Produces: `docs/tutorial.md` with custom extension guides

- [ ] **Step 1: Write 5.1 Adding a New Model**

  Include:
  - File location.
  - Inherit `nn.Module`.
  - Use `@register_model("my_model")`.
  - Import in `model/<task>/__init__.py`.
  - YAML usage.
  - Complete minimal example model class.

- [ ] **Step 2: Write 5.2 Adding a New Loss**

  Include base class, decorator, example, and YAML usage.

- [ ] **Step 3: Write 5.3 Adding a New Metric**

  Include base class, decorator, `reduction` convention, example, and YAML usage.

- [ ] **Step 4: Write 5.4 Adding a New Dataset**

  Include shape convention and `register_dataset` example.

- [ ] **Step 5: Write 5.5 Adding a New Preprocessing Step**

  Include implementation in `tools/preprocessing.py` and pipeline integration note.

- [ ] **Step 6: Commit progress**

  ```bash
  git add docs/tutorial.md
  git commit -m "docs(tutorial): add custom extensions section"
  ```

---

### Task 9: Write Sections 6–7 (Troubleshooting and Quick Reference)

**Files:**
- Modify: `docs/tutorial.md`

**Interfaces:**
- Consumes: all prior sections
- Produces: `docs/tutorial.md` with troubleshooting and reference cards

- [ ] **Step 1: Write Section 6 — Troubleshooting and FAQ**

  Include concrete symptoms and fixes for:
  - Checkpoint not found / path issues.
  - `segyio` missing or SEG-Y path wrong.
  - OOM: reduce batch_size or patch size.
  - SSIM/PSNR `data_range` mismatch.
  - `shot_split` inconsistency.
  - Model not registered.

- [ ] **Step 2: Write Section 7 — Quick Reference Cards**

  Include:
  - CLI command cheat sheet.
  - YAML top-level keys and common fields.
  - Registry decorator / factory / base class cheat sheet.
  - Metric arguments cheat sheet.

- [ ] **Step 3: Add table of contents**

  Insert a Markdown table of contents at the top of `docs/tutorial.md`.

- [ ] **Step 4: Commit progress**

  ```bash
  git add docs/tutorial.md
  git commit -m "docs(tutorial): add troubleshooting and quick reference"
  ```

---

### Task 10: Beginner-Critique Review

**Files:**
- Read: `docs/tutorial.md`
- Create: `docs/superpowers/plans/2026-07-10-tutorial-critic-feedback.md`

**Interfaces:**
- Consumes: complete first draft of `docs/tutorial.md`
- Produces: actionable feedback document and revised `docs/tutorial.md`

- [ ] **Step 1: Dispatch beginner-critic subagent**

  Use the `Agent` tool with `subagent_type: general-purpose` and the following prompt:

  > You are a beginner user who just found the seismic-benchmark-code repository. You know basic PyTorch and Python but have never used this codebase. Read `docs/tutorial.md` from start to finish. Pretend you are following it to train and evaluate a random-noise suppression model. Be harsh but fair: identify every place where the tutorial is unclear, assumes unstated knowledge, uses jargon without explanation, has ambiguous commands, or could block a beginner. Produce a structured list of specific issues with line numbers or section names and concrete suggestions for fixes. Do not fix the document yourself; only produce feedback.

- [ ] **Step 2: Collect feedback into a markdown file**

  Save the subagent output to `docs/superpowers/plans/2026-07-10-tutorial-critic-feedback.md`.

- [ ] **Step 3: Triage feedback**

  Classify each item as:
  - Must fix (blocks a beginner)
  - Should fix (causes confusion)
  - Nice to have

- [ ] **Step 4: Apply must-fix and should-fix items**

  Edit `docs/tutorial.md` to address all must-fix and should-fix items.

- [ ] **Step 5: Commit revised draft**

  ```bash
  git add docs/tutorial.md docs/superpowers/plans/2026-07-10-tutorial-critic-feedback.md
  git commit -m "docs(tutorial): incorporate beginner-critic feedback"
  ```

---

### Task 11: Final Polish and Delivery

**Files:**
- Modify: `docs/tutorial.md`
- Optionally modify: `README.md`

**Interfaces:**
- Consumes: revised `docs/tutorial.md`
- Produces: final tutorial document ready for use

- [ ] **Step 1: Final formatting pass**

  Check:
  - Consistent heading levels.
  - All code blocks have language tags.
  - No broken internal links.
  - English only.

- [ ] **Step 2: Decide on README link**

  If the spec’s optional link to README is approved, add a "Tutorial" link in `README.md` near the top. Otherwise skip.

- [ ] **Step 3: Final commit**

  ```bash
  git add docs/tutorial.md
  # optionally: git add README.md
  git commit -m "docs(tutorial): add comprehensive end-to-end user tutorial"
  ```

- [ ] **Step 4: Report completion**

  Report to the user that the tutorial is complete at `docs/tutorial.md`, summarize what it covers, and note the critic-feedback file location.

---

## Self-Review

1. **Spec coverage**: Every section of the approved spec maps to a task:
   - Sections 1–2 → Task 2
   - Section 3 → Tasks 3–6
   - Section 4 → Task 7
   - Section 5 → Task 8
   - Sections 6–7 → Task 9
   - Critic review requirement → Task 10

2. **Placeholder scan**: No TBD, TODO, or vague instructions remain. All commands use real paths. The optional README link is explicitly called out in Task 11 Step 2.

3. **Type consistency**: This is a documentation task; no code signatures to align beyond exact file paths and decorator names, all captured in Task 1.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-10-seismic-benchmark-tutorial-plan.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

Which approach would you like?
