# Reproduction Coding Laws

> These laws govern all code written for intelligent seismic interpolation / reconstruction reproductions. They are stricter than normal style guidance because benchmark validity depends on fairness, traceability, and repeatability.

## Law 1 - One Protocol, Many Models

- All reproduced methods must consume the same dataset artifact format, split manifests, mask manifests, and metric functions.
- A model may define its own preprocessing only when the paper requires it; the deviation must be visible in the config and `method_manifest.json`.
- Do not create paper-specific hidden data loaders unless the paper's data geometry genuinely requires a new reusable dataset class.

## Law 2 - Registry First

- New models live under `model/` and register through `@register_model`.
- New losses and metrics live under `utils/losses.py` or `utils/metrics.py` and register through the existing registries.
- New data components live under `utils/datasets.py` or a clearly named dataset module if the file becomes too large.
- Entry scripts must wire components together; they must not contain model architecture, loss formulas, metric formulas, or dataset transformation logic.

## Law 3 - Config Owns Behavior

- Every hyper-parameter that affects results must be in YAML.
- No dataset path, patch size, missing ratio, learning rate, loss weight, diffusion step count, or random seed may be hard-coded in model or training code.
- Defaults are allowed only in builders and must preserve backward compatibility with existing configs.

## Law 4 - Paper Fidelity Is Auditable

- Each reproduction must cite its source paper in the config or method manifest using title, year, survey row, and DOI when known.
- Every implemented method must record known deviations from the paper.
- If a paper detail is missing from the survey or source, write `not specified` in the manifest instead of guessing.

## Law 5 - Fairness Beats Local Wins

- A model must not use a stronger data split, easier mask, extra target information, or different normalization statistics than competitors in the same comparison table.
- Any model-specific advantage required by the original paper must be reported as a paper-aligned setting, not mixed into the default fair benchmark.
- Tuning budgets should be comparable across methods; if not, report the budget difference.

## Law 6 - Masks Are First-Class Data

- Masks must be saved and loaded as artifacts, not regenerated silently during evaluation.
- Mask semantics must be explicit: `True = missing` or `True = observed` must be documented at the API boundary.
- Losses that operate only on missing or observed traces must state that convention in the loss name, docstring, and config.

## Law 7 - Reproducibility Manifests Are Mandatory

- Every official run must save `config_resolved.yaml`, `method_manifest.json`, `runtime.json`, loss CSV, metrics CSV, and visual samples.
- `runtime.json` must include device, batch size, parameter count, seed, start/end time, and peak memory when available.
- Runs without a manifest are debug runs and must not be used in paper tables.

## Law 8 - Data Geometry Must Not Be Ambiguous

- Tensor dimension order must be documented at the function boundary and in the config.
- 2D, 3D, and 5D pipelines must not silently flatten spatial/acquisition dimensions without a named transform.
- 5D tensors must use one documented geometry convention for the whole benchmark.

## Law 9 - No Private-Data Dependency In The Core Benchmark

- Private datasets may be used only as optional extensions.
- The default benchmark must run on public, synthetic, or locally documented datasets.
- If a paper used private field data, reproduce the method on the closest public substitute and record the limitation.

## Law 10 - Minimal Viable Reproduction Comes First

- Implement the smallest paper-faithful version that can train and evaluate on the shared protocol before adding advanced variants.
- Do not port every paper detail in the first pass if it blocks a fair baseline.
- Upgrade status from `minimum_viable` to `paper_aligned` only after the missing paper details are implemented and documented.

## Law 11 - Generative Methods Must Report Sampling Cost

- Diffusion, DIP, flow-matching, and other iterative/generative methods must report inference steps and wall-clock inference time.
- Sampling acceleration settings must be explicit in YAML.
- A generative method is not comparable to CNN/Transformer baselines until both reconstruction quality and inference cost are reported.

## Law 12 - Visual QA Is Part Of Correctness

- Every official evaluation must save input, reconstruction, target, residual, and mask visualizations for a fixed set of sample IDs.
- Severe visual artifacts, axis flips, scale inversions, or mask convention mistakes must block promotion from debug to official.
- Frequency-domain visual checks should be included for interpolation methods when practical.

## Law 13 - Respect Existing Memory Rules

- Before adding a new reproduction method, read `memory/code_design.md`, `memory/research_first.md`, and this file.
- Important method additions must update `memory/updates.md`.
- Open-source references or source-code ports must update `memory/research_first.md`.

## Law 14 - No Silent Long Training

- Agents must not launch long training runs automatically.
- Smoke tests and shape checks are allowed only when they are quick and local; long runs must be left as commands for the user to execute.
- Any command intended for an official run must be written in a README, config comment, or run note with expected outputs.

