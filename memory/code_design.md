# Code Design

> Records the project's **architectural intent** and **module layout decisions**. Append entries in chronological order; never rewrite history.

## Module boundaries (initial agreement)

- `model/` — Network structures only; no training loops.
- `utils/` — Training infrastructure (datasets, losses, metrics, visualization, logging, training utilities, position encoding, masking).
- `tools/` — Cross-cutting helpers (I/O, coordinates, data conversion).
- `scripts/` — CLI parsing and wiring only.
- `configs/` — Hyper-parameters; never hard-coded in source.
- `results/` — Experiment artifacts; not tracked by Git.

## Entry template

```markdown
## YYYY-MM-DD - Title
- Context:
- Change:
- Impact:
- Follow-up:
```

## Maintenance Guide

This section is **prescriptive**: it defines how the codebase grows without breaking the training entry point.

### Architectural invariants

- `scripts/train.py` is **component-agnostic**. It must never import a concrete dataset, model, loss, or metric; only the corresponding `build_*` factories.
- Every pluggable component follows the **registry + factory** pattern:
  - `model/registry.py` — `MODEL_REGISTRY`, `@register_model`, `build_model`.
  - `utils/datasets.py` — `DATASET_REGISTRY`, `@register_dataset`, `build_dataset`, `build_dataloader`.
  - `utils/losses.py` — `LOSS_REGISTRY`, `@register_loss`, `build_loss`.
  - `utils/metrics.py` — `METRIC_REGISTRY`, `@register_metric`, `build_metrics`.
- Configuration drives behavior: every pluggable block in YAML is `{ type: <name>, params: {...} }`. New optional fields must get a sensible default in the corresponding builder.
- One responsibility per file: datasets, losses, metrics, visualization, logging, and training utilities each live in their own module under `utils/`.

### Adding a new implementation (loss / model / dataset / metric)

1. Create a new file under the right package (`model/` or `utils/`).
2. Subclass the corresponding base class and decorate with `@register_<kind>("name")`.
3. If it lives in `model/`, also add `from . import <new_file>  # noqa: F401` to `model/__init__.py` so the decorator runs at import time.
4. Reference it from YAML via its registered name.
5. Neither `scripts/train.py` nor `utils/__init__.py` should need changes.
6. Append an entry to `memory/updates.md` (name, location, reference source).

### Adding a new config field

1. Add the field to `configs/default.yaml` with a comment describing its purpose and default.
2. Read it in the appropriate builder (e.g. `build_scheduler`) with a default fallback so older configs still work.
3. Document the field briefly in `configs/README.md`.
4. Log the change in `memory/updates.md`.

### Adding a new entry script

1. Place it under `scripts/`, mirroring the structure of `train.py` (CLI parsing + factories + loop).
2. Do not implement algorithms inside entry scripts.
3. Document the command and expected outputs in `scripts/README.md`.

### Checklist before merging a change

- No business logic was added to `scripts/` or `tools/` (goes into `utils/` or `model/`).
- No new duplicate helper exists for something already in `utils/` / `tools/`.
- New fields added to `configs/default.yaml` have documented defaults.
- If an open-source implementation was referenced, the source is cited in the file header and in `memory/updates.md`.
- `memory/updates.md` carries a new dated entry for the change.

---

<!-- Append new entries below this line -->
