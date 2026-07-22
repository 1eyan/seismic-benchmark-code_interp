# Beginner Critique of `docs/tutorial.md`

## How to read this document

This critique evaluates `docs/tutorial.md` from the perspective of a beginner who knows basic PyTorch and Python but has never used this repository or worked with seismic data. Each item is classified by severity, with the affected file and line number(s), the problem, and a concrete suggested fix.

---

## Critical Blockers (would stop a beginner)

- **`docs/tutorial.md:184-192` / `configs/random_noise_suppression/denoise_unet.yaml:151`** — The default inference checkpoint path in the YAML file points to a different repository (`/data/liuqi/code/Seismic-bench/seismic-benchmark-code-main/result/checkpoint/best.pt`), not the experiment output directory the tutorial describes. A beginner who omits `--checkpoint` will either get a "file not found" error or silently load a stale checkpoint from another directory. The tutorial text at line 187 says the default checkpoint is `results/random_noise/random_noise_unet_base/checkpoints/best.pt`, which is not what the config contains.
  - **Suggested fix:** Update `inference.checkpoint` in `denoise_unet.yaml` to `results/random_noise/random_noise_unet_base/checkpoints/best.pt` and add a troubleshooting note that the CLI `--checkpoint` flag overrides the YAML value.

- **`docs/tutorial.md:163-169` / `configs/random_noise_suppression/denoise_unet.yaml:10,142`** — The tutorial claims the default config points to `data/SEG_45Shot_shots1-9.sgy` relative to the repository root, but the committed config uses an absolute, user-specific path (`/data/liuqi/code/Seismic-bench/seismic-benchmark-code/data/...`). A beginner cloning the repo on another machine will hit a `FileNotFoundError` immediately.
  - **Suggested fix:** Change the SEG-Y path in the config to `data/SEG_45Shot_shots1-9.sgy` (relative to repo root) and add a one-line instruction: "Place the SEG-Y file in `data/SEG_45Shot_shots1-9.sgy` before training." Verify the relative path works in the training script.

- **`docs/tutorial.md:419-438`** — The config snippet shown in the tutorial itself contains the same absolute user-specific path as the committed file, reinforcing the mismatch with the prose.
  - **Suggested fix:** Make the tutorial's YAML snippet match the intended public-facing relative path, and add a comment that users should update the path if they placed the file elsewhere.

- **`docs/tutorial.md:490`** — The tutorial states that `SCRN` is a registered model for the random-noise task, but the model file `model/random_noise_suppression/SCRN.py` does not exist and the task `__init__.py` does not import it. Running the `denoise_SCRN.yaml` config or `train_denoise_SCRN.sh` will fail with a "model not registered" error.
  - **Suggested fix:** Either remove `SCRN` from the list of available models in the tutorial, or add the missing model file and import it in `model/random_noise_suppression/__init__.py`.

- **`docs/tutorial.md:887-903` / `configs/ground_roll_attenuation/denoise_unet.yaml:15-16`** — The ground-roll example command can be copied and run, but the config points to `/data/shared/benchmark/ground_roll/...` files that do not exist on a beginner's machine. The tutorial does not explain that these paths must be changed or where to obtain the data.
  - **Suggested fix:** Add a warning before the ground-roll command: "The default config contains absolute paths to data that is not included in the repository. Replace `input_path` and `target_path` with your own paired volumes before running."

- **`docs/tutorial.md:919-938` / `configs/multiples_attenuation/denoise_unet.yaml:15-16`** — Same absolute-path issue as ground-roll; the multiples config points to `/data/shared/benchmark/multiples/...`.
  - **Suggested fix:** Add the same warning before the multiples example command.

- **`docs/tutorial.md:947-964` / `configs/interpolation/interpolation_unet.yaml:10,97`** — The interpolation config uses absolute paths to `/cloud/cloud-ssd1/SEGC3/...`, which a beginner will not have. The tutorial presents the command as if it works out of the box.
  - **Suggested fix:** Add a warning that the interpolation config paths must be updated to point to the user's own SEG-Y volume, and consider providing a relative-path fallback to the same `data/SEG_45Shot_shots1-9.sgy` file used for denoising.

- **`docs/tutorial.md:661-672` / `configs/random_noise_suppression/denoise_unet.yaml:129`** — The tutorial mentions `train.resume` in the YAML and gives an expected `--resume` command, but then says the training script "does not parse `--resume` from the CLI." This is contradictory: the config field is present but unusable, and the tutorial presents a command that would not work.
  - **Suggested fix:** Either implement the `--resume` CLI argument in `train_denoise_unet.py` or remove the `resume` field from the YAML walkthrough and replace the section with a clear "Resuming is not currently supported" note.

- **`docs/tutorial.md:1241-1244`** — The quick-reference CLI table for `ground_roll_attenuation` and `multiples_attenuation` points to `batch_evaluate.py`, which requires `openpyxl`. The tutorial mentions this only in a side note in Chapter 5.3, but the quick-reference table does not include the install command, so a beginner will hit an import error.
  - **Suggested fix:** Add `pip install openpyxl` to the table caption or to the quick-reference command line.

---

## Major Confusions (would significantly slow a beginner)

- **`docs/tutorial.md:42`** — The tutorial uses "FFID" in the prerequisites without defining it, and again in 4.2 without explanation. It also says "You do not need a deep background in seismology," yet the concept is never explained.
  - **Suggested fix:** Add a one-sentence definition the first time "FFID" appears, e.g., "FFID (Field Record ID) is a SEG-Y header value identifying one shot gather."

- **`docs/tutorial.md:245-254` / `docs/tutorial.md:298-302`** — "Shot gather" and "trace" are core terms used throughout the tutorial but are never defined for a non-seismologist.
  - **Suggested fix:** Add a short "What is a shot gather?" sidebar near the data-format section: a shot gather is a 2D image of seismic traces recorded by receivers from one seismic source.

- **`docs/tutorial.md:307-322`** — The `skip` field is introduced with the example `["spherical_divergence_correction"]`, but the tutorial never lists all valid preprocessing step names that can be skipped. A beginner cannot know whether `"normalize"`, `"add_noise"`, or `"mask_traces"` are valid keys.
  - **Suggested fix:** Add a table of skip-able step names and where each is applied in the code.

- **`docs/tutorial.md:366-381`** — `patchify_uniform` is shown with an `output_ndim` argument and a return shape, but the tutorial does not explain what `info` contains or how `unpatchify_uniform` uses it. A beginner trying to understand the pipeline will be stuck.
  - **Suggested fix:** Add a minimal example of calling `patchify_uniform` and then `unpatchify_uniform(patches, info)` with the shape of `info` described.

- **`docs/tutorial.md:383-395`** — `shot_split` is described as splitting "the 9 shots sequentially by FFID," but the volume name `SEG_45Shot_shots1-9.sgy` suggests 45 shots. The tutorial does not reconcile the name with the 9-shot geometry.
  - **Suggested fix:** Add a note: "The file name refers to the original 45-shot survey; the subset shipped here contains shots 1-9, which is why `n_shots = 9`."

- **`docs/tutorial.md:255-261`** — The tutorial says `dt = 0.008` s for the SEG C3 volume, but the random-noise README (`scripts/random_noise_suppression/README_models.md`) says the time sampling interval is `dt = 2 ms`. This inconsistency will confuse a beginner who compares the two documents.
  - **Suggested fix:** Verify the correct `dt` value and update both files consistently.

- **`docs/tutorial.md:576-603`** — The `binned_metrics` block (EB-WSE and FB-FRE) is dumped into the config walkthrough with no conceptual introduction. Acronyms like "Energy-Binned Weak Signal Evaluation" and "Frequency-Binned Fidelity and Recovery Evaluation" are assumed knowledge.
  - **Suggested fix:** Add a brief paragraph before the block explaining that these are optional diagnostic metrics, not required to understand the main task, and that beginners can set `enabled: false` to skip them.

- **`docs/tutorial.md:738-743`** — The "delta" metric group is described as `denoised - noisy`, which reads as a difference of tensors rather than a difference of metric values. A beginner may misinterpret the group.
  - **Suggested fix:** Rephrase: "`delta` is the change in each metric from the noisy input to the denoised output (`denoised_metric - noisy_metric`). Positive values usually mean improvement."

- **`docs/tutorial.md:547-557`** — The `log_step` and `log_interval` fields are described as "if `true`, log every training step; otherwise log every `log_interval` steps." It is unclear whether this controls per-step logging in addition to per-epoch logging, or replaces it.
  - **Suggested fix:** State explicitly whether `log_step: false` still logs one summary per epoch, and what `log_interval` counts (batches within an epoch).

- **`docs/tutorial.md:264-286`** — The NPY/MAT example shows the `key` field as optional, but the prose below says "if the configured `key` is not present in the file, the loader raises a `KeyError`." A beginner may not know when to set `key` or how to discover the variable names in a MAT file.
  - **Suggested fix:** Add an example MAT inspection command, e.g., `python -c "import scipy.io; print(scipy.io.loadmat('file.mat').keys())"`, and clarify that `key` is only required for MAT files and defaults to the first array if omitted.

- **`docs/tutorial.md:699-709`** — The Chapter 3 inference command omits `--device` and `--batch-size`, but the Chapter 4.5 command includes them. The tutorial never explains why the same command differs between chapters.
  - **Suggested fix:** Add a sentence after the Chapter 3 command: "The device and batch-size can also be overridden; see the full list of CLI flags in Chapter 4.5."

- **`docs/tutorial.md:1246-1260`** — The YAML quick-reference table lists `preprocess` fields such as `mask_mode`, `mask_ratio`, `uniform_stride`, and `clip_percentile` that do not appear in the random-noise config walkthrough. A beginner may try to add them to the denoising config without understanding which task they belong to.
  - **Suggested fix:** Mark interpolation-only or paired-task-only fields in the quick-reference table.

- **`docs/tutorial.md:676-693`** — The multi-GPU section shows `torchrun` without explaining that it is part of PyTorch's distributed launcher, or that `CUDA_VISIBLE_DEVICES` controls which GPUs are used. A beginner with a single GPU may think this command is required.
  - **Suggested fix:** Add a note: "Multi-GPU is optional. For a single GPU, use the single-GPU command in section 4.4."

- **`docs/tutorial.md:779-862`** — The shell sweep section introduces shell variables like `NPROC_PER_NODE`, `TORCHRUN_EXTRA`, and `STOP_ON_ERROR`, but does not explain how to inspect the generated experiment names or how to resume an interrupted sweep.
  - **Suggested fix:** Add a short example of reading the generated `metrics_summary_mean_std.json` and mention that the scripts are best-effort helpers that require reading the `.sh` files before use.

---

## Minor Polish (wording, formatting, consistency)

- **`configs/interpolation/interpolation_unet.yaml:81-84`** — The interpolation config contains Chinese comments (`false: 仅每个 epoch 输出一次...`), violating the project rule that all source comments must be in English.
  - **Suggested fix:** Translate the comments to English.

- **`docs/tutorial.md:200-216`** — The expected output tree shows periodic checkpoints at `epoch_0020.pt`, `epoch_0040.pt`, etc., but the config has `ckpt_interval: 20` and `epochs: 200`, so the final checkpoint is `epoch_0200.pt`. The ellipsis obscures this.
  - **Suggested fix:** Show `epoch_0200.pt` explicitly or note that checkpoints are saved every 20 epochs up to `epoch_0200.pt`.

- **`docs/tutorial.md:81`** — The `model/` directory is described as containing "task-specific subpackages," but the registry explanation earlier says `model/` is for neural network definitions. The relationship between `model/registry.py` and `model/<task>/` could be clearer.
  - **Suggested fix:** Add one sentence: "Shared registration primitives live in `model/registry.py`; task-specific model files live in `model/<task>/`."

- **`docs/tutorial.md:114`** — The custom-model pseudocode is placed before the task-specific `__init__.py` import, which is correct, but the example does not show how `build_model` actually locates the model from the task subpackage.
  - **Suggested fix:** Show the import path used by the training script, e.g., `from model.random_noise_suppression import build_model`.

- **`docs/tutorial.md:298-302`** — The shape-convention paragraph says "When a 2D conv model operates on a single shot, the patch shape is `(n_traces, n_time)`," but the actual patch shape is `(patch_trace, patch_time)`, not the full shot dimensions.
  - **Suggested fix:** Rephrase: "The model receives patches of shape `(1, patch_trace, patch_time)`, where the leading `1` is the channel dimension."

- **`docs/tutorial.md:489-491`** — The sentence about `SCRN` is split across two ideas: that it is a registered model, and that it is not in the default sweep. Because the first claim is false, the parenthetical is confusing.
  - **Suggested fix:** Remove the SCRN sentence entirely until the model is implemented, or move it to a footnote.

- **`docs/tutorial.md:1240-1244`** — The CLI quick-reference table is very wide and will wrap in most terminal/readers. The long single-line commands are hard to read.
  - **Suggested fix:** Use multi-line fenced commands under each task instead of a single table row, or break each command across lines with `\`.

- **`docs/tutorial.md:1248-1260`** — The quick-reference table lists `inference` fields such as `save_npy` and `binned_metrics` but the `preprocess` table does not mention `noise_kind`, `snr_db`, or `mask_ratio`, which are important for the worked examples.
  - **Suggested fix:** Add `noise_kind`, `snr_db`, and `mask_ratio` to the `preprocess` quick-reference row.

- **`docs/tutorial.md:1263-1270`** — The registry cheat sheet says the dataset factory is `build_dataset(cfg) / build_dataloader(cfg)`, but the tutorial never shows `build_dataloader` in a YAML or CLI context.
  - **Suggested fix:** Either remove `build_dataloader` from the cheat sheet or add a one-line note that it is called internally by the training script.

- **`docs/tutorial.md:1272-1281`** — The metric arguments table lists `min_signal_energy` for `snr`, but the tutorial never explains when a user should change this value.
  - **Suggested fix:** Add a one-sentence note: "Use `min_signal_energy` to avoid division by zero when a shot has near-zero signal energy."

---

## Overall Assessment

**Is the tutorial learnable end-to-end?** Almost, but not without manual edits that a beginner will not know to make. The random-noise example is the strongest part, but the committed config file contains an absolute user-specific data path and a checkpoint path pointing to a different directory. Those two issues alone will stop most first-time users. The other tasks (ground-roll, multiples, interpolation) are documented as if they work out of the box, but their configs point to data that is not included in the repository. The `SCRN` model is referenced but not implemented, which will cause a registration error.

**Single biggest improvement:** Update the default `configs/random_noise_suppression/denoise_unet.yaml` so that its data and checkpoint paths are relative to the repository root and match the experiment output tree described in the tutorial. Add a short "Before you run" checklist telling the user to place the SEG-Y file in `data/` and to verify or override absolute paths in the other task configs. This would remove the most common first-run failures and make the tutorial actually followable by a beginner.