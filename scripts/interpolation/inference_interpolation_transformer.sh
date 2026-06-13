#!/usr/bin/env bash
# Transformer interpolation inference wrapper.
# Edit the configuration block below as needed.

set -euo pipefail

# ---------- Configuration ----------
CUDA_VISIBLE_DEVICES="0"
CONFIG="configs/interpolation/interpolation_transformer.yaml"
# Leave empty to read from config.inference
CHECKPOINT=""
OUTPUT_DIR=""
N_VIZ_SHOTS=""
SEED=""
DEVICE=""
BATCH_SIZE=""
# ------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PY_SCRIPT="${REPO_ROOT}/scripts/interpolation/inference_interpolation_transformer.py"

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "Script not found: ${PY_SCRIPT}" >&2
  exit 1
fi

args=(--config "${REPO_ROOT}/${CONFIG}")

if [[ -n "${CHECKPOINT}" ]]; then
  args+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  args+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${N_VIZ_SHOTS}" ]]; then
  args+=(--n-viz-shots "${N_VIZ_SHOTS}")
fi
if [[ -n "${SEED}" ]]; then
  args+=(--seed "${SEED}")
fi
if [[ -n "${DEVICE}" ]]; then
  args+=(--device "${DEVICE}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  args+=(--batch-size "${BATCH_SIZE}")
fi

cd "${REPO_ROOT}"
python "${PY_SCRIPT}" "${args[@]}"

echo "[$(date -Iseconds)] Inference finished."
