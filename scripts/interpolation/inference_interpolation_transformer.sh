#!/usr/bin/env bash
# Transformer interpolation inference wrapper.
#
# Supports two pipelines:
#   PIPELINE=gated -> inference_interpolation_transformer.py (token-based)
#   PIPELINE=patch -> inference_interpolation.py (patch-based, same data flow as U-Net)
#
# Usage:
#   # Gated Transformer (default)
#   bash scripts/interpolation/inference_interpolation_transformer.sh
#
#   # Patch Transformer (trace_token)
#   PIPELINE=patch CONFIG=configs/interpolation/interpolation_trace_transformer.yaml \
#     bash scripts/interpolation/inference_interpolation_transformer.sh
#
#   # Patch Transformer (hf_vit)
#   PIPELINE=patch CONFIG=configs/interpolation/interpolation_hf_vit.yaml \
#     bash scripts/interpolation/inference_interpolation_transformer.sh

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Pipeline selection: "gated" | "patch"
PIPELINE="${PIPELINE:-gated}"

# Config path (gated default: interpolation_transformer.yaml)
# Override CONFIG when PIPELINE=patch to select the Patch Transformer config.
DEFAULT_GATED_CONFIG="configs/interpolation/interpolation_transformer.yaml"
DEFAULT_PATCH_CONFIG="configs/interpolation/interpolation_trace_transformer.yaml"
CONFIG="${CONFIG:-}"

# Inference overrides (leave empty to read from config / use defaults)
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
N_VIZ_SHOTS="${N_VIZ_SHOTS:-}"
SEED="${SEED:-}"
DEVICE="${DEVICE:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
SAVE_NPY="${SAVE_NPY:-}"

# Mask overrides (must match training)
MASK_MODE="${MASK_MODE:-}"
MASK_RATIO="${MASK_RATIO:-}"
CONTINUOUS_MISSING_TRACES="${CONTINUOUS_MISSING_TRACES:-}"
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES

# --- Resolve pipeline ---
case "${PIPELINE}" in
  gated)
    if [[ -n "${CONFIG}" ]]; then
      CFG_PATH="${REPO_ROOT}/${CONFIG}"
    else
      CFG_PATH="${REPO_ROOT}/${DEFAULT_GATED_CONFIG}"
    fi
    PY_SCRIPT="${REPO_ROOT}/scripts/interpolation/inference_interpolation_transformer.py"
    ;;
  patch)
    if [[ -n "${CONFIG}" ]]; then
      CFG_PATH="${REPO_ROOT}/${CONFIG}"
    else
      CFG_PATH="${REPO_ROOT}/${DEFAULT_PATCH_CONFIG}"
    fi
    PY_SCRIPT="${REPO_ROOT}/scripts/interpolation/inference_interpolation.py"
    ;;
  *)
    echo "ERROR: Unknown PIPELINE=${PIPELINE}. Use 'gated' or 'patch'." >&2
    exit 1
    ;;
esac

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "Script not found: ${PY_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${CFG_PATH}" ]]; then
  echo "Config not found: ${CFG_PATH}" >&2
  exit 1
fi

# --- Build argument list ---
args=(--config "${CFG_PATH}")

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
if [[ -n "${SAVE_NPY}" ]]; then
  args+=(--save-npy)
fi
if [[ -n "${MASK_MODE}" ]]; then
  args+=(--mask-mode "${MASK_MODE}")
fi
if [[ -n "${MASK_RATIO}" ]]; then
  args+=(--mask-ratio "${MASK_RATIO}")
fi
if [[ -n "${CONTINUOUS_MISSING_TRACES}" ]]; then
  args+=(--continuous-missing-traces "${CONTINUOUS_MISSING_TRACES}")
fi

echo "Pipeline: ${PIPELINE}"
echo "Config:   ${CFG_PATH}"
echo "Script:   ${PY_SCRIPT}"
echo ""

cd "${REPO_ROOT}"
python "${PY_SCRIPT}" "${args[@]}"

echo "[$(date -Iseconds)] Inference finished."
