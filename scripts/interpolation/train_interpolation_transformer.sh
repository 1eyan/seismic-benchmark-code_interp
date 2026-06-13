#!/usr/bin/env bash
# Multi-seed transformer interpolation training loop.
#
# Supports two pipelines:
#   PIPELINE=gated -> Gated Transformer (token-based, coords+mask)
#                     train_interpolation_transformer.py + interpolation_transformer.yaml
#   PIPELINE=patch -> Patch Transformer (ViT/trace-token, 2D patches)
#                     train_interpolation_patch_transformer.py + PATCH_CONFIG
#
# Usage:
#   # Gated Transformer (default)
#   bash scripts/interpolation/train_interpolation_transformer.sh
#
#   # Patch Transformer (trace_token)
#   PIPELINE=patch PATCH_CONFIG=configs/interpolation/interpolation_trace_transformer.yaml \
#     bash scripts/interpolation/train_interpolation_transformer.sh
#
#   # Patch Transformer (hf_vit)
#   PIPELINE=patch PATCH_CONFIG=configs/interpolation/interpolation_hf_vit.yaml \
#     bash scripts/interpolation/train_interpolation_transformer.sh

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
N_SEEDS=3
START_SEED=42
MASK_MODE="continuous"
MASK_RATIO="0.1"
MASTER_PORT="${MASTER_PORT:-auto}"
TORCHRUN_EXTRA="${TORCHRUN_EXTRA:-}"

# Pipeline selection: "gated" | "patch"
#   gated -> train_interpolation_transformer.py + interpolation_transformer.yaml
#   patch -> train_interpolation_patch_transformer.py + PATCH_CONFIG
PIPELINE="${PIPELINE:-gated}"

# Patch Transformer config (only used when PIPELINE=patch)
# Options: interpolation_trace_transformer.yaml / interpolation_hf_vit.yaml
PATCH_CONFIG="${PATCH_CONFIG:-configs/interpolation/interpolation_trace_transformer.yaml}"
# ==============================================================================

_pick_port() {
  python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES

# --- Resolve pipeline ---
case "${PIPELINE}" in
  gated)
    BASE_CONFIG="${REPO_ROOT}/configs/interpolation/interpolation_transformer.yaml"
    PY_SCRIPT="${REPO_ROOT}/scripts/interpolation/train_interpolation_transformer.py"
    ;;
  patch)
    BASE_CONFIG="${REPO_ROOT}/${PATCH_CONFIG}"
    PY_SCRIPT="${REPO_ROOT}/scripts/interpolation/train_interpolation_patch_transformer.py"
    ;;
  *)
    echo "ERROR: Unknown PIPELINE=${PIPELINE}. Use 'gated' or 'patch'." >&2
    exit 1
    ;;
esac

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Config not found: ${BASE_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "Script not found: ${PY_SCRIPT}" >&2
  exit 1
fi

NAME_BASE="$(grep -m1 -E '^[[:space:]]*name:[[:space:]]*' "${BASE_CONFIG}" \
  | sed -E 's/^[[:space:]]*name:[[:space:]]*//' \
  | sed -E 's/[[:space:]]+#.*$//;s/[[:space:]]*$//')"
if [[ -z "${NAME_BASE}" ]]; then
  echo "Could not parse experiment.name from ${BASE_CONFIG}" >&2
  exit 1
fi

echo "Pipeline: ${PIPELINE}"
echo "Config:   ${BASE_CONFIG}"
echo "Script:   ${PY_SCRIPT}"
echo "Seeds:    ${START_SEED}..$((START_SEED + N_SEEDS - 1)) (${N_SEEDS} runs)"
echo ""

tmpcfg="$(mktemp)"
cleanup() { rm -f "${tmpcfg}"; }
trap cleanup EXIT

for ((i = 0; i < N_SEEDS; i++)); do
  seed=$((START_SEED + i))
  run_name="${NAME_BASE}_seed${seed}"
  sed -E \
    -e 's/^([[:space:]]*seed:[[:space:]]*)[0-9]+$/\1'"${seed}"'/' \
    -e 's/^([[:space:]]*name:[[:space:]]*).*/\1'"${run_name}"'/' \
    "${BASE_CONFIG}" >"${tmpcfg}"
  echo "[$(date -Iseconds)] (${i}+1)/${N_SEEDS} name=${run_name} seed=${seed}"
  cd "${REPO_ROOT}"
  if [[ "${MASTER_PORT}" == "auto" ]]; then
    master_port=$(_pick_port)
  else
    master_port="${MASTER_PORT}"
  fi
  # shellcheck disable=SC2086
  torchrun ${TORCHRUN_EXTRA} --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${master_port}" \
    "${PY_SCRIPT}" --config "${tmpcfg}" \
    --mask-mode "${MASK_MODE}" --mask-ratio "${MASK_RATIO}"
done

echo "[$(date -Iseconds)] Done ${N_SEEDS} runs (seed ${START_SEED}..$((START_SEED + N_SEEDS - 1)))."
