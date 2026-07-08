#!/usr/bin/env bash
# Train + inference loop for Transformer-based seismic trace interpolation.
#
# Supports two pipelines via PIPELINE env var:
#   PIPELINE=gated -> Gated Transformer (token-based, coords+mask)
#                     train_interpolation_transformer.py
#                     inference_interpolation_transformer.py
#   PIPELINE=patch -> Patch Transformer (ViT/trace-token, 2D patches)
#                     train_interpolation_patch_transformer.py
#                     inference_interpolation.py (same data flow as U-Net)
#
# This script supports both:
#   1) ratio-based missing traces:
#        random:0.3
#        uniform:0.5
#        continuous:0.3
#
#   2) fixed contiguous missing traces:
#        continuous:20tr
#        continuous:30tr
#        continuous:40tr
#
# Recommended examples:
#   EXPERIMENTS=("random:0.3" "uniform:0.5" "continuous:30tr")
#
# Naming convention:
#   ratio-based experiments:
#       <base>_seed42_random_miss30
#       <base>_seed42_uniform_miss50
#       <base>_seed42_continuous_miss30
#
#   fixed-trace continuous experiments:
#       <base>_seed42_continuous_miss20tr
#       <base>_seed42_continuous_miss30tr
#
# Usage:
#   # Gated Transformer (default)
#   bash scripts/interpolation/train_infer_loop_transformer.sh
#
#   # Patch Transformer (trace_token)
#   PIPELINE=patch \
#   BASE_CONFIG=configs/interpolation/interpolation_trace_transformer.yaml \
#     bash scripts/interpolation/train_infer_loop_transformer.sh
#
#   # Patch Transformer (hf_vit)
#   PIPELINE=patch \
#   BASE_CONFIG=configs/interpolation/interpolation_hf_vit.yaml \
#     bash scripts/interpolation/train_infer_loop_transformer.sh

set -euo pipefail

# ==============================================================================
# User configuration
# ==============================================================================

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-auto}"
TORCHRUN_EXTRA="${TORCHRUN_EXTRA:-}"

# Pipeline selection: "gated" | "patch"
PIPELINE="${PIPELINE:-gated}"

# Base config (default per pipeline).
# Override BASE_CONFIG when PIPELINE=patch to select the Patch Transformer config.
DEFAULT_GATED_CONFIG="configs/interpolation/interpolation_transformer.yaml"
DEFAULT_PATCH_CONFIG="configs/interpolation/interpolation_trace_transformer.yaml"
BASE_CONFIG="${BASE_CONFIG:-}"

# Resolve training and inference scripts per pipeline (set below after config).
TRAIN_PY=""
INFER_PY=""

# Unified experiment list.
# Format:
#   "<mask_mode>:<ratio>"      e.g., "random:0.3", "uniform:0.5", "continuous:0.3"
#   "<mask_mode>:<N>tr"        e.g., "continuous:20tr"
#
# Fixed trace count is only meaningful for continuous missing traces.
EXPERIMENTS=(
  "random:0.3"
  "random:0.5"
  "random:0.7"
  "uniform:0.3"
  "uniform:0.5"
  "continuous:20tr"
  "continuous:30tr"
  "continuous:40tr"
)

N_SEEDS=3
START_SEED=42

INFER_DEVICE="${INFER_DEVICE:-cuda:2}"
RUN_INFERENCE=true
RUN_EXTRA_INFERENCE=true

# Extra inference for generalization tests.
# For ratio-based experiments, extra ratio = ratio + EXTRA_RATIO_STEP.
# For fixed-trace experiments, extra missing traces = missing_traces + EXTRA_TRACES_STEP.
EXTRA_RATIO_STEP="0.1"
EXTRA_TRACES_STEP=10

# If true, use best.pt when available; otherwise use latest epoch_*.pt.
PREFER_BEST_CHECKPOINT=true

# If true, only print commands.
DRY_RUN=false

# ==============================================================================
# Helper functions
# ==============================================================================

log() {
  echo "[$(date -Iseconds)] $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

run_cmd() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "true" ]]; then
    "$@"
  fi
}

pick_port() {
  python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

ratio_to_pct() {
  awk -v r="$1" 'BEGIN { printf "%d", r * 100 + 0.5 }'
}

ratio_add() {
  awk -v a="$1" -v b="$2" 'BEGIN { printf "%.3f", a + b }'
}

get_experiment_name() {
  local config_path="$1"
  grep -m1 -E '^[[:space:]]*name:[[:space:]]*' "${config_path}" \
    | sed -E 's/^[[:space:]]*name:[[:space:]]*//' \
    | sed -E 's/[[:space:]]+#.*$//;s/[[:space:]]*$//'
}

parse_experiment() {
  local spec="$1"

  if [[ "${spec}" != *:* ]]; then
    die "Invalid experiment spec: ${spec}. Expected format '<mask_mode>:<ratio>' or 'continuous:<N>tr'."
  fi

  EXP_MASK_MODE="${spec%%:*}"
  EXP_VALUE="${spec#*:}"

  case "${EXP_MASK_MODE}" in
    uniform|random|continuous)
      ;;
    *)
      die "Invalid mask mode: ${EXP_MASK_MODE}. Expected uniform, random, or continuous."
      ;;
  esac

  if [[ "${EXP_VALUE}" =~ ^[0-9]+tr$ ]]; then
    EXP_KIND="fixed_traces"
    EXP_MISSING_TRACES="${EXP_VALUE%tr}"
    EXP_MASK_RATIO=""

    if [[ "${EXP_MASK_MODE}" != "continuous" ]]; then
      die "Fixed trace count is only supported for continuous masking, got: ${spec}"
    fi

    if [[ "${EXP_MISSING_TRACES}" -le 0 ]]; then
      die "Missing trace count must be positive, got: ${EXP_MISSING_TRACES}"
    fi
  else
    EXP_KIND="ratio"
    EXP_MASK_RATIO="${EXP_VALUE}"
    EXP_MISSING_TRACES=""

    awk -v r="${EXP_MASK_RATIO}" 'BEGIN { exit !(r > 0 && r < 1) }' \
      || die "Mask ratio must be in (0, 1), got: ${EXP_MASK_RATIO}"
  fi
}

run_suffix() {
  if [[ "${EXP_KIND}" == "fixed_traces" ]]; then
    echo "${EXP_MASK_MODE}_miss${EXP_MISSING_TRACES}tr"
  else
    local pct
    pct="$(ratio_to_pct "${EXP_MASK_RATIO}")"
    echo "${EXP_MASK_MODE}_miss${pct}"
  fi
}

train_args() {
  if [[ "${EXP_KIND}" == "fixed_traces" ]]; then
    echo "--mask-mode ${EXP_MASK_MODE} --continuous-missing-traces ${EXP_MISSING_TRACES}"
  else
    echo "--mask-mode ${EXP_MASK_MODE} --mask-ratio ${EXP_MASK_RATIO}"
  fi
}

infer_args() {
  local kind="$1"
  local mode="$2"
  local value="$3"

  if [[ "${kind}" == "fixed_traces" ]]; then
    echo "--mask-mode ${mode} --continuous-missing-traces ${value}"
  else
    echo "--mask-mode ${mode} --mask-ratio ${value}"
  fi
}

find_checkpoint() {
  local ckpt_dir="$1"
  local ckpt=""

  if [[ "${PREFER_BEST_CHECKPOINT}" == "true" && -f "${ckpt_dir}/best.pt" ]]; then
    echo "${ckpt_dir}/best.pt"
    return 0
  fi

  ckpt="$(ls -t "${ckpt_dir}"/epoch_*.pt 2>/dev/null | head -1 || true)"
  if [[ -n "${ckpt}" ]]; then
    echo "${ckpt}"
    return 0
  fi

  return 1
}

# ==============================================================================
# Initialization
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES

# --- Resolve pipeline ---
case "${PIPELINE}" in
  gated)
    if [[ -z "${BASE_CONFIG}" ]]; then
      BASE_CONFIG="${DEFAULT_GATED_CONFIG}"
    fi
    TRAIN_PY="scripts/interpolation/train_interpolation_transformer.py"
    INFER_PY="scripts/interpolation/inference_interpolation_transformer.py"
    ;;
  patch)
    if [[ -z "${BASE_CONFIG}" ]]; then
      BASE_CONFIG="${DEFAULT_PATCH_CONFIG}"
    fi
    TRAIN_PY="scripts/interpolation/train_interpolation_patch_transformer.py"
    INFER_PY="scripts/interpolation/inference_interpolation.py"
    ;;
  *)
    die "Unknown PIPELINE=${PIPELINE}. Use 'gated' or 'patch'."
    ;;
esac

[[ -f "${REPO_ROOT}/${BASE_CONFIG}" ]] || die "Config not found: ${REPO_ROOT}/${BASE_CONFIG}"
[[ -f "${REPO_ROOT}/${TRAIN_PY}" ]] || die "Training script not found: ${REPO_ROOT}/${TRAIN_PY}"
[[ -f "${REPO_ROOT}/${INFER_PY}" ]] || die "Inference script not found: ${REPO_ROOT}/${INFER_PY}"

NAME_BASE="$(get_experiment_name "${REPO_ROOT}/${BASE_CONFIG}")"
[[ -n "${NAME_BASE}" ]] || die "Could not parse experiment.name from ${BASE_CONFIG}"

TOTAL_RUNS=$(( ${#EXPERIMENTS[@]} * N_SEEDS ))
run_idx=0

log "Pipeline: ${PIPELINE}"
log "Repository root: ${REPO_ROOT}"
log "Base config: ${BASE_CONFIG}"
log "Experiment base name: ${NAME_BASE}"
log "Train script: ${TRAIN_PY}"
log "Infer script: ${INFER_PY}"
log "Total runs: ${TOTAL_RUNS}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ==============================================================================
# Main loop
# ==============================================================================

for spec in "${EXPERIMENTS[@]}"; do
  parse_experiment "${spec}"

  for ((i = 0; i < N_SEEDS; i++)); do
    seed=$((START_SEED + i))
    run_idx=$((run_idx + 1))

    suffix="$(run_suffix)"
    run_name_sed="${NAME_BASE}_seed${seed}"
    run_name_full="${run_name_sed}_${suffix}"

    log "================================================================"
    log "[${run_idx}/${TOTAL_RUNS}] spec=${spec}, seed=${seed}"
    log "Experiment: ${run_name_full}"
    log "================================================================"

    tmpcfg="$(mktemp)"
    sed -E \
      -e 's/^([[:space:]]*seed:[[:space:]]*)[0-9]+$/\1'"${seed}"'/' \
      -e 's/^([[:space:]]*name:[[:space:]]*).*/\1'"${run_name_sed}"'/' \
      "${REPO_ROOT}/${BASE_CONFIG}" > "${tmpcfg}"

    cd "${REPO_ROOT}"

    if [[ "${MASTER_PORT}" == "auto" ]]; then
      master_port="$(pick_port)"
    else
      master_port="${MASTER_PORT}"
    fi

    train_args_string="$(train_args)"
    read -r -a train_args_array <<< "${train_args_string}"

    log "Training started."
    # shellcheck disable=SC2086
    run_cmd torchrun ${TORCHRUN_EXTRA} \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_port="${master_port}" \
      "${TRAIN_PY}" \
      --config "${tmpcfg}" \
      "${train_args_array[@]}"

    ckpt_dir="${REPO_ROOT}/results/${run_name_full}/checkpoints"
    if ! checkpoint="$(find_checkpoint "${ckpt_dir}")"; then
      log "WARNING: No checkpoint found in ${ckpt_dir}; skipping inference."
      rm -f "${tmpcfg}"
      continue
    fi
    log "Using checkpoint: ${checkpoint}"

    if [[ "${RUN_INFERENCE}" == "true" ]]; then
      if [[ "${EXP_KIND}" == "fixed_traces" ]]; then
        infer_args_string="$(infer_args fixed_traces "${EXP_MASK_MODE}" "${EXP_MISSING_TRACES}")"
      else
        infer_args_string="$(infer_args ratio "${EXP_MASK_MODE}" "${EXP_MASK_RATIO}")"
      fi
      read -r -a infer_args_array <<< "${infer_args_string}"

      infer_out="${REPO_ROOT}/results/${run_name_full}/inference"

      log "Inference started: matching training missing setting."
      run_cmd python "${REPO_ROOT}/${INFER_PY}" \
        --config "${tmpcfg}" \
        --checkpoint "${checkpoint}" \
        --output-dir "${infer_out}" \
        "${infer_args_array[@]}" \
        --device "${INFER_DEVICE}"
    fi

    if [[ "${RUN_EXTRA_INFERENCE}" == "true" ]]; then
      if [[ "${EXP_KIND}" == "fixed_traces" ]]; then
        extra_kind="fixed_traces"
        extra_value=$((EXP_MISSING_TRACES + EXTRA_TRACES_STEP))
        extra_suffix="miss${extra_value}tr"
      else
        extra_kind="ratio"
        extra_value="$(ratio_add "${EXP_MASK_RATIO}" "${EXTRA_RATIO_STEP}")"
        extra_suffix="ratio$(ratio_to_pct "${extra_value}")"
      fi

      extra_args_string="$(infer_args "${extra_kind}" "${EXP_MASK_MODE}" "${extra_value}")"
      read -r -a extra_args_array <<< "${extra_args_string}"

      infer_out_extra="${REPO_ROOT}/results/${run_name_full}/inference_${extra_suffix}"

      log "Extra inference started: missing=${extra_value}."
      run_cmd python "${REPO_ROOT}/${INFER_PY}" \
        --config "${tmpcfg}" \
        --checkpoint "${checkpoint}" \
        --output-dir "${infer_out_extra}" \
        "${extra_args_array[@]}" \
        --device "${INFER_DEVICE}"
    fi

    log "Done: ${run_name_full}"
    echo ""

    rm -f "${tmpcfg}"
  done
done

log "All ${TOTAL_RUNS} runs complete."
