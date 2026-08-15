#!/usr/bin/env bash
# Train + inference loop for Park2022 CFunet across three missing scenarios.
#
# Scenario -> training mask mapping (per user requirement):
#   random missing   -> trained with cfunet_random  (paper per-patch protocol)
#   uniform missing  -> trained with cfunet_random  (shares the same model)
#   continuous missing-> trained with one model per level (20tr/30tr/40tr)
#
# This script runs the two training families back-to-back (batched by family):
#   Family A: trains cfunet_random  (park2022_cfunet_paper.yaml) for every
#             seed, then evaluates each on random + uniform shot-level masks;
#   Family B: trains one continuous model per level (20tr/30tr/40tr) with
#             --continuous-missing-traces matching inference, then evaluates
#             each model on its matching continuous shot-level mask.
#
# Inference resolves the checkpoint and the exact training-time config from the
# experiment directory (<experiment.output_dir>/<exp>/config.yaml +
# checkpoints/best.pt), then overrides the mask via CLI, so the trained
# parameters are always found.
#
# Usage:
#   bash scripts/interpolation/train_infer_loop_cfunet.sh
#
# Env (all optional):
#   CUDA_VISIBLE_DEVICES, NPROC_PER_NODE, MASTER_PORT, TORCHRUN_EXTRA
#   INFER_DEVICE, N_SEEDS, START_SEED, DRY_RUN
#   RUN_INFERENCE, RUN_EXTRA_INFERENCE, EXTRA_RATIO_STEP, EXTRA_TRACES_STEP
#   PREFER_BEST_CHECKPOINT

set -euo pipefail

# ==============================================================================
# User configuration
# ==============================================================================

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-auto}"
TORCHRUN_EXTRA="${TORCHRUN_EXTRA:-}"

CFUNET_RANDOM_CONFIG="${CFUNET_RANDOM_CONFIG:-configs/interpolation/park2022_cfunet_paper.yaml}"
CONTINUOUS_CONFIG="${CONTINUOUS_CONFIG:-configs/interpolation/park2022_cfunet_continuous.yaml}"
TRAIN_PY="${TRAIN_PY:-scripts/interpolation/train_interpolation_unet.py}"
INFER_PY="${INFER_PY:-scripts/interpolation/inference_interpolation.py}"

# Shot-level evaluation masks per training family.
# Format: "<mask_mode>:<ratio>" or "continuous:<N>tr".
# Set RANDOM_UNIFORM_EVAL_OVERRIDE to a space-separated list to run a subset,
# e.g. RANDOM_UNIFORM_EVAL_OVERRIDE="uniform:0.7"
if [[ -n "${RANDOM_UNIFORM_EVAL_OVERRIDE:-}" ]]; then
  read -r -a RANDOM_UNIFORM_EVAL <<< "${RANDOM_UNIFORM_EVAL_OVERRIDE}"
else
  RANDOM_UNIFORM_EVAL=(
    "random:0.3"
    "random:0.5"
    "uniform:0.5"
    "uniform:0.7"
  )
fi
CONTINUOUS_EVAL=(
  "continuous:20tr"
  "continuous:30tr"
  "continuous:40tr"
)

N_SEEDS="${N_SEEDS:-3}"
START_SEED="${START_SEED:-42}"

INFER_DEVICE="${INFER_DEVICE:-cuda:0}"
RUN_INFERENCE="${RUN_INFERENCE:-true}"
RUN_EXTRA_INFERENCE="${RUN_EXTRA_INFERENCE:-true}"
RUN_CONTINUOUS="${RUN_CONTINUOUS:-true}"

# Generalization inference: ratio specs use ratio+step, fixed-trace specs use
# missing_traces+step (mirrors train_infer_loop.sh).
EXTRA_RATIO_STEP="${EXTRA_RATIO_STEP:-0.1}"
EXTRA_TRACES_STEP="${EXTRA_TRACES_STEP:-10}"

PREFER_BEST_CHECKPOINT="${PREFER_BEST_CHECKPOINT:-true}"
DRY_RUN="${DRY_RUN:-false}"

# Training-mask suffix appended to the experiment name by
# train_interpolation_unet.py main().  These must stay in sync with the mask
# settings in the two configs above:
#   cfunet_random: mask_ratio_range [0.5, 0.875] -> cfunet_random_miss50-88
#   continuous:    --continuous-missing-traces N  -> continuous_miss{N}tr
#                  (one model per level; config mask_ratio is overridden via CLI)
CFUNET_RANDOM_SUFFIX="cfunet_random_miss50-88"

# ==============================================================================
# Helper functions
# ==============================================================================

log() {
  echo "[$(date -Iseconds)] $*" >&2
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

run_cmd() {
  echo "+ $*" >&2
  if [[ "${DRY_RUN}" != "true" ]]; then
    "$@" >&2
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

get_experiment_output_dir() {
  local config_path="$1"
  sed -n -E '/^experiment:/,/^[a-zA-Z_][a-zA-Z0-9_]*:/p' "${config_path}" \
    | grep -m1 -E '^[[:space:]]*output_dir:[[:space:]]*' \
    | sed -E 's/^[[:space:]]*output_dir:[[:space:]]*//' \
    | sed -E 's/[[:space:]]+#.*$//;s/[[:space:]]*$//'
}

# Parse an evaluation spec "<mask_mode>:<ratio>" or "continuous:<N>tr".
parse_experiment() {
  local spec="$1"

  if [[ "${spec}" != *:* ]]; then
    die "Invalid experiment spec: ${spec}. Expected '<mask_mode>:<ratio>' or 'continuous:<N>tr'."
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

# Suffix describing an evaluation mask (used for the inference output dir).
run_suffix() {
  if [[ "${EXP_KIND}" == "fixed_traces" ]]; then
    echo "${EXP_MASK_MODE}_miss${EXP_MISSING_TRACES}tr"
  else
    local pct
    pct="$(ratio_to_pct "${EXP_MASK_RATIO}")"
    echo "${EXP_MASK_MODE}_miss${pct}"
  fi
}

# CLI mask arguments passed to inference_interpolation.py.
infer_mask_args() {
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
  if [[ "${PREFER_BEST_CHECKPOINT}" == "true" && -f "${ckpt_dir}/best.pt" ]]; then
    echo "${ckpt_dir}/best.pt"
    return 0
  fi
  local ckpt
  ckpt="$(ls -t "${ckpt_dir}"/epoch_*.pt 2>/dev/null | head -1 || true)"
  if [[ -n "${ckpt}" ]]; then
    echo "${ckpt}"
    return 0
  fi
  return 1
}

# Train one family config for one seed; echo the absolute experiment directory.
# The saved config.yaml + checkpoints live under <experiment.output_dir>/<name>.
train_family() {
  local config_path="$1"   # repo-relative config path
  local base_name="$2"     # experiment.name parsed from the config
  local suffix="$3"        # training-mask suffix appended by the train script
  local seed="$4"
  local continuous_missing_traces="${5:-}"  # optional fixed contiguous missing count

  local tmpcfg
  tmpcfg="$(mktemp)"
  sed -E \
    -e 's/^([[:space:]]*seed:[[:space:]]*)[0-9]+$/\1'"${seed}"'/' \
    -e 's/^([[:space:]]*name:[[:space:]]*).*/\1'"${base_name}"'_seed'"${seed}"'/' \
    "${REPO_ROOT}/${config_path}" > "${tmpcfg}"

  local out_root
  out_root="$(get_experiment_output_dir "${REPO_ROOT}/${config_path}")"
  [[ -n "${out_root}" ]] || die "Could not parse experiment.output_dir from ${config_path}"
  if [[ "${out_root}" != /* ]]; then
    out_root="${REPO_ROOT}/${out_root}"
  fi

  local exp_dir_name="${base_name}_seed${seed}_${suffix}"
  local exp_dir="${out_root}/${exp_dir_name}"
  local ckpt_dir="${exp_dir}/checkpoints"

  local master_port
  if [[ "${MASTER_PORT}" == "auto" ]]; then
    master_port="$(pick_port)"
  else
    master_port="${MASTER_PORT}"
  fi

  local extra_train_args=()
  if [[ -n "${continuous_missing_traces}" ]]; then
    extra_train_args+=(--continuous-missing-traces "${continuous_missing_traces}")
  fi

  log "Training: ${config_path} -> ${exp_dir_name}"
  run_cmd torchrun ${TORCHRUN_EXTRA} \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${master_port}" \
    "${REPO_ROOT}/${TRAIN_PY}" \
    --config "${tmpcfg}" \
    "${extra_train_args[@]}"

  if [[ "${DRY_RUN}" != "true" ]] && ! find_checkpoint "${ckpt_dir}" > /dev/null; then
    rm -f "${tmpcfg}"
    die "No checkpoint found in ${ckpt_dir}; training failed or produced no checkpoint."
  fi
  rm -f "${tmpcfg}"

  echo "${exp_dir}"
}

# Run one inference on a trained experiment directory with a given eval spec.
run_inference() {
  local exp_dir="$1"       # absolute experiment directory
  local spec="$2"          # eval spec
  local out_suffix="$3"    # output dir suffix
  local kind="$4"
  local mode="$5"
  local value="$6"

  local config_yaml="${exp_dir}/config.yaml"
  local checkpoint
  if [[ "${DRY_RUN}" == "true" ]]; then
    checkpoint="${exp_dir}/checkpoints/best.pt"
  else
    checkpoint="$(find_checkpoint "${exp_dir}/checkpoints")" \
      || die "No checkpoint found in ${exp_dir}/checkpoints."
  fi

  local out_dir="${exp_dir}/inference_${out_suffix}"

  log "Inference: ${exp_dir_name} spec=${spec}"
  run_cmd python "${REPO_ROOT}/${INFER_PY}" \
    --config "${config_yaml}" \
    --checkpoint "${checkpoint}" \
    --output-dir "${out_dir}" \
    $(infer_mask_args "${kind}" "${mode}" "${value}") \
    --device "${INFER_DEVICE}"
}

# ==============================================================================
# Initialization
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES

for f in "${CFUNET_RANDOM_CONFIG}" "${CONTINUOUS_CONFIG}" "${TRAIN_PY}" "${INFER_PY}"; do
  [[ -f "${REPO_ROOT}/${f}" ]] || die "File not found: ${REPO_ROOT}/${f}"
done

NAME_RANDOM="$(get_experiment_name "${REPO_ROOT}/${CFUNET_RANDOM_CONFIG}")"
NAME_CONTINUOUS="$(get_experiment_name "${REPO_ROOT}/${CONTINUOUS_CONFIG}")"
[[ -n "${NAME_RANDOM}" ]] || die "Could not parse experiment.name from ${CFUNET_RANDOM_CONFIG}"
[[ -n "${NAME_CONTINUOUS}" ]] || die "Could not parse experiment.name from ${CONTINUOUS_CONFIG}"

if [[ "${RUN_CONTINUOUS}" == "true" ]]; then
  TOTAL_TRAIN_RUNS=$(( N_SEEDS + ${#CONTINUOUS_EVAL[@]} * N_SEEDS ))
else
  TOTAL_TRAIN_RUNS=$(( N_SEEDS ))
fi
run_idx=0

log "Repository root: ${REPO_ROOT}"
log "cfunet_random config: ${CFUNET_RANDOM_CONFIG} (${NAME_RANDOM})"
log "continuous config    : ${CONTINUOUS_CONFIG} (${NAME_CONTINUOUS})"
log "Seeds: ${N_SEEDS} (start ${START_SEED})"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ==============================================================================
# Main loop
# ==============================================================================

# ---- Family A: cfunet_random (all seeds; serves random + uniform eval) ----
for ((i = 0; i < N_SEEDS; i++)); do
  seed=$((START_SEED + i))

  run_idx=$((run_idx + 1))
  log "================================================================"
  log "[train ${run_idx}/${TOTAL_TRAIN_RUNS}] cfunet_random seed=${seed}"
  log "================================================================"

  exp_random="$(train_family "${CFUNET_RANDOM_CONFIG}" "${NAME_RANDOM}" "${CFUNET_RANDOM_SUFFIX}" "${seed}")"

  if [[ "${RUN_INFERENCE}" == "true" || "${RUN_EXTRA_INFERENCE}" == "true" ]]; then
    for spec in "${RANDOM_UNIFORM_EVAL[@]}"; do
      parse_experiment "${spec}"
      suffix="$(run_suffix)"

      if [[ "${RUN_INFERENCE}" == "true" ]]; then
        run_inference "${exp_random}" "${spec}" "${suffix}" \
          "${EXP_KIND}" "${EXP_MASK_MODE}" "${EXP_MASK_RATIO}"
      fi

      if [[ "${RUN_EXTRA_INFERENCE}" == "true" ]]; then
        if [[ "${EXP_KIND}" == "fixed_traces" ]]; then
          extra_kind="fixed_traces"
          extra_value=$((EXP_MISSING_TRACES + EXTRA_TRACES_STEP))
          extra_suffix="miss${extra_value}tr"
        else
          extra_kind="ratio"
          extra_value="$(ratio_add "${EXP_MASK_RATIO}" "${EXTRA_RATIO_STEP}")"
          extra_suffix="${EXP_MASK_MODE}$(ratio_to_pct "${extra_value}")"
        fi
        run_inference "${exp_random}" "${spec}" "${extra_suffix}" \
          "${extra_kind}" "${EXP_MASK_MODE}" "${extra_value}"
      fi
    done
  fi

  echo ""
done

# ---- Family B: continuous (one model per level; training mask == inference) ----
if [[ "${RUN_CONTINUOUS}" == "true" ]]; then
  for spec in "${CONTINUOUS_EVAL[@]}"; do
    parse_experiment "${spec}"
    level_traces="${EXP_MISSING_TRACES}"
    level_suffix="$(run_suffix)"

    for ((i = 0; i < N_SEEDS; i++)); do
      seed=$((START_SEED + i))

      run_idx=$((run_idx + 1))
      log "================================================================"
      log "[train ${run_idx}/${TOTAL_TRAIN_RUNS}] continuous:${level_traces}tr seed=${seed}"
      log "================================================================"

      exp_continuous="$(train_family "${CONTINUOUS_CONFIG}" "${NAME_CONTINUOUS}" "${level_suffix}" "${seed}" "${level_traces}")"

      if [[ "${RUN_INFERENCE}" == "true" || "${RUN_EXTRA_INFERENCE}" == "true" ]]; then
        if [[ "${RUN_INFERENCE}" == "true" ]]; then
          run_inference "${exp_continuous}" "${spec}" "${level_suffix}" \
            "${EXP_KIND}" "${EXP_MASK_MODE}" "${level_traces}"
        fi

        if [[ "${RUN_EXTRA_INFERENCE}" == "true" ]]; then
          extra_kind="fixed_traces"
          extra_value=$((level_traces + EXTRA_TRACES_STEP))
          extra_suffix="miss${extra_value}tr"
          run_inference "${exp_continuous}" "${spec}" "${extra_suffix}" \
            "${extra_kind}" "${EXP_MASK_MODE}" "${extra_value}"
        fi
      fi

      echo ""
    done
  done
fi

log "All ${TOTAL_TRAIN_RUNS} training runs complete."
