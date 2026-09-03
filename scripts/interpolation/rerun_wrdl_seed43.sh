#!/usr/bin/env bash
# Retrain the two WRDL seed-43 scenarios that collapsed in the benchmark.
#
#   - continuous_miss40tr
#   - random_miss30
#
# Uses train_interpolation_unet.py directly with the --seed override added for
# direct retraining. After training, runs inference on the same scenario.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/interpolation/rerun_wrdl_seed43.sh
#
# To train only, set RUN_INFERENCE=false:
#   RUN_INFERENCE=false bash scripts/interpolation/rerun_wrdl_seed43.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-auto}"
TORCHRUN_EXTRA="${TORCHRUN_EXTRA:-}"
INFER_DEVICE="${INFER_DEVICE:-cuda:1}"
BASE_CONFIG="${BASE_CONFIG:-configs/interpolation/liu2022_wrdl_conservative.yaml}"
RUN_INFERENCE="${RUN_INFERENCE:-true}"
PREFER_BEST_CHECKPOINT="${PREFER_BEST_CHECKPOINT:-true}"

export CUDA_VISIBLE_DEVICES

NAME_BASE="interp_liu2022_wrdl_conservative"
SEED=43

pick_port() {
  python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

run_training() {
  local mask_mode="$1"
  local arg_name="$2"
  local arg_value="$3"
  local suffix="$4"
  local run_name="${NAME_BASE}_seed${SEED}_${suffix}"
  local master_port

  if [[ "${MASTER_PORT}" == "auto" ]]; then
    master_port="$(pick_port)"
  else
    master_port="${MASTER_PORT}"
  fi

  echo "================================================================"
  echo "Training ${run_name} (GPUs=${CUDA_VISIBLE_DEVICES}, nproc=${NPROC_PER_NODE})"
  echo "================================================================"

  torchrun ${TORCHRUN_EXTRA} \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${master_port}" \
    scripts/interpolation/train_interpolation_unet.py \
    --config "${BASE_CONFIG}" \
    --seed "${SEED}" \
    --mask-mode "${mask_mode}" \
    "${arg_name}" "${arg_value}"
}

run_inference() {
  local mask_mode="$1"
  local arg_name="$2"
  local arg_value="$3"
  local suffix="$4"
  local run_name="${NAME_BASE}_seed${SEED}_${suffix}"
  local ckpt_dir="results/${run_name}/checkpoints"
  local ckpt

  if [[ "${PREFER_BEST_CHECKPOINT}" == "true" && -f "${ckpt_dir}/best.pt" ]]; then
    ckpt="${ckpt_dir}/best.pt"
  else
    ckpt="$(ls -t "${ckpt_dir}"/epoch_*.pt 2>/dev/null | head -1 || true)"
  fi

  if [[ -z "${ckpt}" ]]; then
    echo "WARNING: no checkpoint found for ${run_name}; skipping inference." >&2
    return 0
  fi

  echo "================================================================"
  echo "Inference ${run_name}"
  echo "================================================================"

  python scripts/interpolation/inference_interpolation.py \
    --config "${BASE_CONFIG}" \
    --checkpoint "${ckpt}" \
    --output-dir "results/${run_name}/inference" \
    --seed "${SEED}" \
    --mask-mode "${mask_mode}" \
    --device "${INFER_DEVICE}" \
    "${arg_name}" "${arg_value}"
}

# continuous_miss40tr
run_training continuous "--continuous-missing-traces" 40 "continuous_miss40tr"
if [[ "${RUN_INFERENCE}" == "true" ]]; then
  run_inference continuous "--continuous-missing-traces" 40 "continuous_miss40tr"
fi

# random_miss30
run_training random "--mask-ratio" 0.3 "random_miss30"
if [[ "${RUN_INFERENCE}" == "true" ]]; then
  run_inference random "--mask-ratio" 0.3 "random_miss30"
fi

echo "WRDL seed-${SEED} retraining complete."
