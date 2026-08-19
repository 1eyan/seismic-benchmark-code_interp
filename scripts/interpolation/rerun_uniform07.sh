#!/usr/bin/env bash
# Batch re-run of uniform:0.7 training + inference for the models affected by
# the uniform-mask stride bug (uniform:0.7 was silently degraded to ~50%).
#
# Each model is driven through its existing train+infer loop script with the
# experiment list narrowed to "uniform:0.7". cfunet runs separately because it
# uses the family-batched loop script (cfunet_random family serves the uniform
# evaluation; the continuous family is skipped).
#
# Usage:
#   bash scripts/interpolation/rerun_uniform07.sh
#
# Env (all optional):
#   DRY_RUN=true                 print commands without running
#   MODELS="a b c"               run only a subset (see MODEL_KEYS below)
#   CUDA_VISIBLE_DEVICES         forwarded to each loop script
#   INFER_DEVICE                 forwarded to each loop script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DRY_RUN="${DRY_RUN:-false}"

log() {
  echo "[$(date -Iseconds)] $*"
}

# Each entry drives one model. The driver function maps the key to the correct
# loop script + config + overrides.
MODEL_KEYS=(
  #transformer_v9
  #chai2020_unet_paper
  liu2022_wrdl_conservative
  li2022_caunet_seg_c3_paper
  yu2022_anet_seg_c3_paper
  #park2022_cfunet_paper
)

if [[ -n "${MODELS:-}" ]]; then
  read -r -a MODEL_KEYS <<< "${MODELS}"
fi

run_loop() {
  local key="$1"
  shift
  log "=================================================================="
  log "Model: ${key}"
  log "Driver: $*"
  log "=================================================================="
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "+ DRY_RUN $*"
  else
    "$@"
  fi
  echo ""
}

for key in "${MODEL_KEYS[@]}"; do
  case "${key}" in
    transformer_v9)
      run_loop "${key}" \
        env EXPERIMENTS_OVERRIDE="uniform:0.7" \
          PIPELINE=gated \
          BASE_CONFIG="configs/interpolation/interpolation_transformer.yaml" \
          bash "${SCRIPT_DIR}/train_infer_loop_transformer.sh"
      ;;
    chai2020_unet_paper)
      run_loop "${key}" \
        env EXPERIMENTS_OVERRIDE="uniform:0.7" \
          BASE_CONFIG="configs/interpolation/chai2020_unet_paper.yaml" \
          bash "${SCRIPT_DIR}/train_infer_loop.sh"
      ;;
    liu2022_wrdl_conservative)
      run_loop "${key}" \
        env EXPERIMENTS_OVERRIDE="uniform:0.7" \
          BASE_CONFIG="configs/interpolation/liu2022_wrdl_conservative.yaml" \
          bash "${SCRIPT_DIR}/train_infer_loop.sh"
      ;;
    li2022_caunet_seg_c3_paper)
      run_loop "${key}" \
        env EXPERIMENTS_OVERRIDE="uniform:0.7" \
          BASE_CONFIG="configs/interpolation/li2022_caunet_seg_c3_paper.yaml" \
          bash "${SCRIPT_DIR}/train_infer_loop.sh"
      ;;
    yu2022_anet_seg_c3_paper)
      run_loop "${key}" \
        env EXPERIMENTS_OVERRIDE="uniform:0.7" \
          BASE_CONFIG="configs/interpolation/yu2022_anet_seg_c3_paper.yaml" \
          bash "${SCRIPT_DIR}/train_infer_loop.sh"
      ;;
    park2022_cfunet_paper)
      # cfunet runs separately: only the cfunet_random family (serves the
      # uniform eval), continuous family disabled.
      run_loop "${key}" \
        env RANDOM_UNIFORM_EVAL_OVERRIDE="uniform:0.7" \
          RUN_CONTINUOUS=false \
          CFUNET_RANDOM_CONFIG="configs/interpolation/park2022_cfunet_paper.yaml" \
          bash "${SCRIPT_DIR}/train_infer_loop_cfunet.sh"
      ;;
    *)
      log "ERROR: unknown model key '${key}'. Valid keys: ${MODEL_KEYS[*]}"
      exit 1
      ;;
  esac
done

log "All requested uniform:0.7 re-runs dispatched."
