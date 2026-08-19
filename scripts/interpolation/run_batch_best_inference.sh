#!/usr/bin/env bash
# Batch best.pt inference sourced from the centralized collected/ tree.
#
# Iterates over collected/configs/*.yaml (one YAML per experiment/seed) and runs
# inference for each using its matching checkpoint collected/params/<name>.pt.
# Outputs are written to collected/<name>/inference/ (metrics CSVs, all-trace
# npy volumes via --save-npy).  A downstream aggregator
# (scripts/interpolation/fill_batch_evaluation_xlsx.py) groups experiments by
# setting (seed stripped from the name) and computes mean±std across seeds.
#
# The saved config already carries the mask settings used at training time, so
# no mask CLI args are passed; the seed embedded in the name is passed to
# --seed for reproducibility (matches the seed baked into the config).
#
# The inference script is picked from the config model type (gated transformers
# use inference_interpolation_transformer.py, everything else
# inference_interpolation.py).
#
# Resumable: an output dir carrying an INFERENCE_SUCCESS marker is skipped.
# Set FORCE=true to redo them.
#
# Usage:
#   bash scripts/interpolation/run_batch_best_inference.sh [CONFIG_DIR] [PARAMS_DIR] [COLLECT_ROOT] [DEVICE]
#
# Env:
#   INFER_DEVICE  GPU device for inference (default cuda:0); also settable via
#                 positional arg 4.
#   FORCE         "true" to rerun experiments with an INFERENCE_SUCCESS marker.

set -uo pipefail

CONFIG_DIR="${1:-${CONFIG_DIR:-collected/configs}}"
PARAMS_DIR="${2:-${PARAMS_DIR:-collected/params}}"
COLLECT_ROOT="${3:-${COLLECT_ROOT:-collected}}"
INFER_DEVICE="${4:-${INFER_DEVICE:-cuda:1}}"
FORCE="${FORCE:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

abs_path() {
    local p="$1"
    if [[ "${p}" = /* ]]; then
        realpath -m "${p}"
    else
        realpath -m "${REPO_ROOT}/${p}"
    fi
}

CONFIG_DIR_ABS="$(abs_path "${CONFIG_DIR}")"
PARAMS_DIR_ABS="$(abs_path "${PARAMS_DIR}")"
COLLECT_ROOT_ABS="$(abs_path "${COLLECT_ROOT}")"

if [[ ! -d "${CONFIG_DIR_ABS}" ]]; then
    echo "ERROR: config dir not found: ${CONFIG_DIR_ABS}" >&2
    exit 1
fi
if [[ ! -d "${PARAMS_DIR_ABS}" ]]; then
    echo "ERROR: params dir not found: ${PARAMS_DIR_ABS}" >&2
    exit 1
fi

mkdir -p "${COLLECT_ROOT_ABS}"

BATCH_TIME="$(date '+%Y%m%d_%H%M%S')"
BATCH_LOG="${COLLECT_ROOT_ABS}/batch_inference_${BATCH_TIME}.log"

SUCCESS_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0
FAILED_EXPERIMENTS=()
SKIPPED_EXPERIMENTS=()

log() {
    echo "$@" | tee -a "${BATCH_LOG}"
}

echo "============================================================" | tee "${BATCH_LOG}"
echo "Batch best.pt inference (collected source)" | tee -a "${BATCH_LOG}"
echo "Repository : ${REPO_ROOT}" | tee -a "${BATCH_LOG}"
echo "Configs    : ${CONFIG_DIR_ABS}" | tee -a "${BATCH_LOG}"
echo "Params     : ${PARAMS_DIR_ABS}" | tee -a "${BATCH_LOG}"
echo "Collection : ${COLLECT_ROOT_ABS}" | tee -a "${BATCH_LOG}"
echo "Device     : ${INFER_DEVICE}" | tee -a "${BATCH_LOG}"
echo "Force      : ${FORCE}" | tee -a "${BATCH_LOG}"
echo "Log        : ${BATCH_LOG}" | tee -a "${BATCH_LOG}"
echo "============================================================" | tee -a "${BATCH_LOG}"

# Enumerate inference jobs: one per config that has a matching checkpoint.
JOBS=()
for CONFIG in "${CONFIG_DIR_ABS}"/*.yaml; do
    [[ -e "${CONFIG}" ]] || continue
    NAME="$(basename "${CONFIG}" .yaml)"
    CKPT="${PARAMS_DIR_ABS}/${NAME}.pt"
    if [[ -f "${CKPT}" ]]; then
        JOBS+=("${CONFIG}|${NAME}|${CKPT}")
    fi
done

if [[ ${#JOBS[@]} -eq 0 ]]; then
    log "No inference jobs: no checkpoints found for configs in ${CONFIG_DIR_ABS}."
    exit 1
fi

log "Enqueued ${#JOBS[@]} inference job(s)."
log ""

for JOB in "${JOBS[@]}"; do
    IFS='|' read -r CONFIG NAME CKPT <<< "${JOB}"
    OUT_DIR="${COLLECT_ROOT_ABS}/${NAME}/inference"
    RUN_LOG="${COLLECT_ROOT_ABS}/${NAME}/inference.log"
    SUCCESS_MARKER="${OUT_DIR}/INFERENCE_SUCCESS"

    SEED="$(printf '%s' "${NAME}" | sed -nE 's/.*_seed([0-9]+)_.*/\1/p')"

    log "------------------------------------------------------------"
    log "${NAME}"
    log "Config     : ${CONFIG}"
    log "Checkpoint : ${CKPT}"
    log "Output     : ${OUT_DIR}"

    if [[ -f "${SUCCESS_MARKER}" && "${FORCE}" != "true" ]]; then
        log "SKIPPED: INFERENCE_SUCCESS marker exists."
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        SKIPPED_EXPERIMENTS+=("${NAME}: already successful")
        continue
    fi

    # Pick the inference script from the config model type.
    if ! MODEL_TYPE="$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['model']['type'])" "${CONFIG}" 2>/dev/null)"; then
        log "FAILED: could not read model.type from config."
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_EXPERIMENTS+=("${NAME}: unreadable config.yaml")
        mkdir -p "${OUT_DIR}"
        {
            echo "$(date -Iseconds)"
            echo "${NAME}"
            echo "exit_code=config_parse_error"
        } > "${OUT_DIR}/INFERENCE_FAILED"
        continue
    fi

    case "${MODEL_TYPE}" in
        gated_transformer_v9 | gated_transformer_v9_encdec)
            INFER_PY="scripts/interpolation/inference_interpolation_transformer.py"
            ;;
        *)
            INFER_PY="scripts/interpolation/inference_interpolation.py"
            ;;
    esac

    args=(
        --config "${CONFIG}"
        --checkpoint "${CKPT}"
        --output-dir "${OUT_DIR}"
        --save-npy
        --device "${INFER_DEVICE}"
    )
    if [[ -n "${SEED}" ]]; then
        args+=(--seed "${SEED}")
    fi

    # The UNet-script saves per-shot viz npy arrays by default; the requested
    # all-trace volumes come from --save-npy, so skip the viz arrays.
    if [[ "${INFER_PY}" == *"inference_interpolation.py" ]]; then
        args+=(--no-save-viz-npy)
    fi

    mkdir -p "${OUT_DIR}"
    rm -f "${OUT_DIR}/INFERENCE_FAILED"

    START_TIME="$(date -Iseconds)"
    {
        echo
        echo "============================================================"
        echo "[${START_TIME}] Starting ${NAME}"
        echo "Model type : ${MODEL_TYPE}"
        echo "Command    : python ${REPO_ROOT}/${INFER_PY} ${args[*]}"
    } | tee -a "${BATCH_LOG}" "${RUN_LOG}"

    cd "${REPO_ROOT}"

    if python -u "${REPO_ROOT}/${INFER_PY}" "${args[@]}" 2>&1 | tee -a "${RUN_LOG}" "${BATCH_LOG}"; then
        END_TIME="$(date -Iseconds)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        {
            echo "${END_TIME}"
            echo "${NAME}"
            echo "${CKPT}"
        } > "${SUCCESS_MARKER}"
        log "[${END_TIME}] SUCCESS: ${NAME}"
    else
        EXIT_CODE="${PIPESTATUS[0]}"
        END_TIME="$(date -Iseconds)"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_EXPERIMENTS+=("${NAME}: exit code ${EXIT_CODE}")
        {
            echo "${END_TIME}"
            echo "${NAME}"
            echo "exit_code=${EXIT_CODE}"
        } > "${OUT_DIR}/INFERENCE_FAILED"
        log "[${END_TIME}] FAILED: ${NAME}, exit code ${EXIT_CODE}"
    fi

    log ""
done

log "============================================================"
log "Batch inference finished: $(date -Iseconds)"
log "Total   : ${#JOBS[@]}"
log "Success : ${SUCCESS_COUNT}"
log "Failed  : ${FAILED_COUNT}"
log "Skipped : ${SKIPPED_COUNT}"
log "Log     : ${BATCH_LOG}"

if [[ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]]; then
    log ""
    log "Failed experiments:"
    for ITEM in "${FAILED_EXPERIMENTS[@]}"; do
        log "  ${ITEM}"
    done
fi

if [[ ${#SKIPPED_EXPERIMENTS[@]} -gt 0 ]]; then
    log ""
    log "Skipped experiments:"
    for ITEM in "${SKIPPED_EXPERIMENTS[@]}"; do
        log "  ${ITEM}"
    done
fi

log "============================================================"

if [[ ${FAILED_COUNT} -gt 0 ]]; then
    exit 1
fi
