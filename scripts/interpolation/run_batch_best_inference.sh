#!/usr/bin/env bash
# Batch inference for every trained experiment using checkpoints/best.pt.
#
# Scans the results tree for experiments with checkpoints/best.pt, picks the
# inference script from the config model type (gated transformers use
# inference_interpolation_transformer.py, everything else
# inference_interpolation.py), and writes fresh outputs (metrics CSVs, all-trace
# npy volumes via --save-npy) into a collection directory outside results/.
# The saved config.yaml of each experiment already contains the mask settings
# used at training time, so no mask CLI args are passed.
#
# Resumable: experiments whose output dir carries an INFERENCE_SUCCESS marker
# are skipped.  Set FORCE=true to redo them.
#
# Usage:
#   bash scripts/interpolation/run_batch_best_inference.sh [RESULTS_ROOT] [COLLECT_ROOT]
#
# Env:
#   INFER_DEVICE  GPU device for inference (default cuda:0).

set -uo pipefail

RESULTS_ROOT="${1:-${RESULTS_ROOT:-/cloud/cloud-s3fs}}"
COLLECT_ROOT="${2:-${COLLECT_ROOT:-collected}}"
INFER_DEVICE="${INFER_DEVICE:-cuda:0}"
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

RESULTS_ROOT_ABS="$(abs_path "${RESULTS_ROOT}")"
COLLECT_ROOT_ABS="$(abs_path "${COLLECT_ROOT}")"

if [[ ! -d "${RESULTS_ROOT_ABS}" ]]; then
    echo "ERROR: results root not found: ${RESULTS_ROOT_ABS}" >&2
    exit 1
fi

# Guard against re-scanning the collection output if it lives under results/.
if [[ "${COLLECT_ROOT_ABS}" == "${RESULTS_ROOT_ABS}"* ]]; then
    echo "ERROR: collection root must not live inside the results root:" >&2
    echo "  results : ${RESULTS_ROOT_ABS}" >&2
    echo "  collect : ${COLLECT_ROOT_ABS}" >&2
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
echo "Batch best.pt inference" | tee -a "${BATCH_LOG}"
echo "Repository : ${REPO_ROOT}" | tee -a "${BATCH_LOG}"
echo "Results    : ${RESULTS_ROOT_ABS}" | tee -a "${BATCH_LOG}"
echo "Collection : ${COLLECT_ROOT_ABS}" | tee -a "${BATCH_LOG}"
echo "Device     : ${INFER_DEVICE}" | tee -a "${BATCH_LOG}"
echo "Force      : ${FORCE}" | tee -a "${BATCH_LOG}"
echo "Log        : ${BATCH_LOG}" | tee -a "${BATCH_LOG}"
echo "============================================================" | tee -a "${BATCH_LOG}"

# Enumerate experiments that have checkpoints/best.pt, sorted by name.
mapfile -t EXPERIMENT_DIRS < <(
    find "${RESULTS_ROOT_ABS}" \
        -mindepth 3 \
        -maxdepth 3 \
        -type f \
        -path "*/checkpoints/best.pt" \
        -printf "%h\n" |
        sed 's|/checkpoints$||' |
        sort
)

if [[ ${#EXPERIMENT_DIRS[@]} -eq 0 ]]; then
    log "No experiment directories with checkpoints/best.pt found."
    exit 1
fi

log "Found ${#EXPERIMENT_DIRS[@]} experiments with best.pt."
log ""

for EXPERIMENT_DIR in "${EXPERIMENT_DIRS[@]}"; do
    NAME="$(basename "${EXPERIMENT_DIR}")"
    CONFIG="${EXPERIMENT_DIR}/config.yaml"
    CHECKPOINT="${EXPERIMENT_DIR}/checkpoints/best.pt"
    OUT_DIR="${COLLECT_ROOT_ABS}/${NAME}/inference"
    RUN_LOG="${COLLECT_ROOT_ABS}/${NAME}/inference.log"
    SUCCESS_MARKER="${OUT_DIR}/INFERENCE_SUCCESS"

    log "------------------------------------------------------------"
    log "${NAME}"
    log "Config     : ${CONFIG}"
    log "Checkpoint : ${CHECKPOINT}"
    log "Output     : ${OUT_DIR}"

    if [[ -f "${SUCCESS_MARKER}" && "${FORCE}" != "true" ]]; then
        log "SKIPPED: INFERENCE_SUCCESS marker exists."
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        SKIPPED_EXPERIMENTS+=("${NAME}: already successful")
        continue
    fi

    # Pick the inference script from the config model type.
    if ! MODEL_TYPE="$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['model']['type'])" "${CONFIG}" 2>/dev/null)"; then
        log "FAILED: could not read model.type from config.yaml."
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
        --checkpoint "${CHECKPOINT}"
        --output-dir "${OUT_DIR}"
        --save-npy
        --device "${INFER_DEVICE}"
    )

    # The UNet-script saves per-shot viz npy arrays by default; the requested
    # all-trace volumes come from --save-npy, so skip the viz arrays.
    if [[ "${INFER_PY}" == *"inference_interpolation.py" ]]; then
        args+=(--no-save-viz-npy)
    fi

    # Reproducibility: pin the seed parsed from the experiment name (matches
    # the seed already baked into the saved config.yaml).
    if [[ "${NAME}" =~ _seed([0-9]+)_ ]]; then
        args+=(--seed "${BASH_REMATCH[1]}")
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

    if python "${REPO_ROOT}/${INFER_PY}" "${args[@]}" 2>&1 | tee -a "${RUN_LOG}" "${BATCH_LOG}"; then
        END_TIME="$(date -Iseconds)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        {
            echo "${END_TIME}"
            echo "${NAME}"
            echo "${CHECKPOINT}"
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
log "Total   : ${#EXPERIMENT_DIRS[@]}"
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
