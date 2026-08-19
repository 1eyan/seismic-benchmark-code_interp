#!/usr/bin/env bash
# Centralize every trained experiment's best.pt and config.yaml into flat
# params/ + configs/ directories under the collection root, then remove
# epoch_*.pt so the original checkpoints/ keeps only best.pt.
#
# Hardlinks are used (no extra disk usage); the original paths keep working.
#
# Usage:
#   bash scripts/interpolation/centralize_best_params.sh [RESULTS_ROOT] [COLLECT_ROOT]
#
# Env:
#   DRY_RUN=true     print actions without executing them
#   KEEP_EPOCHS=true skip the epoch_*.pt cleanup

set -uo pipefail

RESULTS_ROOT="${1:-${RESULTS_ROOT:-results}}"
COLLECT_ROOT="${2:-${COLLECT_ROOT:-collected}}"
DRY_RUN="${DRY_RUN:-false}"
KEEP_EPOCHS="${KEEP_EPOCHS:-false}"

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
PARAMS_DIR="${COLLECT_ROOT_ABS}/params"
CONFIGS_DIR="${COLLECT_ROOT_ABS}/configs"

if [[ ! -d "${RESULTS_ROOT_ABS}" ]]; then
    echo "ERROR: results root not found: ${RESULTS_ROOT_ABS}" >&2
    exit 1
fi

if [[ "${COLLECT_ROOT_ABS}" == "${RESULTS_ROOT_ABS}"* ]]; then
    echo "ERROR: collection root must not live inside the results root:" >&2
    echo "  results : ${RESULTS_ROOT_ABS}" >&2
    echo "  collect : ${COLLECT_ROOT_ABS}" >&2
    exit 1
fi

if [[ "${DRY_RUN}" != "true" ]]; then
    mkdir -p "${PARAMS_DIR}" "${CONFIGS_DIR}"
fi

run() {
    echo "+ $*"
    if [[ "${DRY_RUN}" != "true" ]]; then
        "$@"
    fi
}

echo "Results    : ${RESULTS_ROOT_ABS}"
echo "Collection : ${COLLECT_ROOT_ABS}"
echo "Dry run    : ${DRY_RUN}"
echo ""

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
    echo "No experiment directories with checkpoints/best.pt found."
    exit 1
fi

PT_LINKED=0
YAML_LINKED=0
ERRORS=0

for EXPERIMENT_DIR in "${EXPERIMENT_DIRS[@]}"; do
    NAME="$(basename "${EXPERIMENT_DIR}")"
    SRC_PT="${EXPERIMENT_DIR}/checkpoints/best.pt"
    SRC_YAML="${EXPERIMENT_DIR}/config.yaml"
    DST_PT="${PARAMS_DIR}/${NAME}.pt"
    DST_YAML="${CONFIGS_DIR}/${NAME}.yaml"

    if [[ ! -e "${DST_PT}" ]]; then
        run ln "${SRC_PT}" "${DST_PT}"
        if [[ "${DRY_RUN}" != "true" ]]; then
            if [[ ! -e "${DST_PT}" ]]; then
                # Cross-device fallback.
                run cp "${SRC_PT}" "${DST_PT}"
            fi
            if [[ ! -e "${DST_PT}" ]]; then
                echo "ERROR: failed to link ${SRC_PT}" >&2
                ERRORS=$((ERRORS + 1))
                continue
            fi
        fi
        PT_LINKED=$((PT_LINKED + 1))
    fi

    if [[ ! -e "${DST_YAML}" ]]; then
        if [[ ! -f "${SRC_YAML}" ]]; then
            echo "WARNING: ${NAME}: config.yaml missing, skipped." >&2
        else
            run ln "${SRC_YAML}" "${DST_YAML}"
            if [[ "${DRY_RUN}" != "true" ]]; then
                if [[ ! -e "${DST_YAML}" ]]; then
                    run cp "${SRC_YAML}" "${DST_YAML}"
                fi
                if [[ ! -e "${DST_YAML}" ]]; then
                    echo "ERROR: failed to link ${SRC_YAML}" >&2
                    ERRORS=$((ERRORS + 1))
                else
                    YAML_LINKED=$((YAML_LINKED + 1))
                fi
            else
                YAML_LINKED=$((YAML_LINKED + 1))
            fi
        fi
    fi
done

echo ""
echo "Experiments processed : ${#EXPERIMENT_DIRS[@]}"
echo "Params linked         : ${PT_LINKED} -> ${PARAMS_DIR}"
echo "Configs linked        : ${YAML_LINKED} -> ${CONFIGS_DIR}"

# ---------------------------------------------------------------------------
# Cleanup: keep only best.pt in the original checkpoints/ directories.
# ---------------------------------------------------------------------------
mapfile -t EPOCH_FILES < <(
    find "${RESULTS_ROOT_ABS}" -name "epoch_*.pt" -type f | sort
)

if [[ ${#EPOCH_FILES[@]} -gt 0 ]]; then
    EPOCH_SIZE="$(du -ch "${EPOCH_FILES[@]}" 2>/dev/null | tail -1 | awk '{print $1}')"
    echo ""
    echo "epoch_*.pt found : ${#EPOCH_FILES[@]} files (${EPOCH_SIZE})"
    if [[ "${KEEP_EPOCHS}" == "true" ]]; then
        echo "KEEP_EPOCHS=true: epoch files left untouched."
    else
        for F in "${EPOCH_FILES[@]}"; do
            run rm -f "${F}"
        done
        echo "Deleted ${#EPOCH_FILES[@]} epoch files; original checkpoints/ keeps best.pt only."
    fi
else
    echo ""
    echo "No epoch_*.pt files found; original checkpoints/ already keep best.pt only."
fi

if [[ ${ERRORS} -gt 0 ]]; then
    echo ""
    echo "Finished with ${ERRORS} error(s)." >&2
    exit 1
fi
