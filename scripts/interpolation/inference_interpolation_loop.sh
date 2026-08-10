#!/usr/bin/env bash
# Batch interpolation inference wrapper.
# Automatically scans experiment directories and reruns inference sequentially.

set -uo pipefail

# ==================== 配置区 ====================

# 使用的 GPU
export CUDA_VISIBLE_DEVICES="0"

# 实验结果根目录
RESULTS_ROOT="../../../cloud/cloud-s3fs/0614_results"

# 重新推理结果保存根目录
OUTPUT_ROOT="/home/reinference"

# 权重文件相对于每个实验目录的位置
CHECKPOINT_RELATIVE_PATH="checkpoints/best.pt"

# 为空时，由每个实验的 config.inference 读取
N_VIZ_SHOTS=""

# 为空时，由 config.inference.device 或 experiment.device 读取
DEVICE=""

# 输出目录已经存在时的处理方式：
# false：跳过该实验，适合断点续跑
# true ：重新运行，并写入原输出目录
OVERWRITE_EXISTING="false"

# 是否在某个实验失败后立即停止：
# false：记录失败并继续推理其他实验
# true ：遇到失败立即停止
STOP_ON_ERROR="false"

# 只处理匹配该表达式的文件夹
EXPERIMENT_PATTERN="interp_*"

# ===============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PY_SCRIPT="${REPO_ROOT}/scripts/interpolation/inference_interpolation.py"

if [[ ! -f "${PY_SCRIPT}" ]]; then
    echo "Python inference script not found:"
    echo "  ${PY_SCRIPT}" >&2
    exit 1
fi

# 将结果目录转换为绝对路径
if [[ "${RESULTS_ROOT}" = /* ]]; then
    RESULTS_ROOT_ABS="${RESULTS_ROOT}"
else
    RESULTS_ROOT_ABS="$(realpath -m "${REPO_ROOT}/${RESULTS_ROOT}")"
fi

if [[ ! -d "${RESULTS_ROOT_ABS}" ]]; then
    echo "Results directory not found:"
    echo "  ${RESULTS_ROOT_ABS}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

# 保存本次批量推理的总日志
BATCH_TIME="$(date '+%Y%m%d_%H%M%S')"
BATCH_LOG="${OUTPUT_ROOT}/batch_inference_${BATCH_TIME}.log"

SUCCESS_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0
TOTAL_COUNT=0

FAILED_EXPERIMENTS=()
SKIPPED_EXPERIMENTS=()

echo "============================================================"
echo "Batch interpolation inference"
echo "Repository : ${REPO_ROOT}"
echo "Experiments: ${RESULTS_ROOT_ABS}"
echo "Output root: ${OUTPUT_ROOT}"
echo "GPU        : ${CUDA_VISIBLE_DEVICES}"
echo "Log        : ${BATCH_LOG}"
echo "============================================================"

# 获取实验目录并按照名称排序
mapfile -t EXPERIMENT_DIRS < <(
    find "${RESULTS_ROOT_ABS}" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -name "${EXPERIMENT_PATTERN}" \
        -print | sort
)

if [[ ${#EXPERIMENT_DIRS[@]} -eq 0 ]]; then
    echo "No experiment directories found."
    exit 1
fi

echo "Found ${#EXPERIMENT_DIRS[@]} experiment directories."
echo

for EXPERIMENT_DIR in "${EXPERIMENT_DIRS[@]}"; do
    TOTAL_COUNT=$((TOTAL_COUNT + 1))

    EXPERIMENT_NAME="$(basename "${EXPERIMENT_DIR}")"
    CONFIG="${EXPERIMENT_DIR}/config.yaml"
    CHECKPOINT="${EXPERIMENT_DIR}/${CHECKPOINT_RELATIVE_PATH}"

    # 从实验名称提取模型名称
    # interp_atten_unet_base_... -> atten_unet
    # interp_res_unet_base_...   -> res_unet
    if [[ "${EXPERIMENT_NAME}" =~ ^interp_(.+)_base_seed[0-9]+_ ]]; then
        MODEL_NAME="${BASH_REMATCH[1]}"
    else
        MODEL_NAME="other"
    fi

    # 从实验名称提取随机种子
    if [[ "${EXPERIMENT_NAME}" =~ _seed([0-9]+)_ ]]; then
        SEED="${BASH_REMATCH[1]}"
    else
        SEED=""
    fi

    OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/${EXPERIMENT_NAME}"
    RUN_LOG="${OUTPUT_DIR}/inference.log"

    echo "------------------------------------------------------------"
    echo "[${TOTAL_COUNT}/${#EXPERIMENT_DIRS[@]}] ${EXPERIMENT_NAME}"
    echo "Model      : ${MODEL_NAME}"
    echo "Seed       : ${SEED:-config default}"
    echo "Config     : ${CONFIG}"
    echo "Checkpoint : ${CHECKPOINT}"
    echo "Output     : ${OUTPUT_DIR}"

    if [[ ! -f "${CONFIG}" ]]; then
        echo "SKIPPED: config.yaml not found." | tee -a "${BATCH_LOG}"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        SKIPPED_EXPERIMENTS+=("${EXPERIMENT_NAME}: missing config.yaml")
        continue
    fi

    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "SKIPPED: checkpoint not found." | tee -a "${BATCH_LOG}"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        SKIPPED_EXPERIMENTS+=("${EXPERIMENT_NAME}: missing ${CHECKPOINT_RELATIVE_PATH}")
        continue
    fi

    if [[ -d "${OUTPUT_DIR}" && "${OVERWRITE_EXISTING}" != "true" ]]; then
        echo "SKIPPED: output directory already exists." | tee -a "${BATCH_LOG}"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        SKIPPED_EXPERIMENTS+=("${EXPERIMENT_NAME}: output exists")
        continue
    fi

    mkdir -p "${OUTPUT_DIR}"
    # 保存本次推理所使用的配置和网络权重
    mkdir -p "${OUTPUT_DIR}/checkpoint"

    cp -f "${CONFIG}" "${OUTPUT_DIR}/config.yaml"
    cp -f "${CHECKPOINT}" "${OUTPUT_DIR}/checkpoint/$(basename "${CHECKPOINT}")"

    # 保存来源信息，便于追溯
    {
        echo "experiment_name=${EXPERIMENT_NAME}"
        echo "model_name=${MODEL_NAME}"
        echo "seed=${SEED:-config_default}"
        echo "source_config=${CONFIG}"
        echo "source_checkpoint=${CHECKPOINT}"
        echo "saved_config=${OUTPUT_DIR}/config.yaml"
        echo "saved_checkpoint=${OUTPUT_DIR}/checkpoint/$(basename "${CHECKPOINT}")"
        echo "inference_time=$(date -Iseconds)"
        echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    } > "${OUTPUT_DIR}/inference_manifest.txt"

    SAVED_CONFIG="${OUTPUT_DIR}/config.yaml"
    SAVED_CHECKPOINT="${OUTPUT_DIR}/checkpoint/$(basename "${CHECKPOINT}")"

    args=(
        --config "${SAVED_CONFIG}"
        --checkpoint "${SAVED_CHECKPOINT}"
        --output-dir "${OUTPUT_DIR}"
    )

    if [[ -n "${N_VIZ_SHOTS}" ]]; then
        args+=(--n-viz-shots "${N_VIZ_SHOTS}")
    fi

    if [[ -n "${SEED}" ]]; then
        args+=(--seed "${SEED}")
    fi

    if [[ -n "${DEVICE}" ]]; then
        args+=(--device "${DEVICE}")
    fi

    START_TIME="$(date -Iseconds)"

    {
        echo
        echo "============================================================"
        echo "[${START_TIME}] Starting ${EXPERIMENT_NAME}"
        echo "Command: python ${PY_SCRIPT} ${args[*]}"
    } | tee -a "${BATCH_LOG}" "${RUN_LOG}"

    cd "${REPO_ROOT}"

    if python "${PY_SCRIPT}" "${args[@]}" 2>&1 | tee -a "${RUN_LOG}" "${BATCH_LOG}"; then
        END_TIME="$(date -Iseconds)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        # 成功标记，便于后续判断是否真正完成
        {
            echo "${END_TIME}"
            echo "${EXPERIMENT_NAME}"
            echo "${CHECKPOINT}"
        } > "${OUTPUT_DIR}/INFERENCE_SUCCESS"

        echo "[${END_TIME}] SUCCESS: ${EXPERIMENT_NAME}" |
            tee -a "${BATCH_LOG}"
    else
        EXIT_CODE="${PIPESTATUS[0]}"
        END_TIME="$(date -Iseconds)"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_EXPERIMENTS+=("${EXPERIMENT_NAME}: exit code ${EXIT_CODE}")

        {
            echo "${END_TIME}"
            echo "${EXPERIMENT_NAME}"
            echo "exit_code=${EXIT_CODE}"
        } > "${OUTPUT_DIR}/INFERENCE_FAILED"

        echo "[${END_TIME}] FAILED: ${EXPERIMENT_NAME}, exit code ${EXIT_CODE}" |
            tee -a "${BATCH_LOG}"

        if [[ "${STOP_ON_ERROR}" == "true" ]]; then
            echo "STOP_ON_ERROR=true, terminating batch inference."
            exit "${EXIT_CODE}"
        fi
    fi

    echo
done

echo "============================================================"
echo "Batch inference finished: $(date -Iseconds)"
echo "Total   : ${TOTAL_COUNT}"
echo "Success : ${SUCCESS_COUNT}"
echo "Failed  : ${FAILED_COUNT}"
echo "Skipped : ${SKIPPED_COUNT}"
echo "Log     : ${BATCH_LOG}"

if [[ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]]; then
    echo
    echo "Failed experiments:"
    for ITEM in "${FAILED_EXPERIMENTS[@]}"; do
        echo "  ${ITEM}"
    done
fi

if [[ ${#SKIPPED_EXPERIMENTS[@]} -gt 0 ]]; then
    echo
    echo "Skipped experiments:"
    for ITEM in "${SKIPPED_EXPERIMENTS[@]}"; do
        echo "  ${ITEM}"
    done
fi

echo "============================================================"

if [[ ${FAILED_COUNT} -gt 0 ]]; then
    exit 1
fi