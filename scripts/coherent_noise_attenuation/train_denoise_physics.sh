#!/usr/bin/env bash
# Physics-constrained deep learning: pre-train classifier + train separation network.
# Step 1: pre-train f-k classifier (single-GPU, single seed).
# Step 2: train separation network (multi-GPU, noise × seed sweep).
set -euo pipefail

# ---------- Configuration ----------
CUDA_VISIBLE_DEVICES="4,5,6,7" 
NPROC_PER_NODE=4
NOISE_LEVELS=(1.0 3.0 5.0 7.0 9.0)
N_SEEDS=3
START_SEED=42
MASTER_PORT=29200
TORCHRUN_EXTRA=""
# ------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_CONFIG="${REPO_ROOT}/configs/coherent_noise_attenuation/denoise_physics.yaml"
PY_SCRIPT="${REPO_ROOT}/scripts/coherent_noise_attenuation/train_denoise_physics.py"

export CUDA_VISIBLE_DEVICES

if [[ ! -f "${BASE_CONFIG}" ]]; then echo "Config not found: ${BASE_CONFIG}" >&2; exit 1; fi
if [[ ! -f "${PY_SCRIPT}" ]]; then echo "Script not found: ${PY_SCRIPT}" >&2; exit 1; fi

NAME_BASE="$(grep -m1 -E '^[[:space:]]*name:[[:space:]]*' "${BASE_CONFIG}" | sed -E 's/^[[:space:]]*name:[[:space:]]*//' | sed -E 's/[[:space:]]+#.*$//;s/[[:space:]]*$//')"
tmpcfg="$(mktemp)"
cleanup() { rm -f "${tmpcfg}"; }
trap cleanup EXIT

# --- Pre-train classifier (once, single GPU) ---
echo "[$(date -Iseconds)] Pre-training f-k classifier..."
sed -E \
  -e 's/^([[:space:]]*name:[[:space:]]*).*/\1'"${NAME_BASE}"'_classifier/' \
  "${BASE_CONFIG}" >"${tmpcfg}"
cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" python "${PY_SCRIPT}" --config "${tmpcfg}" --pretrain_classifier

# copy classifier to the base name so phase 2 finds it
CLASSIFIER_SRC="$(grep output_dir "${tmpcfg}" | sed -E 's/.*output_dir:[[:space:]]*//')/${NAME_BASE}_classifier/fk_classifier.pt"
echo "Classifier saved; will be copied per experiment below."

# --- Train separation network (noise × seed sweep) ---
n_levels=${#NOISE_LEVELS[@]}
n_total=$((n_levels * N_SEEDS))
run_idx=0

for level in "${NOISE_LEVELS[@]}"; do
  for ((s = 0; s < N_SEEDS; s++)); do
    seed=$((START_SEED + s))
    run_idx=$((run_idx + 1))
    run_name="${NAME_BASE}_level${level}_seed${seed}"
    sed -E \
      -e '/input_path:/s/(noisy_)[0-9.]+(\.sgy)/\1'"${level}"'\2/' \
      -e '/target_path:/s/(noise_)[0-9.]+(\.sgy)/\1'"${level}"'\2/' \
      -e 's/^([[:space:]]*seed:[[:space:]]*)[0-9]+$/\1'"${seed}"'/' \
      -e 's/^([[:space:]]*name:[[:space:]]*).*/\1'"${run_name}"'/' \
      -e '/output_dir:/s|/[^/]*$|/'"${NAME_BASE}_classifier"'|' \
      "${BASE_CONFIG}" >"${tmpcfg}"

    # ensure fk_classifier.pt exists in the experiment dir
    EXP_DIR="$(grep output_dir "${tmpcfg}" | sed -E 's/.*output_dir:[[:space:]]*//')/${run_name}"
    mkdir -p "${EXP_DIR}"
    if [[ -f "${CLASSIFIER_SRC}" ]]; then
      cp "${CLASSIFIER_SRC}" "${EXP_DIR}/fk_classifier.pt"
    fi

    port=$((MASTER_PORT + run_idx - 1))
    echo "[$(date -Iseconds)] (${run_idx}/${n_total}) level=${level} seed=${seed} name=${run_name} port=${port}"
    cd "${REPO_ROOT}"
    torchrun ${TORCHRUN_EXTRA} --nproc_per_node="${NPROC_PER_NODE}" --master_port="${port}" "${PY_SCRIPT}" --config "${tmpcfg}"
  done
done

echo "[$(date -Iseconds)] Done ${n_total} runs."
