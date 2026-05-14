#!/usr/bin/env bash

set -euo pipefail

# vLLM-dual token-level 正式数据构建：只跑 teacher_token_hard。

CONDA_RUN="${HOME}/miniconda3/bin/conda run --no-capture-output -n adistill-unified"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

TEACHER_MODEL="${TEACHER_MODEL:-/home/disk1/public_checkpoint/Qwen3-8B}"
STUDENT_MODEL="${STUDENT_MODEL:-/home/disk1/public_checkpoint/Qwen3-1.7B}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-vllm_dual_deepscaler8000_k1_t0.7_p0.8_len4096}"
RESULT_ROOT="${RESULT_ROOT:-result/vllm_dual_decoding}"
CANDIDATE_DIR="${RESULT_ROOT}/candidates/${EXPERIMENT_NAME}/hard_bs96_s80"
SHARD_DIR="${CANDIDATE_DIR}/shards"
ANALYSIS_DIR="${RESULT_ROOT}/analysis/${EXPERIMENT_NAME}/hard_bs96_s80"
LOG_DIR="${ANALYSIS_DIR}/logs"
mkdir -p "${SHARD_DIR}" "${LOG_DIR}"

DATASET_NAME="${DATASET_NAME:-agentica-org/DeepScaleR-Preview-Dataset}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-default}"
SPLIT="${SPLIT:-train}"
QUESTION_FIELD="${QUESTION_FIELD:-problem}"
ANSWER_FIELD="${ANSWER_FIELD:-answer}"
MAX_SAMPLES="${MAX_SAMPLES:-8000}"
SUBSET_SEED="${SUBSET_SEED:-42}"

NUM_CANDIDATES="${NUM_CANDIDATES:-1}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-4096}"
GEN_MAX_MODEL_LEN="${GEN_MAX_MODEL_LEN:-8192}"
GEN_PROMPT_BATCH_SIZE="${GEN_PROMPT_BATCH_SIZE:-96}"
GEN_MAX_NUM_SEQS="${GEN_MAX_NUM_SEQS:-80}"
SAVE_EVERY_PROMPTS="${SAVE_EVERY_PROMPTS:-64}"

GENERATION_GPU_IDS="${GENERATION_GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a GENERATION_GPU_ID_LIST <<< "${GENERATION_GPU_IDS//,/ }"
GENERATION_NUM_SHARDS="${GENERATION_NUM_SHARDS:-${#GENERATION_GPU_ID_LIST[@]}}"
TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-1}"
SMALL_TP_SIZE="${SMALL_TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.78}"
SMALL_GPU_MEMORY_UTILIZATION="${SMALL_GPU_MEMORY_UTILIZATION:-0.35}"

HARD_CANDIDATE_TOP_K="${HARD_CANDIDATE_TOP_K:-20}"
HARD_CANDIDATE_TOP_P="${HARD_CANDIDATE_TOP_P:-0.95}"
DEBUG_LOG_INTERVAL="${DEBUG_LOG_INTERVAL:-256}"

# hard/soft 先保守使用 eager，降低 CUDA graph 额外显存和 capture 不确定性。
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
EAGER_ARG=""
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  EAGER_ARG="--enforce-eager"
fi

ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_GENERATE="${RUN_GENERATE:-1}"

run_cmd() {
  echo
  echo "[$(date '+%F %T')] $*"
  "$@"
}

guard_output_file() {
  local path="$1"
  if [[ "${ALLOW_OVERWRITE}" != "1" && -s "${path}" ]]; then
    echo "Refusing to overwrite non-empty file: ${path}" >&2
    echo "Set ALLOW_OVERWRITE=1 if you intentionally want to rerun this stage." >&2
    exit 1
  fi
}

validate_gpu_plan() {
  if [[ "${#GENERATION_GPU_ID_LIST[@]}" -eq 0 ]]; then
    echo "GENERATION_GPU_IDS must contain at least one GPU id." >&2
    exit 1
  fi
  if [[ "${GENERATION_NUM_SHARDS}" -lt 1 ]]; then
    echo "GENERATION_NUM_SHARDS must be >= 1." >&2
    exit 1
  fi
  if [[ "${GENERATION_NUM_SHARDS}" -gt "${#GENERATION_GPU_ID_LIST[@]}" ]]; then
    echo "GENERATION_NUM_SHARDS=${GENERATION_NUM_SHARDS} exceeds GENERATION_GPU_IDS count=${#GENERATION_GPU_ID_LIST[@]}." >&2
    exit 1
  fi
}

candidate_file() {
  echo "${CANDIDATE_DIR}/candidate_pool.jsonl"
}

candidate_shard_file() {
  local shard_idx="$1"
  printf "%s/candidate_pool.shard_%02d_of_%02d.jsonl" \
    "${SHARD_DIR}" "${shard_idx}" "${GENERATION_NUM_SHARDS}"
}

wait_for_pids() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    return 1
  fi
}

merge_candidate_shards() {
  local merged_file
  merged_file="$(candidate_file)"
  guard_output_file "${merged_file}"
  : > "${merged_file}"
  for ((shard_idx = 0; shard_idx < GENERATION_NUM_SHARDS; shard_idx++)); do
    local shard_file
    shard_file="$(candidate_shard_file "${shard_idx}")"
    if [[ ! -s "${shard_file}" ]]; then
      echo "Missing or empty candidate shard: ${shard_file}" >&2
      exit 1
    fi
    cat "${shard_file}" >> "${merged_file}"
  done
}

run_hard_generate() {
  local pids=()
  local shard_logs=()
  local aggregate_log="${LOG_DIR}/teacher_generate_teacher_token_hard.log"
  guard_output_file "$(candidate_file)"
  guard_output_file "${aggregate_log}"

  echo
  echo "[$(date '+%F %T')] launching teacher_token_hard generation: shards=${GENERATION_NUM_SHARDS}, TP=${TEACHER_TP_SIZE}, GPUs=${GENERATION_GPU_IDS}, prompt_batch=${GEN_PROMPT_BATCH_SIZE}, max_num_seqs=${GEN_MAX_NUM_SEQS}"
  for ((shard_idx = 0; shard_idx < GENERATION_NUM_SHARDS; shard_idx++)); do
    local gpu_id="${GENERATION_GPU_ID_LIST[$shard_idx]}"
    local shard_file
    local shard_log
    shard_file="$(candidate_shard_file "${shard_idx}")"
    shard_log="${LOG_DIR}/teacher_generate_teacher_token_hard_shard_$(printf "%02d" "${shard_idx}").log"
    guard_output_file "${shard_file}"
    guard_output_file "${shard_log}"
    shard_logs+=("${shard_log}")

    echo "[$(date '+%F %T')] teacher_token_hard shard ${shard_idx}/${GENERATION_NUM_SHARDS} -> GPU ${gpu_id}"
    (
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      ${CONDA_RUN} python src/vllm_dual_decoding/teacher_generate.py \
        --generation-mode hard \
        --model-name-or-path "${TEACHER_MODEL}" \
        --student-model-name-or-path "${STUDENT_MODEL}" \
        --dataset-name "${DATASET_NAME}" \
        --dataset-config-name "${DATASET_CONFIG_NAME}" \
        --split "${SPLIT}" \
        --question-field "${QUESTION_FIELD}" \
        --answer-field "${ANSWER_FIELD}" \
        --output-file "${shard_file}" \
        --max-samples "${MAX_SAMPLES}" \
        --subset-seed "${SUBSET_SEED}" \
        --num-shards "${GENERATION_NUM_SHARDS}" \
        --shard-index "${shard_idx}" \
        --num-candidates "${NUM_CANDIDATES}" \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --max-new-tokens "${GEN_MAX_NEW_TOKENS}" \
        --prompt-batch-size "${GEN_PROMPT_BATCH_SIZE}" \
        --save-every-prompts "${SAVE_EVERY_PROMPTS}" \
        --max-num-seqs "${GEN_MAX_NUM_SEQS}" \
        --max-model-len "${GEN_MAX_MODEL_LEN}" \
        --tensor-parallel-size "${TEACHER_TP_SIZE}" \
        --small-tensor-parallel-size "${SMALL_TP_SIZE}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --small-gpu-memory-utilization "${SMALL_GPU_MEMORY_UTILIZATION}" \
        --hard-candidate-top-k "${HARD_CANDIDATE_TOP_K}" \
        --hard-candidate-top-p "${HARD_CANDIDATE_TOP_P}" \
        --debug-log-interval "${DEBUG_LOG_INTERVAL}" \
        ${EAGER_ARG} \
        --trust-remote-code \
        --use-tqdm \
        > "${shard_log}" 2>&1
    ) &
    pids+=("$!")
  done

  if ! wait_for_pids "${pids[@]}"; then
    echo "At least one teacher_token_hard generation shard failed. Logs:" >&2
    printf '  %s\n' "${shard_logs[@]}" >&2
    exit 1
  fi

  merge_candidate_shards
  {
    echo "teacher_token_hard_generate_done shards=${GENERATION_NUM_SHARDS} output_file=$(candidate_file)"
    for shard_log in "${shard_logs[@]}"; do
      echo "shard_log=${shard_log}"
    done
  } | tee "${aggregate_log}"
}

validate_gpu_plan

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  run_cmd ${CONDA_RUN} python -c \
    "import pathlib, vllm; from vllm.worker.dual_worker import DualModelWorker; print('vllm_version', vllm.__version__); print('vllm_path', pathlib.Path(vllm.__file__).parent); print('dual_worker', DualModelWorker.__module__, DualModelWorker.__name__)"
fi

if [[ "${RUN_GENERATE}" == "1" ]]; then
  run_hard_generate
fi

echo
echo "teacher_token_hard data build finished."
echo "Candidates: $(candidate_file)"
echo "Logs:       ${LOG_DIR}"
