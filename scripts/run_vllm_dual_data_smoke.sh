#!/usr/bin/env bash

set -euo pipefail

# vLLM-dual token-level decoding 的 data-side smoke：
# vllm_dual_decoding/teacher_generate(plain/hard/soft)
# -> vllm_dual_decoding/student_score
# -> vllm_dual_decoding/analyze_generation_modes
#
# 本脚本刻意不包含 SFT 训练和 checkpoint eval。当前目标是先确认：
# 1. plain 仍走普通 vLLM Worker。
# 2. hard/soft 能通过独立 token-level 入口启用 DualModelWorker。
# 3. 三组生成结果能被现有质量标注和 Student NLL 打分消费。

CONDA_RUN="${HOME}/miniconda3/bin/conda run -n adistill-unified"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1


###############################################################################
# 路径与模型
###############################################################################

TEACHER_MODEL="/home/disk1/public_checkpoint/Qwen3-8B"
STUDENT_MODEL="/home/disk1/public_checkpoint/Qwen3-1.7B"

EXPERIMENT_NAME="vllm_dual_data_smoke_gsm8k24"
RESULT_ROOT="result/vllm_dual_decoding"
CANDIDATE_ROOT="${RESULT_ROOT}/candidates/${EXPERIMENT_NAME}"
ANALYSIS_DIR="${RESULT_ROOT}/analysis/${EXPERIMENT_NAME}"
LOG_DIR="${ANALYSIS_DIR}/logs"
DATA_QUALITY_SUMMARY_FILE="${ANALYSIS_DIR}/data_quality_summary.json"

mkdir -p "${CANDIDATE_ROOT}" "${ANALYSIS_DIR}" "${LOG_DIR}"


###############################################################################
# 数据与生成参数
###############################################################################

DATASET_NAME="openai/gsm8k"
DATASET_CONFIG_NAME="main"
SPLIT="train"
QUESTION_FIELD="question"
ANSWER_FIELD="answer"

MAX_SAMPLES=24
SUBSET_SEED=42

NUM_CANDIDATES=1
TEMPERATURE=0.7
TOP_P=0.9
GEN_MAX_NEW_TOKENS=512
GEN_MAX_MODEL_LEN=2048
GEN_PROMPT_BATCH_SIZE=8
GEN_MAX_NUM_SEQS=16
SAVE_EVERY_PROMPTS=8


###############################################################################
# GPU 与 vLLM-dual 参数
###############################################################################

# GPU 当前紧张时不要直接运行本脚本；等资源空出来后再按需调整这里。
DUAL_GPU_IDS="4,5,6,7"
WORKER_SMOKE_GPU_IDS="4,5"
TEACHER_TP_SIZE=4
SMALL_TP_SIZE=1

GPU_MEMORY_UTILIZATION=0.70
SMALL_GPU_MEMORY_UTILIZATION=0.35
ENFORCE_EAGER=1
EAGER_ARG=""
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  EAGER_ARG="--enforce-eager"
fi

HARD_CANDIDATE_TOP_K=20
HARD_CANDIDATE_TOP_P=0.95
SOFT_STUDENT_WEIGHT=1.0
SOFT_TEMPERATURE=1.0
DEBUG_LOG_INTERVAL=16


###############################################################################
# Student 打分参数
###############################################################################

SCORE_VISIBLE_DEVICES="4"
SCORE_DEVICE="cuda:0"
SCORE_BATCH_SIZE=2
SCORE_MAX_LENGTH=2048
HF_ATTN_IMPLEMENTATION="flash_attention_2"


###############################################################################
# 阶段开关
###############################################################################

RUN_GENERATE=1
RUN_SCORE=1
RUN_ANALYZE=1

MODE_LABELS=(
  "teacher_plain"
  "teacher_token_hard"
  "teacher_token_soft"
)

MODE_ARGS=(
  "plain"
  "hard"
  "soft"
)


###############################################################################
# 工具函数
###############################################################################

run_cmd() {
  echo
  echo "[$(date '+%F %T')] $*"
  "$@"
}

mode_dir() {
  local mode_label="$1"
  echo "${CANDIDATE_ROOT}/${mode_label}"
}

candidate_file() {
  local mode_label="$1"
  echo "$(mode_dir "${mode_label}")/candidate_pool.jsonl"
}

scored_file() {
  local mode_label="$1"
  echo "$(mode_dir "${mode_label}")/scored_candidates.jsonl"
}


###############################################################################
# 阶段 A: 三种模式 Teacher 生成
###############################################################################

if [[ "${RUN_GENERATE}" == "1" ]]; then
  for idx in "${!MODE_LABELS[@]}"; do
    mode_label="${MODE_LABELS[$idx]}"
    mode_arg="${MODE_ARGS[$idx]}"
    mkdir -p "$(mode_dir "${mode_label}")"

    run_cmd bash -lc "
      CUDA_VISIBLE_DEVICES=${DUAL_GPU_IDS} \
      ${CONDA_RUN} python src/vllm_dual_decoding/teacher_generate.py \
        --generation-mode ${mode_arg} \
        --model-name-or-path ${TEACHER_MODEL} \
        --student-model-name-or-path ${STUDENT_MODEL} \
        --dataset-name ${DATASET_NAME} \
        --dataset-config-name ${DATASET_CONFIG_NAME} \
        --split ${SPLIT} \
        --question-field ${QUESTION_FIELD} \
        --answer-field ${ANSWER_FIELD} \
        --output-file $(candidate_file "${mode_label}") \
        --max-samples ${MAX_SAMPLES} \
        --subset-seed ${SUBSET_SEED} \
        --num-candidates ${NUM_CANDIDATES} \
        --temperature ${TEMPERATURE} \
        --top-p ${TOP_P} \
        --max-new-tokens ${GEN_MAX_NEW_TOKENS} \
        --prompt-batch-size ${GEN_PROMPT_BATCH_SIZE} \
        --save-every-prompts ${SAVE_EVERY_PROMPTS} \
        --max-num-seqs ${GEN_MAX_NUM_SEQS} \
        --max-model-len ${GEN_MAX_MODEL_LEN} \
        --tensor-parallel-size ${TEACHER_TP_SIZE} \
        --small-tensor-parallel-size ${SMALL_TP_SIZE} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --small-gpu-memory-utilization ${SMALL_GPU_MEMORY_UTILIZATION} \
        --hard-candidate-top-k ${HARD_CANDIDATE_TOP_K} \
        --hard-candidate-top-p ${HARD_CANDIDATE_TOP_P} \
        --soft-student-weight ${SOFT_STUDENT_WEIGHT} \
        --soft-temperature ${SOFT_TEMPERATURE} \
        --debug-log-interval ${DEBUG_LOG_INTERVAL} \
        ${EAGER_ARG} \
        --trust-remote-code \
        --use-tqdm \
        2>&1 | tee ${LOG_DIR}/teacher_generate_${mode_label}.log
    "
  done
fi


###############################################################################
# 阶段 B: 三种模式 Student NLL 打分
###############################################################################

if [[ "${RUN_SCORE}" == "1" ]]; then
  for mode_label in "${MODE_LABELS[@]}"; do
    run_cmd bash -lc "
      CUDA_VISIBLE_DEVICES=${SCORE_VISIBLE_DEVICES} \
      ${CONDA_RUN} python src/vllm_dual_decoding/student_score.py \
        --model-name-or-path ${STUDENT_MODEL} \
        --input-file $(candidate_file "${mode_label}") \
        --output-file $(scored_file "${mode_label}") \
        --batch-size ${SCORE_BATCH_SIZE} \
        --max-length ${SCORE_MAX_LENGTH} \
        --device ${SCORE_DEVICE} \
        --attn-implementation ${HF_ATTN_IMPLEMENTATION} \
        2>&1 | tee ${LOG_DIR}/student_score_${mode_label}.log
    "
  done
fi


###############################################################################
# 阶段 C: 数据质量分析
###############################################################################

if [[ "${RUN_ANALYZE}" == "1" ]]; then
  run_cmd ${CONDA_RUN} python src/vllm_dual_decoding/analyze_generation_modes.py \
    --mode-file "teacher_plain=$(scored_file teacher_plain)" \
    --mode-file "teacher_token_hard=$(scored_file teacher_token_hard)" \
    --mode-file "teacher_token_soft=$(scored_file teacher_token_soft)" \
    --output-file "${DATA_QUALITY_SUMMARY_FILE}"
fi

echo
echo "vLLM-dual data-side smoke finished."
echo "Candidates: ${CANDIDATE_ROOT}"
echo "Analysis:   ${DATA_QUALITY_SUMMARY_FILE}"
