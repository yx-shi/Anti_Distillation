#!/usr/bin/env bash

set -euo pipefail

# DeepScaleR 版本的预实验 smoke，只跑“数据侧”：
# teacher_generate -> student_score -> select_candidates -> build_distill_dataset -> analyze_dataset
#
# 这个脚本刻意不包含 SFT 训练阶段。当前目标是先回答三个数据质量问题：
# 1. 更难数据集上 teacher 的候选正确率是否够用。
# 2. 1024 max_new_tokens 下长度截断是否严重。
# 3. baseline/adversarial 选择出来的 NLL gap 是否比 GSM8K 更明显。

CONDA_RUN="${HOME}/miniconda3/bin/conda run -n adistill-unified"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# vLLM 0.8.5 需要固定走 V0 engine，和项目当前实验口径保持一致。
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

TEACHER_MODEL="/home/disk1/public_checkpoint/Qwen3-8B"
STUDENT_MODEL="/home/disk1/public_checkpoint/Qwen3-1.7B"

DATASET_NAME="agentica-org/DeepScaleR-Preview-Dataset"
DATASET_CONFIG_NAME="default"
SPLIT="train"
QUESTION_FIELD="problem"
ANSWER_FIELD="answer"

EXPERIMENT_NAME="deepscaler_smoke256_k10_t1.0_p0.95_len1024"
RESULT_ROOT="result/pre_exp"
CANDIDATE_DIR="${RESULT_ROOT}/candidates/${EXPERIMENT_NAME}"
DATASET_DIR="${RESULT_ROOT}/datasets/${EXPERIMENT_NAME}"
SELECTION_DIR="${DATASET_DIR}/selections"
ANALYSIS_DIR="${RESULT_ROOT}/analysis/${EXPERIMENT_NAME}"
LOG_DIR="${ANALYSIS_DIR}/logs"

mkdir -p "${CANDIDATE_DIR}" "${DATASET_DIR}" "${SELECTION_DIR}" "${ANALYSIS_DIR}" "${LOG_DIR}"

MAX_SAMPLES=256
SUBSET_SEED=42

# 这轮故意把采样随机性调高一点，增加候选之间的差异：
# - temperature 控制 logits softmax 前的缩放，值越大分布越平，低概率 token 更容易被采到。
# - top_p 是 nucleus sampling，保留累计概率达到 p 的最小 token 集合；0.95 比 0.8 更开放。
NUM_CANDIDATES=10
TEMPERATURE=1.0
TOP_P=0.95

# DeepScaleR 的推理链通常比 GSM8K 长，先从 1024 开始看 finish_reason=length 的比例。
GEN_MAX_NEW_TOKENS=1024
GEN_PROMPT_BATCH_SIZE=32
GEN_MAX_NUM_SEQS=128
SAVE_EVERY_PROMPTS=32

# Teacher 生成吃满 8 张 A100。Qwen3-8B 的 attention heads 可被 8 整除，TP=8 合法。
TEACHER_GPU_IDS="0,1,2,3,4,5,6,7"
TEACHER_TP_SIZE=8

# Student NLL 打分只需要前向，不需要多卡；teacher 结束后单卡跑即可。
SCORE_VISIBLE_DEVICES="0"
SCORE_DEVICE="cuda:0"
SCORE_BATCH_SIZE=2
SCORE_MAX_LENGTH=4096

CANDIDATE_POOL_FILE="${CANDIDATE_DIR}/candidate_pool.jsonl"
SCORED_CANDIDATES_FILE="${CANDIDATE_DIR}/scored_candidates.jsonl"
DATASET_SUMMARY_FILE="${ANALYSIS_DIR}/dataset_summary.json"

MODES=(
  "teacher_baseline"
  "teacher_adversarial"
)

run_cmd() {
  echo
  echo "[$(date '+%F %T')] $*"
  "$@"
}

run_cmd bash -lc "
  CUDA_VISIBLE_DEVICES=${TEACHER_GPU_IDS} \
  ${CONDA_RUN} python src/pre_exp/teacher_generate.py \
    --model-name-or-path ${TEACHER_MODEL} \
    --dataset-name ${DATASET_NAME} \
    --dataset-config-name ${DATASET_CONFIG_NAME} \
    --split ${SPLIT} \
    --question-field ${QUESTION_FIELD} \
    --answer-field ${ANSWER_FIELD} \
    --output-file ${CANDIDATE_POOL_FILE} \
    --max-samples ${MAX_SAMPLES} \
    --subset-seed ${SUBSET_SEED} \
    --num-candidates ${NUM_CANDIDATES} \
    --temperature ${TEMPERATURE} \
    --top-p ${TOP_P} \
    --max-new-tokens ${GEN_MAX_NEW_TOKENS} \
    --prompt-batch-size ${GEN_PROMPT_BATCH_SIZE} \
    --save-every-prompts ${SAVE_EVERY_PROMPTS} \
    --max-num-seqs ${GEN_MAX_NUM_SEQS} \
    --max-model-len ${SCORE_MAX_LENGTH} \
    --tensor-parallel-size ${TEACHER_TP_SIZE} \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --trust-remote-code \
    --use-tqdm \
    2>&1 | tee ${LOG_DIR}/teacher_generate.log
"

run_cmd bash -lc "
  CUDA_VISIBLE_DEVICES=${SCORE_VISIBLE_DEVICES} \
  ${CONDA_RUN} python src/pre_exp/student_score.py \
    --model-name-or-path ${STUDENT_MODEL} \
    --input-file ${CANDIDATE_POOL_FILE} \
    --output-file ${SCORED_CANDIDATES_FILE} \
    --batch-size ${SCORE_BATCH_SIZE} \
    --max-length ${SCORE_MAX_LENGTH} \
    --device ${SCORE_DEVICE} \
    2>&1 | tee ${LOG_DIR}/student_score.log
"

run_cmd ${CONDA_RUN} python src/pre_exp/select_candidates.py \
  --input-file "${SCORED_CANDIDATES_FILE}" \
  --output-dir "${SELECTION_DIR}"

for mode in "${MODES[@]}"; do
  run_cmd ${CONDA_RUN} python src/pre_exp/build_distill_dataset.py \
    --selection-file "${SELECTION_DIR}/${mode}.selected.jsonl" \
    --output-file "${DATASET_DIR}/distill_${mode}.jsonl"
done

run_cmd ${CONDA_RUN} python src/pre_exp/analyze_dataset.py \
  --scored-candidates "${SCORED_CANDIDATES_FILE}" \
  --baseline-file "${SELECTION_DIR}/teacher_baseline.selected.jsonl" \
  --adversarial-file "${SELECTION_DIR}/teacher_adversarial.selected.jsonl" \
  --output-file "${DATASET_SUMMARY_FILE}"

echo
echo "DeepScaleR data-only smoke finished."
echo "Candidates: ${CANDIDATE_DIR}"
echo "Datasets:   ${DATASET_DIR}"
echo "Analysis:   ${DATASET_SUMMARY_FILE}"
