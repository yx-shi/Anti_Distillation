#!/usr/bin/env bash

set -euo pipefail

# DeepScaleR main8000 response-level 预实验流水线：
# data side -> analysis -> SFT -> checkpoint eval -> final eval -> curves
#
# main8000 数据侧已经完成，所以默认跳过 teacher generate / score / select，
# 只跑剩余分析、训练、DeepScaleR holdout 评测和绘图。


###############################################################################
# 基础环境
###############################################################################

CONDA_RUN="${HOME}/miniconda3/bin/conda run -n adistill-unified"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# 固定使用 vLLM 0.8.5 的 V0 engine。
export VLLM_USE_V1=0

# 避免 vLLM 多进程 fork 后重新初始化 CUDA。
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 给 Triton/PyTorch 编译 CUDA helper 时补上 libcuda 链接路径。
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}"

# 减少多进程场景下 CPU 线程争抢。
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1


###############################################################################
# 路径与实验规模
###############################################################################

TEACHER_MODEL="/home/disk1/public_checkpoint/Qwen3-8B"
STUDENT_MODEL="/home/disk1/public_checkpoint/Qwen3-1.7B"

EXPERIMENT_NAME="deepscaler_main8000_k8_t0.9_p0.85_len4096"
RESULT_ROOT="result/pre_exp"
CANDIDATE_DIR="${RESULT_ROOT}/candidates/${EXPERIMENT_NAME}"
DATASET_DIR="${RESULT_ROOT}/datasets/${EXPERIMENT_NAME}"
SELECTION_DIR="${DATASET_DIR}/selections"
# 训练 checkpoint 较大，放到 /home/disk2，避免写满仓库所在磁盘。
RUN_DIR="/home/disk2/shiyixuan/pre_exp_runs/${EXPERIMENT_NAME}"
ANALYSIS_DIR="${RESULT_ROOT}/analysis/${EXPERIMENT_NAME}"
LOG_DIR="${ANALYSIS_DIR}/logs"

mkdir -p "${CANDIDATE_DIR}" "${DATASET_DIR}" "${SELECTION_DIR}" "${ANALYSIS_DIR}" "${LOG_DIR}"

# 默认拒绝覆盖非空产物；确认重跑某阶段时显式设置 ALLOW_OVERWRITE=1。
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"


###############################################################################
# 数据与生成参数
###############################################################################

DATASET_NAME="agentica-org/DeepScaleR-Preview-Dataset"
DATASET_CONFIG_NAME="default"
SPLIT="train"
QUESTION_FIELD="problem"
ANSWER_FIELD="answer"

MAX_SAMPLES=8000
SUBSET_SEED=42

NUM_CANDIDATES=8
TEMPERATURE=0.9
TOP_P=0.85

GEN_MAX_NEW_TOKENS=4096

# 每个 TP=1 Teacher 副本一次送入 vLLM 的 prompt 数。
GEN_PROMPT_BATCH_SIZE=64

# vLLM scheduler 同时维护的序列数；太大会增加 KV cache 压力。
GEN_MAX_NUM_SEQS=128

# 每个 outer batch 落盘一次，便于长任务中断后排查。
SAVE_EVERY_PROMPTS=64


###############################################################################
# Student 打分参数
###############################################################################

# Student NLL 打分只需单卡；物理卡由 CUDA_VISIBLE_DEVICES 控制。
SCORE_VISIBLE_DEVICES="5"
SCORE_DEVICE="cuda:0"
SCORE_BATCH_SIZE=4
SCORE_MAX_LENGTH=8192

# Student 打分和 SFT 使用 Transformers/PyTorch，不走 vLLM。
HF_ATTN_IMPLEMENTATION="flash_attention_2"


###############################################################################
# Teacher 生成资源
###############################################################################

# Teacher generation 用 TP=1 多副本，每张卡跑一个 shard。
TEACHER_REPLICA_GPU_IDS=(0 1 2 3 4 5 6 7)
TEACHER_TP_SIZE=1
TEACHER_NUM_REPLICAS=${#TEACHER_REPLICA_GPU_IDS[@]}

# 默认允许 vLLM CUDA graph；遇到 capture 相关问题时改成 1。
TEACHER_ENFORCE_EAGER=0
TEACHER_EAGER_ARG=""
if [[ "${TEACHER_ENFORCE_EAGER}" == "1" ]]; then
  TEACHER_EAGER_ARG="--enforce-eager"
fi


###############################################################################
# SFT 训练参数
###############################################################################

TRAIN_GPU_IDS="0,1,2,3,4,5,6,7"
TRAIN_NPROC=8

# 8000 样本、8 卡、per-device batch=2 时，1000 step 约等于 2 个 epoch。
MAX_STEPS="${MAX_STEPS:-1000}"
EVAL_EVERY="${EVAL_EVERY:-200}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-200}"
MAX_CHECKPOINTS_TO_KEEP="${MAX_CHECKPOINTS_TO_KEEP:-5}"

# main8000 full length p95 约 4.2k；5120 基本覆盖本轮样本。
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-5120}"

# DeepScaleR 推理链较长；rollout eval 对齐 Teacher 生成上限。
ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-4096}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
TRAIN_SEED="${TRAIN_SEED:-42}"


###############################################################################
# 离线评测参数
###############################################################################

EVAL_ENGINE="vllm"
# Qwen3-1.7B eval 单卡足够；checkpoint eval 在脚本层按 GPU 并行调度。
EVAL_GPU_IDS="${EVAL_GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a EVAL_GPU_ID_LIST <<< "${EVAL_GPU_IDS}"
EVAL_TP_SIZE="${EVAL_TP_SIZE:-1}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-8192}"
EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-32}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.85}"

CHECKPOINT_EVAL_MAX_SAMPLES="${CHECKPOINT_EVAL_MAX_SAMPLES:-1024}"

FINAL_EVAL_MAX_SAMPLES="${FINAL_EVAL_MAX_SAMPLES:-4096}"

# DeepScaleR 只有 train split；holdout 定义为排除 main8000 训练子集后的剩余样本。
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-${DATASET_NAME}}"
EVAL_DATASET_CONFIG_NAME="${EVAL_DATASET_CONFIG_NAME:-${DATASET_CONFIG_NAME}}"
EVAL_SPLIT="${EVAL_SPLIT:-${SPLIT}}"
EVAL_QUESTION_FIELD="${EVAL_QUESTION_FIELD:-${QUESTION_FIELD}}"
EVAL_ANSWER_FIELD="${EVAL_ANSWER_FIELD:-${ANSWER_FIELD}}"
EVAL_EXCLUDE_MAX_SAMPLES="${EVAL_EXCLUDE_MAX_SAMPLES:-${MAX_SAMPLES}}"
EVAL_EXCLUDE_SEED="${EVAL_EXCLUDE_SEED:-${SUBSET_SEED}}"

# trainer val_loss/ppl 使用固定 holdout JSONL，避免每次 eval 扫全量 40k。
HOLDOUT_EVAL_MAX_SAMPLES="${HOLDOUT_EVAL_MAX_SAMPLES:-1024}"
HOLDOUT_EVAL_COMPLETION_FIELD="${HOLDOUT_EVAL_COMPLETION_FIELD:-solution}"
HOLDOUT_EVAL_FILE="${DATASET_DIR}/deepscaler_holdout_eval_${HOLDOUT_EVAL_MAX_SAMPLES}_seed${SUBSET_SEED}.jsonl"
HOLDOUT_EVAL_SUMMARY_FILE="${HOLDOUT_EVAL_FILE%.jsonl}.summary.json"

EVAL_COMMON_ARGS=(
  --engine "${EVAL_ENGINE}"
  --dataset-name "${EVAL_DATASET_NAME}"
  --dataset-config-name "${EVAL_DATASET_CONFIG_NAME}"
  --split "${EVAL_SPLIT}"
  --question-field "${EVAL_QUESTION_FIELD}"
  --answer-field "${EVAL_ANSWER_FIELD}"
  --exclude-subset-max-samples "${EVAL_EXCLUDE_MAX_SAMPLES}"
  --exclude-subset-seed "${EVAL_EXCLUDE_SEED}"
  --subset-seed "${SUBSET_SEED}"
  --max-new-tokens "${ROLLOUT_MAX_NEW_TOKENS}"
  --tensor-parallel-size "${EVAL_TP_SIZE}"
  --max-model-len "${EVAL_MAX_MODEL_LEN}"
  --max-num-seqs "${EVAL_MAX_NUM_SEQS}"
  --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION}"
  --trust-remote-code
)


###############################################################################
# 阶段开关
###############################################################################

# 阶段开关：默认跳过已完成的数据侧，直接跑剩余 main8000 链路。
RUN_TEACHER_GENERATE="${RUN_TEACHER_GENERATE:-0}"
RUN_STUDENT_SCORE="${RUN_STUDENT_SCORE:-0}"
RUN_SELECT_AND_BUILD="${RUN_SELECT_AND_BUILD:-0}"
RUN_ANALYZE_DATASET="${RUN_ANALYZE_DATASET:-1}"
RUN_BUILD_HOLDOUT_EVAL="${RUN_BUILD_HOLDOUT_EVAL:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"

RUN_CHECKPOINT_EVAL="${RUN_CHECKPOINT_EVAL:-1}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-1}"
RUN_PLOT_CURVES="${RUN_PLOT_CURVES:-1}"


###############################################################################
# 文件名约定
###############################################################################

CANDIDATE_POOL_FILE="${CANDIDATE_DIR}/candidate_pool.jsonl"
SCORED_CANDIDATES_FILE="${CANDIDATE_DIR}/scored_candidates.jsonl"
DATASET_SUMMARY_FILE="${ANALYSIS_DIR}/dataset_summary_expanded.json"
CURVE_DIR="${ANALYSIS_DIR}/curves"

MODES=(
  "teacher_baseline"
  "teacher_adversarial"
)


###############################################################################
# 工具函数
###############################################################################

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

guard_output_dir() {
  local path="$1"
  if [[ "${ALLOW_OVERWRITE}" != "1" && -d "${path}" && -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to reuse non-empty directory: ${path}" >&2
    echo "Set ALLOW_OVERWRITE=1 if you intentionally want to rerun this stage." >&2
    exit 1
  fi
}

teacher_shard_file() {
  local shard_idx="$1"
  printf "%s/candidate_pool.shard_%02d_of_%02d.jsonl" \
    "${CANDIDATE_DIR}" "${shard_idx}" "${TEACHER_NUM_REPLICAS}"
}

checkpoint_label_from_dir() {
  local checkpoint_dir="$1"
  local name
  name="$(basename "${checkpoint_dir}")"
  if [[ "${name}" == checkpoint-step-* ]]; then
    printf "%s" "${name#checkpoint-step-}"
  else
    printf "%s" "${name}"
  fi
}

wait_for_eval_batch() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "At least one eval worker failed. Check logs under ${LOG_DIR}." >&2
    exit 1
  fi
}

merge_teacher_shards() {
  local merged_file="$1"
  guard_output_file "${merged_file}"
  : > "${merged_file}"
  for ((shard_idx = 0; shard_idx < TEACHER_NUM_REPLICAS; shard_idx++)); do
    local shard_file
    shard_file="$(teacher_shard_file "${shard_idx}")"
    if [[ ! -s "${shard_file}" ]]; then
      echo "Missing or empty teacher shard: ${shard_file}" >&2
      exit 1
    fi
    cat "${shard_file}" >> "${merged_file}"
  done
}

run_teacher_generate_replicas() {
  guard_output_file "${CANDIDATE_POOL_FILE}"

  local pids=()
  local shard_logs=()
  for ((shard_idx = 0; shard_idx < TEACHER_NUM_REPLICAS; shard_idx++)); do
    local gpu_id="${TEACHER_REPLICA_GPU_IDS[${shard_idx}]}"
    local shard_file
    local shard_log
    shard_file="$(teacher_shard_file "${shard_idx}")"
    shard_log="${LOG_DIR}/teacher_generate.shard_$(printf "%02d" "${shard_idx}").log"
    guard_output_file "${shard_file}"
    guard_output_file "${shard_log}"
    shard_logs+=("${shard_log}")

    echo
    echo "[$(date '+%F %T')] launching teacher shard ${shard_idx}/${TEACHER_NUM_REPLICAS} on GPU ${gpu_id}"
    (
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      ${CONDA_RUN} python src/pre_exp/teacher_generate.py \
        --model-name-or-path "${TEACHER_MODEL}" \
        --dataset-name "${DATASET_NAME}" \
        --dataset-config-name "${DATASET_CONFIG_NAME}" \
        --split "${SPLIT}" \
        --question-field "${QUESTION_FIELD}" \
        --answer-field "${ANSWER_FIELD}" \
        --output-file "${shard_file}" \
        --max-samples "${MAX_SAMPLES}" \
        --subset-seed "${SUBSET_SEED}" \
        --num-shards "${TEACHER_NUM_REPLICAS}" \
        --shard-index "${shard_idx}" \
        --num-candidates "${NUM_CANDIDATES}" \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --max-new-tokens "${GEN_MAX_NEW_TOKENS}" \
        --prompt-batch-size "${GEN_PROMPT_BATCH_SIZE}" \
        --save-every-prompts "${SAVE_EVERY_PROMPTS}" \
        --max-num-seqs "${GEN_MAX_NUM_SEQS}" \
        --max-model-len "${SCORE_MAX_LENGTH}" \
        --tensor-parallel-size "${TEACHER_TP_SIZE}" \
        --gpu-memory-utilization 0.85 \
        ${TEACHER_EAGER_ARG} \
        --trust-remote-code \
        --use-tqdm \
        > "${shard_log}" 2>&1
    ) &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done

  if [[ "${failed}" != "0" ]]; then
    echo "At least one teacher shard failed. Logs:" >&2
    printf '  %s\n' "${shard_logs[@]}" >&2
    exit 1
  fi

  merge_teacher_shards "${CANDIDATE_POOL_FILE}"
  guard_output_file "${LOG_DIR}/teacher_generate.log"
  {
    echo "teacher_generate_replicas_done shards=${TEACHER_NUM_REPLICAS} output_file=${CANDIDATE_POOL_FILE}"
    for shard_log in "${shard_logs[@]}"; do
      echo "shard_log=${shard_log}"
    done
  } | tee "${LOG_DIR}/teacher_generate.log"
}

train_one_mode() {
  local mode="$1"
  local port="$2"
  local train_file="${DATASET_DIR}/distill_${mode}.jsonl"
  local output_dir="${RUN_DIR}/${mode}"
  local train_log="${output_dir}/train.log"

  guard_output_dir "${output_dir}"
  mkdir -p "${output_dir}"

  # 这里用 torchrun + 8 卡 FSDP。
  # --standalone：单机实验时自动起 rendezvous，不需要额外 master 节点配置。
  # --nproc_per_node=8：每张卡起一个进程，是 PyTorch 多卡训练的常见范式。
  run_cmd bash -lc "
    CUDA_VISIBLE_DEVICES=${TRAIN_GPU_IDS} \
    ${CONDA_RUN} torchrun --standalone --nproc_per_node=${TRAIN_NPROC} --master_port=${port} \
      src/train_sft.py \
      --model-name-or-path ${STUDENT_MODEL} \
      --dataset-format distill_jsonl \
      --train-file ${train_file} \
      --eval-file ${HOLDOUT_EVAL_FILE} \
      --output-dir ${output_dir} \
      --max-steps ${MAX_STEPS} \
      --eval-every ${EVAL_EVERY} \
      --checkpoint-every ${CHECKPOINT_EVERY} \
      --max-checkpoints-to-keep ${MAX_CHECKPOINTS_TO_KEEP} \
      --max-length ${TRAIN_MAX_LENGTH} \
      --train-batch-size ${TRAIN_BATCH_SIZE} \
      --eval-batch-size ${EVAL_BATCH_SIZE} \
      --learning-rate ${LEARNING_RATE} \
      --weight-decay ${WEIGHT_DECAY} \
      --warmup-ratio ${WARMUP_RATIO} \
      --seed ${TRAIN_SEED} \
      --attn-implementation ${HF_ATTN_IMPLEMENTATION} \
      --rollout-max-new-tokens ${ROLLOUT_MAX_NEW_TOKENS} \
      --disable-rollout-eval \
      --disable-eval-preview \
      2>&1 | tee ${train_log}
  "
}

checkpoint_eval_one_mode() {
  local mode="$1"
  local output_file="${ANALYSIS_DIR}/checkpoint_eval_${mode}.json"
  local output_prefix="checkpoint_eval_${mode}"

  guard_output_file "${output_file}"
  shopt -s nullglob
  local checkpoint_dirs=("${RUN_DIR}/${mode}"/checkpoint-step-*)
  shopt -u nullglob
  if [[ -d "${RUN_DIR}/${mode}/final_checkpoint" ]]; then
    checkpoint_dirs+=("${RUN_DIR}/${mode}/final_checkpoint")
  fi
  local checkpoint_outputs=()
  local checkpoint_dir
  for checkpoint_dir in "${checkpoint_dirs[@]}"; do
    local checkpoint_label
    local checkpoint_output
    checkpoint_label="$(checkpoint_label_from_dir "${checkpoint_dir}")"
    checkpoint_output="${ANALYSIS_DIR}/${output_prefix}_${checkpoint_label}.json"
    guard_output_file "${checkpoint_output}"
    guard_output_file "${checkpoint_output%.json}.records.jsonl"
    guard_output_file "${LOG_DIR}/${output_prefix}_${checkpoint_label}.log"
    checkpoint_outputs+=("${checkpoint_output}")
  done

  if [[ "${#checkpoint_dirs[@]}" == "0" ]]; then
    echo "No checkpoints found for mode ${mode} under ${RUN_DIR}/${mode}" >&2
    exit 1
  fi
  if [[ "${#EVAL_GPU_ID_LIST[@]}" == "0" ]]; then
    echo "EVAL_GPU_IDS is empty." >&2
    exit 1
  fi

  local pids=()
  local launched_in_batch=0
  local checkpoint_idx=0
  for checkpoint_dir in "${checkpoint_dirs[@]}"; do
    local checkpoint_label
    local checkpoint_output
    local checkpoint_log
    local gpu_id
    checkpoint_label="$(checkpoint_label_from_dir "${checkpoint_dir}")"
    checkpoint_output="${ANALYSIS_DIR}/${output_prefix}_${checkpoint_label}.json"
    checkpoint_log="${LOG_DIR}/${output_prefix}_${checkpoint_label}.log"
    gpu_id="${EVAL_GPU_ID_LIST[$((checkpoint_idx % ${#EVAL_GPU_ID_LIST[@]}))]}"

    echo
    echo "[$(date '+%F %T')] launching checkpoint eval mode=${mode} checkpoint=${checkpoint_label} gpu=${gpu_id}"
    (
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      ${CONDA_RUN} python src/pre_exp/final_eval.py \
        "${EVAL_COMMON_ARGS[@]}" \
        --model-name-or-path "${checkpoint_dir}" \
        --max-samples "${CHECKPOINT_EVAL_MAX_SAMPLES}" \
        --output-file "${checkpoint_output}" \
        > "${checkpoint_log}" 2>&1
    ) &
    pids+=("$!")
    launched_in_batch=$((launched_in_batch + 1))
    checkpoint_idx=$((checkpoint_idx + 1))

    if [[ "${launched_in_batch}" == "${#EVAL_GPU_ID_LIST[@]}" ]]; then
      wait_for_eval_batch "${pids[@]}"
      pids=()
      launched_in_batch=0
    fi
  done
  if [[ "${#pids[@]}" != "0" ]]; then
    wait_for_eval_batch "${pids[@]}"
  fi

  run_cmd ${CONDA_RUN} python src/pre_exp/collect_eval_summary.py \
    --output-file "${output_file}" \
    --input-files "${checkpoint_outputs[@]}"
}

final_eval_one_mode() {
  local mode="$1"
  local gpu_id="$2"
  local output_file="${ANALYSIS_DIR}/final_eval_${mode}.json"
  guard_output_file "${output_file}"
  guard_output_file "${output_file%.json}.records.jsonl"
  guard_output_file "${LOG_DIR}/final_eval_${mode}.log"

  run_cmd env CUDA_VISIBLE_DEVICES="${gpu_id}" \
    ${CONDA_RUN} python src/pre_exp/final_eval.py \
      "${EVAL_COMMON_ARGS[@]}" \
      --model-name-or-path "${RUN_DIR}/${mode}/final_checkpoint" \
      --max-samples "${FINAL_EVAL_MAX_SAMPLES}" \
      --output-file "${output_file}" \
      > "${LOG_DIR}/final_eval_${mode}.log" 2>&1
}

plot_curves() {
  guard_output_file "${CURVE_DIR}/curve_data.json"
  guard_output_file "${CURVE_DIR}/train_loss_curve.svg"
  guard_output_file "${CURVE_DIR}/val_loss_ppl_curve.svg"
  guard_output_file "${CURVE_DIR}/rollout_accuracy_curve.svg"

  run_cmd ${CONDA_RUN} python src/pre_exp/plot_curves.py \
    --run-dir "${RUN_DIR}" \
    --analysis-dir "${ANALYSIS_DIR}" \
    --output-dir "${CURVE_DIR}" \
    --modes "${MODES[@]}"
}


###############################################################################
# 阶段 A: Teacher 候选生成
###############################################################################

if [[ "${RUN_TEACHER_GENERATE}" == "1" ]]; then
  run_teacher_generate_replicas
fi


###############################################################################
# 阶段 B: Student 打分
###############################################################################

if [[ "${RUN_STUDENT_SCORE}" == "1" ]]; then
  guard_output_file "${SCORED_CANDIDATES_FILE}"
  run_cmd bash -lc "
    CUDA_VISIBLE_DEVICES=${SCORE_VISIBLE_DEVICES} \
    ${CONDA_RUN} python src/pre_exp/student_score.py \
      --model-name-or-path ${STUDENT_MODEL} \
      --input-file ${CANDIDATE_POOL_FILE} \
      --output-file ${SCORED_CANDIDATES_FILE} \
      --batch-size ${SCORE_BATCH_SIZE} \
      --max-length ${SCORE_MAX_LENGTH} \
      --device ${SCORE_DEVICE} \
      --attn-implementation ${HF_ATTN_IMPLEMENTATION} \
      2>&1 | tee ${LOG_DIR}/student_score.log
  "
fi


###############################################################################
# 阶段 C: 候选选择 + 蒸馏数据构建
###############################################################################

if [[ "${RUN_SELECT_AND_BUILD}" == "1" ]]; then
  guard_output_file "${SELECTION_DIR}/teacher_baseline.selected.jsonl"
  guard_output_file "${SELECTION_DIR}/teacher_adversarial.selected.jsonl"
  for mode in "${MODES[@]}"; do
    guard_output_file "${DATASET_DIR}/distill_${mode}.jsonl"
  done

  run_cmd ${CONDA_RUN} python src/pre_exp/select_candidates.py \
    --input-file "${SCORED_CANDIDATES_FILE}" \
    --output-dir "${SELECTION_DIR}"

  for mode in "${MODES[@]}"; do
    run_cmd ${CONDA_RUN} python src/pre_exp/build_distill_dataset.py \
      --selection-file "${SELECTION_DIR}/${mode}.selected.jsonl" \
      --output-file "${DATASET_DIR}/distill_${mode}.jsonl"
  done
fi


###############################################################################
# 阶段 D: 数据分析
###############################################################################

if [[ "${RUN_ANALYZE_DATASET}" == "1" ]]; then
  guard_output_file "${DATASET_SUMMARY_FILE}"
  run_cmd ${CONDA_RUN} python src/pre_exp/analyze_dataset.py \
    --scored-candidates "${SCORED_CANDIDATES_FILE}" \
    --baseline-file "${SELECTION_DIR}/teacher_baseline.selected.jsonl" \
    --adversarial-file "${SELECTION_DIR}/teacher_adversarial.selected.jsonl" \
    --output-file "${DATASET_SUMMARY_FILE}"
fi


###############################################################################
# 阶段 D2: 构建固定 DeepScaleR holdout eval JSONL
###############################################################################

if [[ "${RUN_BUILD_HOLDOUT_EVAL}" == "1" ]]; then
  guard_output_file "${HOLDOUT_EVAL_FILE}"
  guard_output_file "${HOLDOUT_EVAL_SUMMARY_FILE}"
  run_cmd ${CONDA_RUN} python src/pre_exp/build_holdout_eval_dataset.py \
    --dataset-name "${DATASET_NAME}" \
    --dataset-config-name "${DATASET_CONFIG_NAME}" \
    --split "${SPLIT}" \
    --question-field "${QUESTION_FIELD}" \
    --answer-field "${ANSWER_FIELD}" \
    --completion-field "${HOLDOUT_EVAL_COMPLETION_FIELD}" \
    --output-file "${HOLDOUT_EVAL_FILE}" \
    --summary-file "${HOLDOUT_EVAL_SUMMARY_FILE}" \
    --max-samples "${HOLDOUT_EVAL_MAX_SAMPLES}" \
    --subset-seed "${SUBSET_SEED}" \
    --exclude-subset-max-samples "${MAX_SAMPLES}" \
    --exclude-subset-seed "${SUBSET_SEED}"
fi


###############################################################################
# 阶段 E: 两组 SFT
###############################################################################

if [[ "${RUN_TRAIN}" == "1" ]]; then
  train_one_mode "teacher_baseline" "29601"
  train_one_mode "teacher_adversarial" "29603"
fi


###############################################################################
# 阶段 F: 中间 checkpoint 学习曲线评测（DeepScaleR holdout 固定子集）
###############################################################################

if [[ "${RUN_CHECKPOINT_EVAL}" == "1" ]]; then
  for mode in "${MODES[@]}"; do
    checkpoint_eval_one_mode "${mode}"
  done
fi


###############################################################################
# 阶段 G: 最终 DeepScaleR holdout 评测
###############################################################################

if [[ "${RUN_FINAL_EVAL}" == "1" ]]; then
  pids=()
  mode_idx=0
  for mode in "${MODES[@]}"; do
    gpu_id="${EVAL_GPU_ID_LIST[$((mode_idx % ${#EVAL_GPU_ID_LIST[@]}))]}"
    echo
    echo "[$(date '+%F %T')] launching final eval mode=${mode} gpu=${gpu_id}"
    final_eval_one_mode "${mode}" "${gpu_id}" &
    pids+=("$!")
    mode_idx=$((mode_idx + 1))
  done
  wait_for_eval_batch "${pids[@]}"
fi


###############################################################################
# 阶段 H: 曲线汇总与 SVG 绘图
###############################################################################

if [[ "${RUN_PLOT_CURVES}" == "1" ]]; then
  plot_curves
fi


echo
echo "DeepScaleR main pipeline finished."
echo "Candidates: ${CANDIDATE_DIR}"
echo "Datasets:   ${DATASET_DIR}"
echo "Analysis:   ${DATASET_SUMMARY_FILE}"
echo "Holdout:    ${HOLDOUT_EVAL_FILE}"
echo "Curves:     ${CURVE_DIR}"
