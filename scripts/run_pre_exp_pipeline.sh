#!/usr/bin/env bash

set -euo pipefail

# 这个脚本把 DeepScaleR 正式预实验链路串起来：
# teacher_generate -> student_score -> select_candidates -> build_distill_dataset
# -> analyze_dataset -> train -> checkpoint_eval -> final_eval -> plot_curves
#
# 设计目标：
# 1. 后续你自己复现实验时，只需要改顶部变量，不需要手工拼很多命令。
# 2. 默认沿用当前项目已经验证过的 DeepScaleR smoke 配置。
# 3. 重要参数都在脚本里加中文注释，方便回看“为什么当时这么设”。


###############################################################################
# 基础环境
###############################################################################

CONDA_RUN="${HOME}/miniconda3/bin/conda run -n adistill-unified"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# 固定使用 vLLM 0.8.5 的 V0 engine。
# 这是本项目当前冻结的实验口径。
export VLLM_USE_V1=0

# 在这台服务器上，8 卡 vLLM 默认多进程方式会触发
# “Cannot re-initialize CUDA in forked subprocess”。
# 显式改成 spawn 可以稳定启动多卡 vLLM。
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 一些 PyTorch / Triton 的运行时会在首次编译 CUDA helper 时走到
# `gcc ... -lcuda` 这条链路；服务器默认库路径里往往只有 `libcuda.so.1`，
# 缺少链接阶段需要的无版本名 `libcuda.so`。
# 这里补上 driver stub 所在目录，属于很常见的“只影响编译期搜索路径”的修复范式。
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}"

# 减少多进程场景下 CPU 线程争抢。
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1


###############################################################################
# 路径与实验规模
###############################################################################

TEACHER_MODEL="/home/disk1/public_checkpoint/Qwen3-8B"
STUDENT_MODEL="/home/disk1/public_checkpoint/Qwen3-1.7B"

# 这次是较大规模的数据侧预实验，不再沿用 smoke 目录，
# 避免把已经跑好的 128-sample smoke 结果覆盖掉。
EXPERIMENT_NAME="deepscaler_main8000_k8_t0.9_p0.85_len4096"
RESULT_ROOT="result/pre_exp"
CANDIDATE_DIR="${RESULT_ROOT}/candidates/${EXPERIMENT_NAME}"
DATASET_DIR="${RESULT_ROOT}/datasets/${EXPERIMENT_NAME}"
SELECTION_DIR="${DATASET_DIR}/selections"
# 训练产物（中间 checkpoint / final checkpoint / train.log）显著更占空间，
# 这里单独放到 /home/disk2/shiyixuan，避免把仓库所在磁盘写满。
RUN_DIR="/home/disk2/shiyixuan/pre_exp_runs/${EXPERIMENT_NAME}"
ANALYSIS_DIR="${RESULT_ROOT}/analysis/${EXPERIMENT_NAME}"
LOG_DIR="${ANALYSIS_DIR}/logs"

mkdir -p "${CANDIDATE_DIR}" "${DATASET_DIR}" "${SELECTION_DIR}" "${ANALYSIS_DIR}" "${LOG_DIR}"

# 默认拒绝覆盖已经存在且非空的重要产物，避免长任务误重跑时把结果冲掉。
# 确认要重跑某个阶段时，在命令前显式加 `ALLOW_OVERWRITE=1`。
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

# Teacher 每题采 8 个候选，这是当前 response-level 预实验的核心设置。
NUM_CANDIDATES=8
TEMPERATURE=0.9
TOP_P=0.85

GEN_MAX_NEW_TOKENS=4096

# prompt_batch_size 是“每个 TP=1 副本”一次送进 vLLM 的 prompt 数。
# 多副本并行后，总吞吐来自多个单卡 engine 同时推进，不再依赖一个 TP=8 engine。
GEN_PROMPT_BATCH_SIZE=64

# max_num_seqs 控制 vLLM scheduler 同时维护的序列数。
# 这是吞吐相关参数，不是“总共生成多少条”的参数：
# - 太小：GPU 并行度吃不满，teacher generate 会非常慢。
# - 太大：KV cache 压力更高，可能 OOM。
# 128 是 vLLM 0.8.5 的常见默认量级，适合作为单卡 Qwen3-8B 副本的起点。
GEN_MAX_NUM_SEQS=128

# save_every_prompts 控制增量落盘频率。
# 这里直接和每个副本的 prompt_batch_size 对齐成 64，属于长任务里很常见的稳妥范式：
# - 每完成一个 outer batch 就落一次盘
# - 中途被打断时，最多只损失当前 batch 的结果
SAVE_EVERY_PROMPTS=64


###############################################################################
# Student 打分参数
###############################################################################

# 打分是前向算 NLL，不需要多卡；默认单卡即可。
# 这里把物理 GPU 和进程内 device 分开写，是一个很常见也很实用的范式：
# - `CUDA_VISIBLE_DEVICES=5` 决定“这个进程实际上只能看到哪张物理卡”
# - 进程内部再用 `cuda:0`，表示“当前可见设备里的第 0 张”
# 这样可以稳定避开服务器上已经被别人占用的物理 GPU0。
SCORE_VISIBLE_DEVICES="5"
SCORE_DEVICE="cuda:0"
SCORE_BATCH_SIZE=4
SCORE_MAX_LENGTH=8192

# Student 打分和 SFT 使用 Transformers/PyTorch，不走 vLLM。
# 当前 Qwen3 + Transformers 环境支持 flash_attention_2，长上下文 NLL 打分会明显受益。
HF_ATTN_IMPLEMENTATION="flash_attention_2"


###############################################################################
# Teacher 生成资源
###############################################################################

# Qwen3-8B 单卡即可放下。这里用 TP=1 的多副本并行替代 TP=8：
# - 每张卡各跑一个完整 Teacher 副本，避免 tensor parallel 通信拖慢长生成
# - teacher_generate.py 用 deterministic shard 切分样本，最后合并 JSONL
TEACHER_REPLICA_GPU_IDS=(0 1 2 3 4 5 6 7)
TEACHER_TP_SIZE=1
TEACHER_NUM_REPLICAS=${#TEACHER_REPLICA_GPU_IDS[@]}

# 大规模 teacher generate 优先关闭 enforce-eager，让 vLLM 使用 CUDA graph。
# 若遇到 CUDA graph capture OOM、卡死或静态图相关异常，把这里改成 1 可快速回退。
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

# max_steps 现在是训练预算的一等公民。
# 当前 8000 条训练样本、8 卡、per-device batch size=2 时，
# global batch size = 8 * 2 = 16；
# 因此 1000 step 大约对应 8000 / 16 = 500 step/epoch，也就是约 2 个 epoch。
# 这个预算足够拉开 baseline / adversarial 的学习曲线，又不会把训练拖得过长。
MAX_STEPS="${MAX_STEPS:-1000}"
EVAL_EVERY="${EVAL_EVERY:-200}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-200}"
MAX_CHECKPOINTS_TO_KEEP="${MAX_CHECKPOINTS_TO_KEEP:-5}"

# main8000 distill 样本的 full length 中位数约 1.27k，p95 约 4.2k。
# 5120 基本覆盖本轮样本，避免 700 长度设置只学到 response 开头。
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-5120}"

# DeepScaleR 推理链比 GSM8K 长很多；rollout eval 对齐 Teacher 生成上限。
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
# Qwen3-1.7B 单卡可以完整承载 eval engine；TP=1 可以避免不必要通信。
# checkpoint eval 会把多个 checkpoint 分配到不同 GPU 上并行跑。
EVAL_GPU_IDS="${EVAL_GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a EVAL_GPU_ID_LIST <<< "${EVAL_GPU_IDS}"
EVAL_TP_SIZE="${EVAL_TP_SIZE:-1}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-8192}"
EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-32}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.85}"

# 固定 1024 条 DeepScaleR holdout 子集，用来看 checkpoint 学习曲线。
CHECKPOINT_EVAL_MAX_SAMPLES="${CHECKPOINT_EVAL_MAX_SAMPLES:-1024}"

# final eval 跑更大的 DeepScaleR holdout 子集，作为本轮主结论。
FINAL_EVAL_MAX_SAMPLES="${FINAL_EVAL_MAX_SAMPLES:-4096}"

# DeepScaleR 只有 train split；holdout 定义为排除 main8000 训练子集后的剩余样本。
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-${DATASET_NAME}}"
EVAL_DATASET_CONFIG_NAME="${EVAL_DATASET_CONFIG_NAME:-${DATASET_CONFIG_NAME}}"
EVAL_SPLIT="${EVAL_SPLIT:-${SPLIT}}"
EVAL_QUESTION_FIELD="${EVAL_QUESTION_FIELD:-${QUESTION_FIELD}}"
EVAL_ANSWER_FIELD="${EVAL_ANSWER_FIELD:-${ANSWER_FIELD}}"
EVAL_EXCLUDE_MAX_SAMPLES="${EVAL_EXCLUDE_MAX_SAMPLES:-${MAX_SAMPLES}}"
EVAL_EXCLUDE_SEED="${EVAL_EXCLUDE_SEED:-${SUBSET_SEED}}"

# trainer 的 val_loss/ppl 使用固定的 DeepScaleR holdout JSONL，避免每次 eval 扫全量 40k。
HOLDOUT_EVAL_MAX_SAMPLES="${HOLDOUT_EVAL_MAX_SAMPLES:-1024}"
HOLDOUT_EVAL_COMPLETION_FIELD="${HOLDOUT_EVAL_COMPLETION_FIELD:-solution}"
HOLDOUT_EVAL_FILE="${DATASET_DIR}/deepscaler_holdout_eval_${HOLDOUT_EVAL_MAX_SAMPLES}_seed${SUBSET_SEED}.jsonl"
HOLDOUT_EVAL_SUMMARY_FILE="${HOLDOUT_EVAL_FILE%.jsonl}.summary.json"


###############################################################################
# 阶段开关
###############################################################################

# 这些开关方便你重跑局部阶段。
# 例如只想重跑分析，可以把前面都改成 0，只保留 RUN_ANALYZE_DATASET=1。
# 现在 8000 条 main 数据已经产出后，可以用下面这种方式只启动训练：
# RUN_TEACHER_GENERATE=0 RUN_STUDENT_SCORE=0 RUN_SELECT_AND_BUILD=0 RUN_ANALYZE_DATASET=0 RUN_TRAIN=1 bash scripts/run_pre_exp_pipeline.sh
# main8000 数据侧已经产出，默认只跑剩余分析、训练和评测阶段。
RUN_TEACHER_GENERATE="${RUN_TEACHER_GENERATE:-0}"
RUN_STUDENT_SCORE="${RUN_STUDENT_SCORE:-0}"
RUN_SELECT_AND_BUILD="${RUN_SELECT_AND_BUILD:-0}"
RUN_ANALYZE_DATASET="${RUN_ANALYZE_DATASET:-1}"
RUN_BUILD_HOLDOUT_EVAL="${RUN_BUILD_HOLDOUT_EVAL:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"

# checkpoint 子集评测不是必须，但对看学习曲线很有帮助。
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
        --engine "${EVAL_ENGINE}" \
        --model-name-or-path "${checkpoint_dir}" \
        --dataset-name "${EVAL_DATASET_NAME}" \
        --dataset-config-name "${EVAL_DATASET_CONFIG_NAME}" \
        --split "${EVAL_SPLIT}" \
        --question-field "${EVAL_QUESTION_FIELD}" \
        --answer-field "${EVAL_ANSWER_FIELD}" \
        --exclude-subset-max-samples "${EVAL_EXCLUDE_MAX_SAMPLES}" \
        --exclude-subset-seed "${EVAL_EXCLUDE_SEED}" \
        --max-samples "${CHECKPOINT_EVAL_MAX_SAMPLES}" \
        --subset-seed "${SUBSET_SEED}" \
        --max-new-tokens "${ROLLOUT_MAX_NEW_TOKENS}" \
        --tensor-parallel-size "${EVAL_TP_SIZE}" \
        --max-model-len "${EVAL_MAX_MODEL_LEN}" \
        --max-num-seqs "${EVAL_MAX_NUM_SEQS}" \
        --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION}" \
        --trust-remote-code \
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

  # final eval 使用 DeepScaleR holdout，排除 main8000 训练子集。
  run_cmd bash -lc "
    CUDA_VISIBLE_DEVICES=${gpu_id} \
    ${CONDA_RUN} python src/pre_exp/final_eval.py \
      --engine ${EVAL_ENGINE} \
      --model-name-or-path ${RUN_DIR}/${mode}/final_checkpoint \
      --dataset-name ${EVAL_DATASET_NAME} \
      --dataset-config-name ${EVAL_DATASET_CONFIG_NAME} \
      --split ${EVAL_SPLIT} \
      --question-field ${EVAL_QUESTION_FIELD} \
      --answer-field ${EVAL_ANSWER_FIELD} \
      --exclude-subset-max-samples ${EVAL_EXCLUDE_MAX_SAMPLES} \
      --exclude-subset-seed ${EVAL_EXCLUDE_SEED} \
      --max-samples ${FINAL_EVAL_MAX_SAMPLES} \
      --subset-seed ${SUBSET_SEED} \
      --max-new-tokens ${ROLLOUT_MAX_NEW_TOKENS} \
      --tensor-parallel-size ${EVAL_TP_SIZE} \
      --max-model-len ${EVAL_MAX_MODEL_LEN} \
      --max-num-seqs ${EVAL_MAX_NUM_SEQS} \
      --gpu-memory-utilization ${EVAL_GPU_MEMORY_UTILIZATION} \
      --trust-remote-code \
      --output-file ${output_file} \
      > ${LOG_DIR}/final_eval_${mode}.log 2>&1
  "
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
# 阶段 G: 最终 full test 评测
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
