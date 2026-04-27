#!/usr/bin/env bash

set -euo pipefail

# 这个脚本把当前 smoke 预实验的整条链路串起来：
# teacher_generate -> student_score -> select_candidates -> build_distill_dataset
# -> analyze_dataset -> 两组 SFT -> checkpoint subset eval(可选) -> final eval
#
# 设计目标：
# 1. 后续你自己复现实验时，只需要改顶部变量，不需要手工拼很多命令。
# 2. 默认沿用当前项目已经验证过的 smoke 配置。
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
EXPERIMENT_NAME="main5000"
RESULT_ROOT="result/pre_exp"
CANDIDATE_DIR="${RESULT_ROOT}/candidates/${EXPERIMENT_NAME}"
DATASET_DIR="${RESULT_ROOT}/datasets/${EXPERIMENT_NAME}"
SELECTION_DIR="${DATASET_DIR}/selections"
# 训练产物（中间 checkpoint / final checkpoint / train.log）显著更占空间，
# 这里单独放到 /home/disk2/shiyixuan，避免把仓库所在磁盘写满。
RUN_DIR="/home/disk2/shiyixuan/pre_exp_runs/${EXPERIMENT_NAME}"
ANALYSIS_DIR="${RESULT_ROOT}/analysis/${EXPERIMENT_NAME}"
LOG_DIR="${ANALYSIS_DIR}/logs"

mkdir -p "${CANDIDATE_DIR}" "${DATASET_DIR}" "${SELECTION_DIR}" "${RUN_DIR}" "${ANALYSIS_DIR}" "${LOG_DIR}"


###############################################################################
# Smoke 数据与生成参数
###############################################################################

MAX_SAMPLES=5000
SUBSET_SEED=42

# Teacher 每题采 8 个候选，这是当前 response-level 预实验的核心设置。
NUM_CANDIDATES=8
TEMPERATURE=0.7
TOP_P=0.8

# 这里保留 512，而不是 256：
# 数学题推理链较长，太小容易在最终答案出现前被截断。
GEN_MAX_NEW_TOKENS=512

# prompt_batch_size 控制一次送进 vLLM 的 prompt 数。
# 这次把它调到 64，而不是更激进的 256：
# - 对当前 5000 条长任务，更容易更早看到 batch 完成和中间落盘
# - 即使某个 batch 因为长回答拖慢，也不会把“无可见进度”的时间拉得太长
GEN_PROMPT_BATCH_SIZE=64

# max_num_seqs 控制 vLLM scheduler 同时维护的序列数。
# 这是吞吐相关参数，不是“总共生成多少条”的参数：
# - 太小：GPU 并行度吃不满，teacher generate 会非常慢。
# - 太大：KV cache 压力更高，可能 OOM。
# 128 是 vLLM 0.8.5 的常见默认量级，适合作为本次 8 卡 teacher 生成的起点。
GEN_MAX_NUM_SEQS=128

# save_every_prompts 控制增量落盘频率。
# 这里直接和 prompt_batch_size 对齐成 64，属于长任务里很常见的稳妥范式：
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
SCORE_MAX_LENGTH=2048


###############################################################################
# SFT 训练参数
###############################################################################

TRAIN_GPU_IDS="1,2,3,4,5,6,7"
TRAIN_NPROC=7

# max_steps 现在是训练预算的一等公民。
# 当前 5000 条训练样本、7 卡、per-device batch size=2 时，
# global batch size = 7 * 2 = 14；
# 因此 1000 step 大约对应 5000 / 14 ≈ 357 step/epoch，也就是约 2.8 个 epoch。
# 这个预算足够拉开 baseline / adversarial 的学习曲线，又不会把训练拖得过长。
MAX_STEPS=1000
EVAL_EVERY=200
CHECKPOINT_EVERY=200
MAX_CHECKPOINTS_TO_KEEP=5

# main5000 数据里 completion 更长一些，因此把训练长度稍微放宽到 700。
TRAIN_MAX_LENGTH=700

# rollout token 上限用于离线评测，不再在 trainer 里做阻塞式 rollout。
ROLLOUT_MAX_NEW_TOKENS=512

TRAIN_BATCH_SIZE=2
EVAL_BATCH_SIZE=2
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
TRAIN_SEED=42


###############################################################################
# 离线评测参数
###############################################################################

EVAL_ENGINE="vllm"
# vLLM 的 tensor parallel size 需要能整除模型 attention heads。
# Qwen3-8B 的 attention heads 是 32，因此 7 卡 TP 不合法；
# 这里退回到 4 卡，是当前空闲卡里最稳妥、兼顾吞吐和兼容性的设置。
EVAL_GPU_IDS="1,2,3,4"
EVAL_TP_SIZE=4

# 固定 64 条子集，用来看 checkpoint 学习曲线。
CHECKPOINT_EVAL_MAX_SAMPLES=64

# 最终 full test 用 max_samples=0，表示评测整个 split。
FINAL_EVAL_MAX_SAMPLES=0


###############################################################################
# 阶段开关
###############################################################################

# 这些开关方便你重跑局部阶段。
# 例如只想重跑最终评测，可以把前面都改成 0，只保留 RUN_FINAL_EVAL=1。
RUN_TEACHER_GENERATE=0
RUN_STUDENT_SCORE=0
RUN_SELECT_AND_BUILD=0
RUN_ANALYZE_DATASET=0
RUN_TRAIN=1

# checkpoint 子集评测不是必须，但对看学习曲线很有帮助。
RUN_CHECKPOINT_EVAL=0
RUN_FINAL_EVAL=1


###############################################################################
# 文件名约定
###############################################################################

CANDIDATE_POOL_FILE="${CANDIDATE_DIR}/candidate_pool.jsonl"
SCORED_CANDIDATES_FILE="${CANDIDATE_DIR}/scored_candidates.jsonl"
DATASET_SUMMARY_FILE="${ANALYSIS_DIR}/dataset_summary.json"

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

train_one_mode() {
  local mode="$1"
  local port="$2"
  local train_file="${DATASET_DIR}/distill_${mode}.jsonl"
  local output_dir="${RUN_DIR}/${mode}"
  local train_log="${output_dir}/train.log"

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
      --rollout-max-new-tokens ${ROLLOUT_MAX_NEW_TOKENS} \
      --disable-rollout-eval \
      --disable-eval-preview \
      2>&1 | tee ${train_log}
  "
}

checkpoint_eval_one_mode() {
  local mode="$1"

  # 学习曲线阶段建议一个 checkpoint 一个进程去跑。
  # 这样更稳，不会踩到“同一进程里连续拉起多个 8 卡 vLLM engine”的生命周期问题。
  run_cmd bash -lc "
    CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDS} \
    ${CONDA_RUN} python src/pre_exp/final_eval.py \
      --engine ${EVAL_ENGINE} \
      --model-name-or-path ${RUN_DIR}/${mode}/checkpoint-step-000050 \
      --max-samples ${CHECKPOINT_EVAL_MAX_SAMPLES} \
      --subset-seed ${SUBSET_SEED} \
      --max-new-tokens ${ROLLOUT_MAX_NEW_TOKENS} \
      --tensor-parallel-size ${EVAL_TP_SIZE} \
      --trust-remote-code \
      --output-file ${ANALYSIS_DIR}/checkpoint_eval_${mode}_step50.json \
      > ${LOG_DIR}/checkpoint_eval_${mode}_step50.log 2>&1
  "

  run_cmd bash -lc "
    CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDS} \
    ${CONDA_RUN} python src/pre_exp/final_eval.py \
      --engine ${EVAL_ENGINE} \
      --model-name-or-path ${RUN_DIR}/${mode}/final_checkpoint \
      --max-samples ${CHECKPOINT_EVAL_MAX_SAMPLES} \
      --subset-seed ${SUBSET_SEED} \
      --max-new-tokens ${ROLLOUT_MAX_NEW_TOKENS} \
      --tensor-parallel-size ${EVAL_TP_SIZE} \
      --trust-remote-code \
      --output-file ${ANALYSIS_DIR}/checkpoint_eval_${mode}_final64.json \
      > ${LOG_DIR}/checkpoint_eval_${mode}_final64.log 2>&1
  "
}

final_eval_one_mode() {
  local mode="$1"

  # final eval 这里跑 GSM8K test 全量。
  # --max-samples=0 在当前脚本里表示“不抽子集，评测整个 split”。
  run_cmd bash -lc "
    CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDS} \
    ${CONDA_RUN} python src/pre_exp/final_eval.py \
      --engine ${EVAL_ENGINE} \
      --model-name-or-path ${RUN_DIR}/${mode}/final_checkpoint \
      --max-samples ${FINAL_EVAL_MAX_SAMPLES} \
      --subset-seed ${SUBSET_SEED} \
      --max-new-tokens ${ROLLOUT_MAX_NEW_TOKENS} \
      --tensor-parallel-size ${EVAL_TP_SIZE} \
      --trust-remote-code \
      --output-file ${ANALYSIS_DIR}/final_eval_${mode}.json \
      > ${LOG_DIR}/final_eval_${mode}.log 2>&1
  "
}


###############################################################################
# 阶段 A: Teacher 候选生成
###############################################################################

if [[ "${RUN_TEACHER_GENERATE}" == "1" ]]; then
  # 这里显式给 8 卡 tensor parallel，是为了把 teacher 生成速度尽量拉满。
  run_cmd bash -lc "
    CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDS} \
    ${CONDA_RUN} python src/pre_exp/teacher_generate.py \
      --model-name-or-path ${TEACHER_MODEL} \
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
      --tensor-parallel-size ${EVAL_TP_SIZE} \
      --enforce-eager \
      --trust-remote-code \
      --use-tqdm
  "
fi


###############################################################################
# 阶段 B: Student 打分
###############################################################################

if [[ "${RUN_STUDENT_SCORE}" == "1" ]]; then
  run_cmd bash -lc "
    CUDA_VISIBLE_DEVICES=${SCORE_VISIBLE_DEVICES} \
    ${CONDA_RUN} python src/pre_exp/student_score.py \
      --model-name-or-path ${STUDENT_MODEL} \
      --input-file ${CANDIDATE_POOL_FILE} \
      --output-file ${SCORED_CANDIDATES_FILE} \
      --batch-size ${SCORE_BATCH_SIZE} \
      --max-length ${SCORE_MAX_LENGTH} \
      --device ${SCORE_DEVICE}
  "
fi


###############################################################################
# 阶段 C: 候选选择 + 蒸馏数据构建
###############################################################################

if [[ "${RUN_SELECT_AND_BUILD}" == "1" ]]; then
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
  run_cmd ${CONDA_RUN} python src/pre_exp/analyze_dataset.py \
    --scored-candidates "${SCORED_CANDIDATES_FILE}" \
    --baseline-file "${SELECTION_DIR}/teacher_baseline.selected.jsonl" \
    --adversarial-file "${SELECTION_DIR}/teacher_adversarial.selected.jsonl" \
    --output-file "${DATASET_SUMMARY_FILE}"
fi


###############################################################################
# 阶段 E: 两组 SFT
###############################################################################

if [[ "${RUN_TRAIN}" == "1" ]]; then
  train_one_mode "teacher_baseline" "29601"
  train_one_mode "teacher_adversarial" "29603"
fi


###############################################################################
# 阶段 F: 中间 checkpoint 学习曲线评测（64 条固定子集）
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
  for mode in "${MODES[@]}"; do
    final_eval_one_mode "${mode}"
  done
fi


echo
echo "Smoke pre-experiment pipeline finished."
echo "Training runs:   ${RUN_DIR}"
echo "Analysis files:  ${ANALYSIS_DIR}"
