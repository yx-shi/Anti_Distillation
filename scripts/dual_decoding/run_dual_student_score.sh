#!/usr/bin/env bash

set -euo pipefail

CONDA_RUN="${HOME}/miniconda3/bin/conda run --no-capture-output -n adistill-unified"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

MODEL="${MODEL:-/home/disk1/public_checkpoint/Qwen3-1.7B}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
DEVICE="${DEVICE:-cuda:0}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"

INPUT_FILE=""
INPUT_DIR=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/dual_decoding/run_dual_student_score.sh --file PATH
  bash scripts/dual_decoding/run_dual_student_score.sh --dir PATH

Environment overrides:
  MODEL=/path/to/student
  BATCH_SIZE=2
  MAX_LENGTH=8192
  ATTN_IMPLEMENTATION=flash_attention_2
  GPU_IDS="0 1 2 3 4 5 6 7"
  ALLOW_OVERWRITE=1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --file)
      INPUT_FILE="$2"
      shift 2
      ;;
    --dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "${INPUT_FILE}" && -n "${INPUT_DIR}" ]]; then
  echo "Use only one of --file or --dir." >&2
  exit 1
fi
if [[ -z "${INPUT_FILE}" && -z "${INPUT_DIR}" ]]; then
  echo "Missing --file or --dir." >&2
  usage >&2
  exit 1
fi

read -r -a GPU_ID_LIST <<< "${GPU_IDS//,/ }"
if [[ "${#GPU_ID_LIST[@]}" -eq 0 ]]; then
  echo "GPU_IDS must contain at least one GPU id." >&2
  exit 1
fi

output_for_file() {
  local input="$1"
  local dir
  local name
  dir="$(dirname "${input}")"
  name="$(basename "${input}")"
  if [[ "${name}" == candidate_pool* ]]; then
    echo "${dir}/${name/candidate_pool/scored_candidates}"
  else
    echo "${dir}/${name%.jsonl}.scored.jsonl"
  fi
}

guard_output_file() {
  local path="$1"
  if [[ "${ALLOW_OVERWRITE}" != "1" && -s "${path}" ]]; then
    echo "Refusing to overwrite non-empty file: ${path}" >&2
    echo "Set ALLOW_OVERWRITE=1 if you intentionally want to rerun." >&2
    exit 1
  fi
}

score_one_file() {
  local input="$1"
  local gpu_id="$2"
  local output
  output="$(output_for_file "${input}")"
  guard_output_file "${output}"

  echo
  echo "[$(date '+%F %T')] score input=${input} output=${output} gpu=${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  ${CONDA_RUN} python src/vllm_dual_decoding/student_score.py \
    --model-name-or-path "${MODEL}" \
    --input-file "${input}" \
    --output-file "${output}" \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}" \
    --device "${DEVICE}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}"
}

wait_for_batch() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    exit 1
  fi
}

if [[ -n "${INPUT_FILE}" ]]; then
  if [[ ! -s "${INPUT_FILE}" ]]; then
    echo "Input file is missing or empty: ${INPUT_FILE}" >&2
    exit 1
  fi
  score_one_file "${INPUT_FILE}" "${GPU_ID_LIST[0]}"
  exit 0
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input dir does not exist: ${INPUT_DIR}" >&2
  exit 1
fi

mapfile -t INPUT_FILES < <(find "${INPUT_DIR}" -maxdepth 1 -type f -name 'candidate_pool*.jsonl' | sort)
if [[ "${#INPUT_FILES[@]}" -eq 0 ]]; then
  echo "No candidate_pool*.jsonl files found in: ${INPUT_DIR}" >&2
  exit 1
fi

pids=()
for idx in "${!INPUT_FILES[@]}"; do
  gpu_id="${GPU_ID_LIST[$((idx % ${#GPU_ID_LIST[@]}))]}"
  score_one_file "${INPUT_FILES[$idx]}" "${gpu_id}" &
  pids+=("$!")

  if [[ "${#pids[@]}" -ge "${#GPU_ID_LIST[@]}" ]]; then
    wait_for_batch "${pids[@]}"
    pids=()
  fi
done

if [[ "${#pids[@]}" -gt 0 ]]; then
  wait_for_batch "${pids[@]}"
fi

echo
echo "student scoring finished for dir: ${INPUT_DIR}"
