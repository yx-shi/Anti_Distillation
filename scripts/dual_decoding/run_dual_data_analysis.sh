#!/usr/bin/env bash

set -euo pipefail

CONDA_RUN="${HOME}/miniconda3/bin/conda run --no-capture-output -n adistill-unified"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

EXP_DIR="vllm_dual_deepscaler8000_k1_t0.7_p0.8_len4096"
ROOT="result/vllm_dual_decoding"

PLAIN_FILE="${ROOT}/candidates/${EXP_DIR}/plain_bs128_s128/scored_candidates.jsonl"
HARD_FILE="${ROOT}/candidates/${EXP_DIR}/hard_bs96_s80/scored_candidates.jsonl"
SOFT_FILE="${ROOT}/candidates/${EXP_DIR}/soft_bs96_s80/scored_candidates.jsonl"
OUTPUT_FILE="${ROOT}/analysis/${EXP_DIR}/data_quality_summary.json"

${CONDA_RUN} python src/vllm_dual_decoding/analyze_generation_modes.py \
  --mode-file "teacher_plain=${PLAIN_FILE}" \
  --mode-file "teacher_token_hard=${HARD_FILE}" \
  --mode-file "teacher_token_soft=${SOFT_FILE}" \
  --output-file "${OUTPUT_FILE}"
