#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

INPUT_DIR=""
PATTERN="scored_candidates.shard_*.jsonl"
OUTPUT_FILE=""
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/dual_decoding/merge_jsonl_files.sh --dir DIR --out PATH [--pattern GLOB]

Examples:
  bash scripts/dual_decoding/merge_jsonl_files.sh \
    --dir result/.../teacher_token_soft/shards \
    --pattern 'scored_candidates.shard_*.jsonl' \
    --out result/.../teacher_token_soft/scored_candidates.jsonl

Environment overrides:
  ALLOW_OVERWRITE=1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --pattern)
      PATTERN="$2"
      shift 2
      ;;
    --out)
      OUTPUT_FILE="$2"
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

if [[ -z "${INPUT_DIR}" || -z "${OUTPUT_FILE}" ]]; then
  echo "Missing --dir or --out." >&2
  usage >&2
  exit 1
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input dir does not exist: ${INPUT_DIR}" >&2
  exit 1
fi
if [[ "${ALLOW_OVERWRITE}" != "1" && -s "${OUTPUT_FILE}" ]]; then
  echo "Refusing to overwrite non-empty file: ${OUTPUT_FILE}" >&2
  echo "Set ALLOW_OVERWRITE=1 if you intentionally want to rerun." >&2
  exit 1
fi

mapfile -t INPUT_FILES < <(find "${INPUT_DIR}" -maxdepth 1 -type f -name "${PATTERN}" | sort)
if [[ "${#INPUT_FILES[@]}" -eq 0 ]]; then
  echo "No files matched pattern '${PATTERN}' in: ${INPUT_DIR}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_FILE}")"
: > "${OUTPUT_FILE}"

for input_file in "${INPUT_FILES[@]}"; do
  cat "${input_file}" >> "${OUTPUT_FILE}"
done

echo "merged ${#INPUT_FILES[@]} files -> ${OUTPUT_FILE}"
wc -l "${OUTPUT_FILE}"
