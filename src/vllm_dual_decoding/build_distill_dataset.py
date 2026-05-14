from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sft.prompting import normalize_completion_text
from vllm_dual_decoding.common import read_jsonl, write_json, write_jsonl


DEFAULT_INPUT_FILE = (
    "result/vllm_dual_decoding/candidates/vllm_dual_deepscaler8000_k1_t0.7_p0.8_len4096/"
    "plain_bs128_s128/scored_candidates.jsonl"
)
DEFAULT_OUTPUT_FILE = (
    "result/vllm_dual_decoding/datasets/vllm_dual_deepscaler8000_k1_t0.7_p0.8_len4096/"
    "distill_teacher_plain.jsonl"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert token-level vLLM-dual scored candidates into SFT-ready distill JSONL."
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument(
        "--summary-file",
        default="",
        help="Optional JSON summary path. Defaults to OUTPUT_FILE with .summary.json suffix.",
    )
    parser.add_argument(
        "--mode",
        default="",
        help=(
            "Token-level generation mode label. If omitted, read from the first record's "
            "`generation_mode` field."
        ),
    )
    parser.add_argument(
        "--candidate-policy",
        choices=["first", "highest_nll"],
        default="first",
        help=(
            "Policy used only if a scored file contains multiple candidates per sample. "
            "The formal token-level k=1 data uses `first`."
        ),
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Keep empty completions. Default skips them because they provide no SFT target tokens.",
    )
    return parser


def infer_mode(records: list[dict[str, Any]], explicit_mode: str) -> str:
    if explicit_mode:
        return explicit_mode
    for record in records:
        mode = record.get("generation_mode")
        if mode:
            return str(mode)
    raise ValueError("Unable to infer generation mode from records; pass --mode explicitly.")


def candidate_sort_key(record: dict[str, Any]) -> tuple[int, int]:
    candidate_id = record.get("candidate_id", 0)
    try:
        candidate_id_int = int(candidate_id)
    except (TypeError, ValueError):
        candidate_id_int = 0
    return candidate_id_int, int(record.get("sample_id", 0))


def choose_candidate(records: list[dict[str, Any]], candidate_policy: str) -> dict[str, Any]:
    if candidate_policy == "first":
        return sorted(records, key=candidate_sort_key)[0]
    if candidate_policy == "highest_nll":
        scored = [record for record in records if record.get("student_mean_nll") is not None]
        if scored:
            return max(scored, key=lambda record: float(record["student_mean_nll"]))
        return sorted(records, key=candidate_sort_key)[0]
    raise ValueError(f"Unsupported candidate policy: {candidate_policy}")


def build_distill_record(
    record: dict[str, Any],
    *,
    mode: str,
    teacher_candidate_count: int,
) -> dict[str, Any]:
    completion = normalize_completion_text(str(record.get("candidate_text", "")))
    candidate_id = record.get("candidate_id", 0)
    try:
        selected_candidate_id = int(candidate_id)
    except (TypeError, ValueError):
        selected_candidate_id = candidate_id

    return {
        "sample_id": record["sample_id"],
        "question": record["question"],
        "prompt": record["prompt_text"],
        "completion": completion,
        # `selection_mode` is kept for compatibility with the existing distill_jsonl schema.
        # For token-level experiments it directly names the generation mode, not a response selector.
        "selection_mode": mode,
        "generation_mode": mode,
        "selected_candidate_id": selected_candidate_id,
        "teacher_candidate_count": teacher_candidate_count,
        "teacher_answer_correct": bool(record.get("is_correct", False)),
        "teacher_candidate_valid": bool(record.get("is_valid_candidate", False)),
        "teacher_generation_truncated": bool(record.get("is_generation_truncated", False)),
        "student_mean_nll": record.get("student_mean_nll"),
        "student_token_count": int(record.get("student_token_count") or 0),
        "score_truncated": bool(record.get("score_truncated", False)),
        "fallback_reason": record.get("fallback_reason"),
        "adversarial_mode": record.get("adversarial_mode"),
        "worker_cls": record.get("worker_cls"),
        "num_generated_tokens": record.get("num_generated_tokens"),
        "source_candidate_id": record.get("candidate_id"),
        "source_finish_reason": record.get("finish_reason"),
        "source_stop_reason": record.get("stop_reason"),
    }


def build_distill_records(
    records: list[dict[str, Any]],
    *,
    mode: str,
    candidate_policy: str = "first",
    include_empty: bool = False,
) -> list[dict[str, Any]]:
    records_by_sample: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_sample[record.get("sample_id")].append(record)

    distill_records: list[dict[str, Any]] = []
    for sample_id in sorted(records_by_sample):
        sample_records = records_by_sample[sample_id]
        selected = choose_candidate(sample_records, candidate_policy)
        completion = normalize_completion_text(str(selected.get("candidate_text", "")))
        if not completion and not include_empty:
            continue
        distill_records.append(
            build_distill_record(
                selected,
                mode=mode,
                teacher_candidate_count=len(sample_records),
            )
        )
    return distill_records


def summarize_distill_build(
    *,
    mode: str,
    input_file: Path,
    output_file: Path,
    input_records: list[dict[str, Any]],
    distill_records: list[dict[str, Any]],
) -> dict[str, Any]:
    input_sample_ids = {record.get("sample_id") for record in input_records}
    distill_sample_ids = {record.get("sample_id") for record in distill_records}
    empty_input_count = sum(
        1
        for record in input_records
        if normalize_completion_text(str(record.get("candidate_text", ""))) == ""
    )
    return {
        "mode": mode,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "input_record_count": len(input_records),
        "input_sample_count": len(input_sample_ids),
        "distill_record_count": len(distill_records),
        "distill_sample_count": len(distill_sample_ids),
        "skipped_empty_completion_count": empty_input_count,
        "teacher_correct_count": sum(
            1 for record in distill_records if record.get("teacher_answer_correct", False)
        ),
        "teacher_valid_count": sum(
            1 for record in distill_records if record.get("teacher_candidate_valid", False)
        ),
        "teacher_truncated_count": sum(
            1 for record in distill_records if record.get("teacher_generation_truncated", False)
        ),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    records = read_jsonl(input_file)
    mode = infer_mode(records, args.mode)
    distill_records = build_distill_records(
        records,
        mode=mode,
        candidate_policy=args.candidate_policy,
        include_empty=args.include_empty,
    )

    write_jsonl(output_file, distill_records)
    summary_file = Path(args.summary_file) if args.summary_file else output_file.with_suffix(".summary.json")
    summary = summarize_distill_build(
        mode=mode,
        input_file=input_file,
        output_file=output_file,
        input_records=records,
        distill_records=distill_records,
    )
    write_json(summary_file, summary)
    print(
        "vllm_dual_build_distill_dataset_done "
        f"mode={mode} "
        f"records={len(distill_records)} "
        f"output_file={output_file} "
        f"summary_file={summary_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
