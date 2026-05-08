from __future__ import annotations

import argparse
import statistics
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

from pre_exp.common import read_jsonl, write_json


DEFAULT_SCORED_FILE = "result/pre_exp/candidates/smoke/scored_candidates.jsonl"
DEFAULT_BASELINE_FILE = "result/pre_exp/datasets/smoke/selections/teacher_baseline.selected.jsonl"
DEFAULT_ADVERSARIAL_FILE = "result/pre_exp/datasets/smoke/selections/teacher_adversarial.selected.jsonl"
DEFAULT_OUTPUT_FILE = "result/pre_exp/analysis/smoke/dataset_summary.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze scored/selected datasets for the smoke pre-experiment.")
    parser.add_argument("--scored-candidates", default=DEFAULT_SCORED_FILE)
    parser.add_argument("--baseline-file", default=DEFAULT_BASELINE_FILE)
    parser.add_argument("--adversarial-file", default=DEFAULT_ADVERSARIAL_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    return parser


def summarize_numeric(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": float(len(values)),
        "mean": float(statistics.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def fallback_rate(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    fallback_count = sum(1 for item in records if item.get("fallback_reason"))
    return fallback_count / len(records)


def boolean_rate(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(1 for item in records if item.get(key, False)) / len(records)


def rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def group_by_sample(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["sample_id"])].append(record)
    return dict(grouped)


def numeric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        values.append(float(value))
    return values


def summarize_selection(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "fallback_rate": fallback_rate(records),
        "selected_correct_rate": boolean_rate(records, "teacher_answer_correct"),
        "selected_valid_rate": boolean_rate(records, "teacher_candidate_valid"),
        "selected_generation_truncated_rate": boolean_rate(
            records,
            "teacher_generation_truncated",
        ),
        "completion_token_count": summarize_numeric(numeric_values(records, "student_token_count")),
        "student_mean_nll": summarize_numeric(numeric_values(records, "student_mean_nll")),
    }


def main() -> None:
    args = build_arg_parser().parse_args()

    scored_records = read_jsonl(args.scored_candidates)
    baseline_records = read_jsonl(args.baseline_file)
    adversarial_records = read_jsonl(args.adversarial_file)

    total_candidates = len(scored_records)
    valid_candidates = [item for item in scored_records if item.get("is_valid_candidate", False)]
    correct_candidates = [item for item in scored_records if item.get("is_correct", False)]
    scoreable_candidates = [item for item in scored_records if item.get("student_mean_nll") is not None]
    grouped_candidates = group_by_sample(scored_records)

    baseline_by_sample = {int(item["sample_id"]): item for item in baseline_records}
    adversarial_by_sample = {int(item["sample_id"]): item for item in adversarial_records}

    nll_gaps: list[float] = []
    token_count_gaps: list[float] = []
    different_candidate_count = 0
    for sample_id, baseline_record in baseline_by_sample.items():
        adversarial_record = adversarial_by_sample.get(sample_id)
        if adversarial_record is None:
            continue
        if baseline_record.get("selected_candidate_id") != adversarial_record.get("selected_candidate_id"):
            different_candidate_count += 1
        baseline_nll = baseline_record.get("student_mean_nll")
        adversarial_nll = adversarial_record.get("student_mean_nll")
        if baseline_nll is None or adversarial_nll is None:
            continue
        nll_gaps.append(float(adversarial_nll) - float(baseline_nll))

        baseline_tokens = baseline_record.get("student_token_count")
        adversarial_tokens = adversarial_record.get("student_token_count")
        if baseline_tokens is not None and adversarial_tokens is not None:
            token_count_gaps.append(float(adversarial_tokens) - float(baseline_tokens))

    sample_count = len(grouped_candidates)
    samples_with_valid = 0
    samples_with_correct = 0
    samples_with_valid_and_correct = 0
    samples_with_any_truncated = 0
    samples_with_all_truncated = 0
    samples_with_all_empty = 0
    samples_with_all_score_truncated = 0
    for candidates in grouped_candidates.values():
        has_valid = any(item.get("is_valid_candidate", False) for item in candidates)
        has_correct = any(item.get("is_correct", False) for item in candidates)
        has_valid_and_correct = any(
            item.get("is_valid_candidate", False) and item.get("is_correct", False)
            for item in candidates
        )
        has_any_truncated = any(item.get("is_generation_truncated", False) for item in candidates)
        has_all_truncated = all(item.get("is_generation_truncated", False) for item in candidates)
        has_all_empty = all(item.get("is_empty", False) for item in candidates)
        has_all_score_truncated = all(item.get("score_truncated", False) for item in candidates)

        samples_with_valid += int(has_valid)
        samples_with_correct += int(has_correct)
        samples_with_valid_and_correct += int(has_valid_and_correct)
        samples_with_any_truncated += int(has_any_truncated)
        samples_with_all_truncated += int(has_all_truncated)
        samples_with_all_empty += int(has_all_empty)
        samples_with_all_score_truncated += int(has_all_score_truncated)

    summary = {
        "candidate_pool": {
            "total_candidates": total_candidates,
            "empty_candidate_rate": boolean_rate(scored_records, "is_empty"),
            "generation_truncated_rate": boolean_rate(scored_records, "is_generation_truncated"),
            "score_truncated_rate": boolean_rate(scored_records, "score_truncated"),
            "extractable_candidate_rate": boolean_rate(scored_records, "is_extractable"),
            "valid_candidate_rate": (len(valid_candidates) / total_candidates) if total_candidates else 0.0,
            "correct_candidate_rate": (len(correct_candidates) / total_candidates) if total_candidates else 0.0,
            "scoreable_candidate_rate": (
                len(scoreable_candidates) / total_candidates if total_candidates else 0.0
            ),
        },
        "sample_pool": {
            "total_samples": sample_count,
            "at_least_one_valid_candidate_rate": rate(samples_with_valid, sample_count),
            "at_least_one_correct_candidate_rate": rate(samples_with_correct, sample_count),
            "at_least_one_valid_and_correct_candidate_rate": rate(
                samples_with_valid_and_correct,
                sample_count,
            ),
            "any_generation_truncated_candidate_rate": rate(samples_with_any_truncated, sample_count),
            "all_generation_truncated_candidate_rate": rate(samples_with_all_truncated, sample_count),
            "all_empty_candidate_rate": rate(samples_with_all_empty, sample_count),
            "all_score_truncated_candidate_rate": rate(samples_with_all_score_truncated, sample_count),
        },
        "selection_modes": {
            "teacher_baseline": summarize_selection(baseline_records),
            "teacher_adversarial": summarize_selection(adversarial_records),
        },
        "baseline_vs_adversarial": {
            "different_selected_candidate_rate": (
                different_candidate_count / len(baseline_by_sample) if baseline_by_sample else 0.0
            ),
            "avg_student_mean_nll_gap": summarize_numeric(nll_gaps),
            "avg_completion_token_count_gap": summarize_numeric(token_count_gaps),
        },
    }

    write_json(args.output_file, summary)
    print(
        "analyze_dataset_done "
        f"total_candidates={total_candidates} "
        f"output_file={args.output_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
