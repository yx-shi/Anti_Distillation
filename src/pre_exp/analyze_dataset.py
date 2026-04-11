from __future__ import annotations

import argparse
import statistics
import sys
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
DEFAULT_RANDOM_FILE = "result/pre_exp/datasets/smoke/selections/teacher_random_from_k.selected.jsonl"
DEFAULT_ADVERSARIAL_FILE = "result/pre_exp/datasets/smoke/selections/teacher_adversarial.selected.jsonl"
DEFAULT_OUTPUT_FILE = "result/pre_exp/analysis/smoke/dataset_summary.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze scored/selected datasets for the smoke pre-experiment.")
    parser.add_argument("--scored-candidates", default=DEFAULT_SCORED_FILE)
    parser.add_argument("--baseline-file", default=DEFAULT_BASELINE_FILE)
    parser.add_argument("--random-file", default=DEFAULT_RANDOM_FILE)
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


def main() -> None:
    args = build_arg_parser().parse_args()

    scored_records = read_jsonl(args.scored_candidates)
    baseline_records = read_jsonl(args.baseline_file)
    random_records = read_jsonl(args.random_file)
    adversarial_records = read_jsonl(args.adversarial_file)

    total_candidates = len(scored_records)
    valid_candidates = [item for item in scored_records if item.get("is_valid_candidate", False)]
    correct_candidates = [item for item in scored_records if item.get("is_correct", False)]

    baseline_by_sample = {int(item["sample_id"]): item for item in baseline_records}
    adversarial_by_sample = {int(item["sample_id"]): item for item in adversarial_records}

    nll_gaps: list[float] = []
    for sample_id, baseline_record in baseline_by_sample.items():
        adversarial_record = adversarial_by_sample.get(sample_id)
        if adversarial_record is None:
            continue
        baseline_nll = baseline_record.get("student_mean_nll")
        adversarial_nll = adversarial_record.get("student_mean_nll")
        if baseline_nll is None or adversarial_nll is None:
            continue
        nll_gaps.append(float(adversarial_nll) - float(baseline_nll))

    summary = {
        "candidate_pool": {
            "total_candidates": total_candidates,
            "valid_candidate_rate": (len(valid_candidates) / total_candidates) if total_candidates else 0.0,
            "correct_candidate_rate": (len(correct_candidates) / total_candidates) if total_candidates else 0.0,
        },
        "selection_modes": {
            "teacher_baseline": {
                "fallback_rate": fallback_rate(baseline_records),
                "completion_token_count": summarize_numeric(
                    [float(item.get("student_token_count", 0)) for item in baseline_records]
                ),
            },
            "teacher_random_from_k": {
                "fallback_rate": fallback_rate(random_records),
                "completion_token_count": summarize_numeric(
                    [float(item.get("student_token_count", 0)) for item in random_records]
                ),
            },
            "teacher_adversarial": {
                "fallback_rate": fallback_rate(adversarial_records),
                "completion_token_count": summarize_numeric(
                    [float(item.get("student_token_count", 0)) for item in adversarial_records]
                ),
            },
        },
        "baseline_vs_adversarial": {
            "avg_student_mean_nll_gap": summarize_numeric(nll_gaps),
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
