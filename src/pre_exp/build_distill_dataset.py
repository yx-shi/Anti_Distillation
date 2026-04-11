from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pre_exp.common import read_jsonl, write_jsonl


DEFAULT_SELECTION_FILE = "result/pre_exp/datasets/smoke/selections/teacher_baseline.selected.jsonl"
DEFAULT_OUTPUT_FILE = "result/pre_exp/datasets/smoke/distill_teacher_baseline.jsonl"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert selected candidates into SFT-ready distill JSONL.")
    parser.add_argument("--selection-file", default=DEFAULT_SELECTION_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    selection_records = read_jsonl(args.selection_file)

    distill_records: list[dict[str, object]] = []
    for record in selection_records:
        distill_records.append(
            {
                "sample_id": record["sample_id"],
                "question": record["question"],
                "prompt": record["prompt_text"],
                "completion": record["completion"],
                "selection_mode": record["selection_mode"],
                "selected_candidate_id": record["selected_candidate_id"],
                "teacher_candidate_count": record["teacher_candidate_count"],
                "teacher_answer_correct": record["teacher_answer_correct"],
                "student_mean_nll": record["student_mean_nll"],
                "fallback_reason": record["fallback_reason"],
            }
        )

    write_jsonl(args.output_file, distill_records)
    print(
        "build_distill_dataset_done "
        f"records={len(distill_records)} "
        f"output_file={args.output_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
