from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pre_exp.common import choose_holdout_indices, choose_subset_indices, write_json, write_jsonl
from sft.hf_cache import ensure_writable_hf_datasets_cache


DEFAULT_OUTPUT_FILE = (
    "result/pre_exp/datasets/deepscaler_main8000_k8_t0.9_p0.85_len4096/"
    "deepscaler_holdout_eval_1024_seed42.jsonl"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a fixed local eval JSONL from a dataset holdout split."
    )
    parser.add_argument("--dataset-name", default="agentica-org/DeepScaleR-Preview-Dataset")
    parser.add_argument("--dataset-config-name", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--question-field", default="problem")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument(
        "--completion-field",
        default="solution",
        help=(
            "Field used as the supervised completion for validation loss. "
            "For DeepScaleR, `solution` gives an in-domain reasoning target."
        ),
    )
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--summary-file", default="")
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--exclude-subset-max-samples", type=int, default=8000)
    parser.add_argument("--exclude-subset-seed", type=int, default=42)
    return parser


def require_field(sample: dict[str, Any], field: str, *, role: str) -> str:
    if field not in sample:
        available_fields = ", ".join(sorted(sample.keys()))
        raise KeyError(
            f"{role} field `{field}` is not present in dataset sample. "
            f"Available fields: {available_fields}"
        )
    return str(sample[field]).strip()


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_writable_hf_datasets_cache()
    dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.split)

    selected_indices = choose_holdout_indices(
        len(dataset),
        exclude_max_samples=args.exclude_subset_max_samples,
        exclude_seed=args.exclude_subset_seed,
        max_samples=args.max_samples,
        subset_seed=args.subset_seed,
    )

    records: list[dict[str, Any]] = []
    for dataset_idx in selected_indices:
        sample = dataset[int(dataset_idx)]
        question = require_field(sample, args.question_field, role="question")
        gold_answer = require_field(sample, args.answer_field, role="answer")
        completion = require_field(sample, args.completion_field, role="completion")
        records.append(
            {
                "question": question,
                "answer": completion,
                "source_dataset_idx": int(dataset_idx),
                "gold_answer": gold_answer,
                "dataset_name": args.dataset_name,
                "dataset_config_name": args.dataset_config_name,
                "split": args.split,
                "question_field": args.question_field,
                "answer_field": args.answer_field,
                "completion_field": args.completion_field,
            }
        )

    output_file = Path(args.output_file)
    write_jsonl(output_file, records)

    summary_file = Path(args.summary_file) if args.summary_file else output_file.with_suffix(".summary.json")
    excluded_count = (
        len(choose_subset_indices(len(dataset), args.exclude_subset_max_samples, args.exclude_subset_seed))
        if args.exclude_subset_max_samples > 0
        else 0
    )
    holdout_size = len(dataset) - excluded_count
    write_json(
        summary_file,
        {
            "dataset_name": args.dataset_name,
            "dataset_config_name": args.dataset_config_name,
            "split": args.split,
            "dataset_size": len(dataset),
            "excluded_sample_count": excluded_count,
            "holdout_size": holdout_size,
            "max_samples": args.max_samples,
            "actual_samples": len(records),
            "subset_seed": args.subset_seed,
            "exclude_subset_seed": args.exclude_subset_seed,
            "question_field": args.question_field,
            "answer_field": args.answer_field,
            "completion_field": args.completion_field,
            "output_file": str(output_file),
        },
    )
    print(
        "build_holdout_eval_dataset_done "
        f"records={len(records)} "
        f"output_file={output_file} "
        f"summary_file={summary_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
