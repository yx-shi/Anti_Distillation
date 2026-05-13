from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vllm_dual_decoding.common import read_jsonl, write_json


DEFAULT_OUTPUT_FILE = "result/vllm_dual_decoding/analysis/vllm_dual_data_smoke_gsm8k24/data_quality_summary.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize standalone token-level vLLM-dual data-side smoke outputs."
    )
    parser.add_argument(
        "--mode-file",
        action="append",
        default=[],
        metavar="MODE=PATH",
        help=(
            "Generation mode and scored_candidates JSONL path. "
            "Repeat this argument, e.g. --mode-file teacher_plain=/path/scored.jsonl."
        ),
    )
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    return parser


def parse_mode_files(raw_items: list[str]) -> dict[str, Path]:
    mode_files: dict[str, Path] = {}
    for raw_item in raw_items:
        if "=" not in raw_item:
            raise ValueError(f"`--mode-file` must use MODE=PATH format, got: {raw_item}")
        mode, path = raw_item.split("=", maxsplit=1)
        mode = mode.strip()
        path = path.strip()
        if not mode:
            raise ValueError(f"`--mode-file` has an empty mode name: {raw_item}")
        if not path:
            raise ValueError(f"`--mode-file` has an empty path: {raw_item}")
        mode_files[mode] = Path(path)
    if not mode_files:
        raise ValueError("At least one `--mode-file MODE=PATH` argument is required.")
    return mode_files


def boolean_rate(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(1 for item in records if item.get(key, False)) / len(records)


def summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def counter_summary(values: list[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


def summarize_mode(mode: str, path: Path) -> dict[str, Any]:
    records = read_jsonl(path)
    sample_ids = {int(item["sample_id"]) for item in records if "sample_id" in item}
    nll_values = [
        float(item["student_mean_nll"])
        for item in records
        if item.get("student_mean_nll") is not None
    ]
    token_counts = [
        float(item["student_token_count"])
        for item in records
        if item.get("student_token_count") is not None
    ]
    generated_tokens = [
        float(item["num_generated_tokens"])
        for item in records
        if item.get("num_generated_tokens") is not None
    ]

    return {
        "mode": mode,
        "input_file": str(path),
        "record_count": len(records),
        "sample_count": len(sample_ids),
        "observed_generation_modes": counter_summary(
            [item.get("generation_mode") for item in records if item.get("generation_mode") is not None]
        ),
        "worker_classes": counter_summary(
            [item.get("worker_cls") for item in records if item.get("worker_cls") is not None]
        ),
        "adversarial_modes": counter_summary(
            [item.get("adversarial_mode") for item in records if item.get("adversarial_mode") is not None]
        ),
        "candidate_quality": {
            "empty_candidate_rate": boolean_rate(records, "is_empty"),
            "generation_truncated_rate": boolean_rate(records, "is_generation_truncated"),
            "extractable_candidate_rate": boolean_rate(records, "is_extractable"),
            "valid_candidate_rate": boolean_rate(records, "is_valid_candidate"),
            "correct_candidate_rate": boolean_rate(records, "is_correct"),
            "scoreable_candidate_rate": (len(nll_values) / len(records)) if records else 0.0,
        },
        "length": {
            "num_generated_tokens": summarize_numeric(generated_tokens),
            "student_token_count": summarize_numeric(token_counts),
        },
        "student_nll": summarize_numeric(nll_values),
        "intervention_summary": {
            "source": "logs_only",
            "note": (
                "vLLM-dual hard/soft intervention counts are printed in runtime logs "
                "as ADISTILL_DUAL_ADVERSARIAL step lines; they are not currently "
                "serialized into candidate JSONL records."
            ),
        },
    }


def build_pairwise_nll_gaps(mode_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    plain = mode_summaries.get("teacher_plain")
    if plain is None:
        return {}

    gaps: dict[str, Any] = {}
    plain_records = read_jsonl(plain["input_file"])
    plain_by_sample = {
        int(item["sample_id"]): item
        for item in plain_records
        if item.get("student_mean_nll") is not None
    }

    for mode, summary in mode_summaries.items():
        if mode == "teacher_plain":
            continue
        records = read_jsonl(summary["input_file"])
        values: list[float] = []
        for item in records:
            sample_id = int(item["sample_id"])
            plain_item = plain_by_sample.get(sample_id)
            if plain_item is None or item.get("student_mean_nll") is None:
                continue
            values.append(float(item["student_mean_nll"]) - float(plain_item["student_mean_nll"]))
        gaps[f"{mode}_minus_teacher_plain"] = summarize_numeric(values)

    return gaps


def main() -> None:
    args = build_arg_parser().parse_args()
    mode_files = parse_mode_files(args.mode_file)
    mode_summaries = {
        mode: summarize_mode(mode, path)
        for mode, path in mode_files.items()
    }
    summary = {
        "modes": mode_summaries,
        "pairwise_student_nll_gaps": build_pairwise_nll_gaps(mode_summaries),
        "notes": [
            "This summary is for standalone vLLM-dual data-side smoke only; it does not include SFT checkpoints or eval metrics.",
            "teacher_plain / teacher_token_hard / teacher_token_soft are independent token-level generation modes, not response-level selection modes.",
        ],
    }

    write_json(args.output_file, summary)
    print(
        "vllm_dual_analyze_generation_modes_done "
        f"modes={','.join(mode_summaries)} "
        f"output_file={args.output_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
