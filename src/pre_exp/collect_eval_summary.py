from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CURRENT_FILE = Path(__file__).resolve()
SRC_ROOT = CURRENT_FILE.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pre_exp.common import write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect per-checkpoint eval JSON files into one summary.")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--input-files", nargs="+", required=True)
    return parser


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = build_arg_parser().parse_args()
    checkpoints: list[dict[str, Any]] = []
    shared: dict[str, Any] = {}

    for raw_path in args.input_files:
        path = Path(raw_path)
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object JSON in {path}.")
        if not shared:
            for key in (
                "engine",
                "dataset_name",
                "dataset_config_name",
                "split",
                "question_field",
                "answer_field",
                "max_new_tokens",
                "num_rollouts",
                "temperature",
                "top_p",
                "sampling_seed",
                "max_samples",
                "subset_seed",
                "exclude_subset_max_samples",
                "exclude_subset_seed",
                "eligible_sample_count",
            ):
                if key in payload:
                    shared[key] = payload[key]
        checkpoints.append(
            {
                "checkpoint_label": payload.get("checkpoint_label"),
                "checkpoint_step": payload.get("checkpoint_step"),
                "model_name_or_path": payload.get("model_name_or_path"),
                "output_file": str(path),
                "records_file": str(path.with_name(path.stem + ".records.jsonl")),
                "metrics": payload.get("metrics", {}),
            }
        )

    write_json(args.output_file, {**shared, "checkpoints": checkpoints})
    print(
        "collect_eval_summary_done "
        f"checkpoints={len(checkpoints)} "
        f"output_file={args.output_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
