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

from pre_exp.common import read_jsonl, write_jsonl


DEFAULT_INPUT_FILE = "result/pre_exp/candidates/smoke/scored_candidates.jsonl"
DEFAULT_OUTPUT_DIR = "result/pre_exp/datasets/smoke/selections"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select baseline/adversarial candidates from scored teacher outputs.")
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def sort_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda item: int(item.get("candidate_id", 0)))


def first_teacher_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """选择 Teacher 采样返回的第一条候选，不按质量字段过滤。"""

    if not candidates:
        raise ValueError("No candidate is available for this sample.")
    return sort_candidates(candidates)[0]


def build_selection_record(
    selected: dict[str, Any],
    *,
    selection_mode: str,
    teacher_candidate_count: int,
    fallback_reason: str,
) -> dict[str, Any]:
    """把被选中的候选转换成后续构建 distill JSONL 所需的结构。"""

    completion_text = str(selected.get("candidate_text_clean") or selected.get("candidate_text") or "").strip()
    return {
        "sample_id": selected["sample_id"],
        "question": selected["question"],
        "prompt_text": selected["prompt_text"],
        "completion": completion_text,
        "selection_mode": selection_mode,
        "selected_candidate_id": selected["candidate_id"],
        "teacher_candidate_count": teacher_candidate_count,
        "teacher_answer_correct": bool(selected.get("is_correct", False)),
        "teacher_candidate_valid": bool(selected.get("is_valid_candidate", False)),
        "teacher_generation_truncated": bool(selected.get("is_generation_truncated", False)),
        "student_mean_nll": selected.get("student_mean_nll"),
        "student_token_count": int(selected.get("student_token_count", 0)),
        "fallback_reason": fallback_reason,
    }


def choose_adversarial(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    scoreable_candidates = [item for item in candidates if item.get("student_mean_nll") is not None]
    if not scoreable_candidates:
        raise ValueError("No scoreable candidate is available for adversarial selection.")

    # 完整 Teacher 分布蒸馏口径：不按正误、截断或可抽取性过滤，只选择
    # Student completion-token mean NLL 最大的 Teacher 样本。
    # 如果 NLL 并列，再按 candidate_id 升序打破平局，保证结果可复现。
    return max(
        scoreable_candidates,
        key=lambda item: (
            float(item["student_mean_nll"]),
            -int(item.get("candidate_id", 0)),
        ),
    )


def main() -> None:
    args = build_arg_parser().parse_args()

    records = read_jsonl(args.input_file)
    grouped_candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped_candidates[int(record["sample_id"])].append(record)

    baseline_records: list[dict[str, Any]] = []
    adversarial_records: list[dict[str, Any]] = []

    for sample_id in sorted(grouped_candidates):
        candidates = sort_candidates(grouped_candidates[sample_id])
        teacher_candidate_count = len(candidates)

        baseline_selected = first_teacher_candidate(candidates)
        baseline_records.append(
            build_selection_record(
                baseline_selected,
                selection_mode="teacher_baseline",
                teacher_candidate_count=teacher_candidate_count,
                fallback_reason="",
            )
        )

        try:
            adversarial_selected = choose_adversarial(candidates)
            adversarial_fallback_reason = ""
        except ValueError:
            # 理论上只有所有 Teacher completion 都为空时才会触发。此时没有 NLL
            # 可比较，退回到第一条 Teacher 样本以保持数据规模对齐。
            adversarial_selected = baseline_selected
            adversarial_fallback_reason = "no_scoreable_candidate"

        adversarial_records.append(
            build_selection_record(
                adversarial_selected,
                selection_mode="teacher_adversarial",
                teacher_candidate_count=teacher_candidate_count,
                fallback_reason=adversarial_fallback_reason,
            )
        )

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "teacher_baseline.selected.jsonl", baseline_records)
    write_jsonl(output_dir / "teacher_adversarial.selected.jsonl", adversarial_records)

    print(
        "select_candidates_done "
        f"samples={len(grouped_candidates)} "
        f"output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
