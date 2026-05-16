from __future__ import annotations

import unittest
from pathlib import Path

from src.vllm_dual_decoding.build_distill_dataset import (
    build_distill_records,
    summarize_distill_build,
)


def scored_record(
    sample_id: int,
    *,
    candidate_text: str,
    candidate_id: int = 0,
    generation_mode: str = "teacher_token_hard",
    nll: float | None = 0.5,
) -> dict:
    return {
        "sample_id": sample_id,
        "question": f"question-{sample_id}",
        "prompt_text": f"prompt-{sample_id}\n",
        "candidate_id": candidate_id,
        "candidate_text": candidate_text,
        "generation_mode": generation_mode,
        "adversarial_mode": "hard",
        "worker_cls": "vllm.worker.dual_worker.DualModelWorker",
        "is_correct": True,
        "is_valid_candidate": True,
        "is_generation_truncated": False,
        "student_mean_nll": nll,
        "student_token_count": 12 if nll is not None else 0,
        "score_truncated": False,
        "num_generated_tokens": 13,
    }


class VllmDualBuildDistillDatasetTest(unittest.TestCase):
    def test_build_distill_records_preserves_token_level_mode_metadata(self) -> None:
        records = [
            scored_record(7, candidate_text="  answer with space  "),
            scored_record(8, candidate_text=""),
        ]

        distill_records = build_distill_records(records, mode="teacher_token_hard")

        self.assertEqual(len(distill_records), 1)
        distill = distill_records[0]
        self.assertEqual(distill["sample_id"], 7)
        self.assertEqual(distill["prompt"], "prompt-7\n")
        self.assertEqual(distill["completion"], "answer with space")
        self.assertEqual(distill["selection_mode"], "teacher_token_hard")
        self.assertEqual(distill["generation_mode"], "teacher_token_hard")
        self.assertEqual(distill["selected_candidate_id"], 0)
        self.assertEqual(distill["teacher_candidate_count"], 1)
        self.assertEqual(distill["teacher_answer_correct"], True)
        self.assertEqual(distill["teacher_candidate_valid"], True)
        self.assertEqual(distill["teacher_generation_truncated"], False)
        self.assertEqual(distill["student_mean_nll"], 0.5)
        self.assertEqual(distill["worker_cls"], "vllm.worker.dual_worker.DualModelWorker")

    def test_build_distill_records_selects_first_candidate_per_sample(self) -> None:
        records = [
            scored_record(7, candidate_text="second", candidate_id=1, nll=1.0),
            scored_record(7, candidate_text="first", candidate_id=0, nll=0.5),
        ]

        distill_records = build_distill_records(records, mode="teacher_plain")

        self.assertEqual(len(distill_records), 1)
        self.assertEqual(distill_records[0]["completion"], "first")
        self.assertEqual(distill_records[0]["teacher_candidate_count"], 2)

    def test_summarize_distill_build_reports_skipped_empty_records(self) -> None:
        records = [
            scored_record(1, candidate_text="answer"),
            scored_record(2, candidate_text=""),
        ]
        distill_records = build_distill_records(records, mode="teacher_plain")

        summary = summarize_distill_build(
            mode="teacher_plain",
            input_file=Path("scored.jsonl"),
            output_file=Path("distill.jsonl"),
            input_records=records,
            distill_records=distill_records,
        )

        self.assertEqual(summary["input_record_count"], 2)
        self.assertEqual(summary["distill_record_count"], 1)
        self.assertEqual(summary["skipped_empty_completion_count"], 1)


if __name__ == "__main__":
    unittest.main()
