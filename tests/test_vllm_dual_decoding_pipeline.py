from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.vllm_dual_decoding.build_distill_dataset import (
    build_distill_records,
    summarize_distill_build,
)
from src.vllm_dual_decoding.run_full_pipeline import (
    ModeInput,
    discover_mode_inputs,
    mode_to_safe_port_offset,
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


class VllmDualRunFullPipelineTest(unittest.TestCase):
    def test_discover_mode_inputs_accepts_legacy_mode_directory_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "candidates" / "exp"
            for dirname in ("plain_bs128_s128", "hard_bs96_s80", "soft_bs96_s80"):
                mode_dir = experiment_dir / dirname
                mode_dir.mkdir(parents=True)
                (mode_dir / "scored_candidates.jsonl").write_text("{}\n", encoding="utf-8")

            mode_inputs = discover_mode_inputs(Path(tmpdir), "exp")

        self.assertEqual(
            mode_inputs,
            [
                ModeInput("teacher_plain", experiment_dir / "plain_bs128_s128" / "scored_candidates.jsonl"),
                ModeInput("teacher_token_hard", experiment_dir / "hard_bs96_s80" / "scored_candidates.jsonl"),
                ModeInput("teacher_token_soft", experiment_dir / "soft_bs96_s80" / "scored_candidates.jsonl"),
            ],
        )

    def test_mode_to_safe_port_offset_is_stable(self) -> None:
        self.assertEqual(mode_to_safe_port_offset("teacher_plain"), 0)
        self.assertEqual(mode_to_safe_port_offset("teacher_token_hard"), 2)
        self.assertEqual(mode_to_safe_port_offset("teacher_token_soft"), 4)


if __name__ == "__main__":
    unittest.main()
