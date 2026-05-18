from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.experiment.config import load_experiment_config
from src.experiment.final_summary import build_final_result_summary, write_final_result_summary
from src.vllm_dual_decoding.common import write_json


def eval_payload(step: int | None, label: str, acc: float, std: float, total: float = 8.0) -> dict:
    return {
        "checkpoint_label": label,
        "checkpoint_step": step,
        "max_samples": int(total),
        "metrics": {
            "rollout_acc": acc,
            "rollout_acc_std": std,
            "rollout_acc_variance": std * std,
            "rollout_correct": acc * total,
            "rollout_total": total,
            "num_rollouts": 4.0,
        },
    }


class FinalSummaryTest(unittest.TestCase):
    def test_final_summary_writes_group_curves_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "gsm8k.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment_group: smoke",
                        "dataset: gsm8k",
                        "modes: [teacher_plain, teacher_token_hard, teacher_token_soft]",
                        "paths:",
                        f"  result_root: {root / 'results'}",
                        f"  run_root: {root / 'runs'}",
                        f"  summary_root: {root / 'summary'}",
                        "generation:",
                        "  max_samples: 8",
                        "  subset_seed: 123",
                        "  temperature: 0.7",
                        "  top_p: 0.8",
                        "  hard_candidate_top_k:",
                        "  soft_student_weight:",
                        "mode_overrides:",
                        "  teacher_token_hard:",
                        "    generation:",
                        "      hard_candidate_top_k: 20",
                        "  teacher_token_soft:",
                        "    generation:",
                        "      hard_candidate_top_k: 20",
                        "      soft_student_weight: 0.6",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_experiment_config(config_path)
            modes = config.modes
            group_run_id = config.group_run_id_for_modes(modes)
            summary_dir = root / "summary" / group_run_id
            write_json(
                summary_dir / "data_quality_summary.json",
                {
                    "group_run_id": group_run_id,
                    "modes": {
                        mode: {
                            "record_count": 8,
                            "correct_rate": 0.75,
                            "valid_rate": 1.0,
                            "student_mean_nll": {"mean": 0.2},
                            "num_generated_tokens": {"mean": 12.0},
                        }
                        for mode in modes
                    },
                    "comparisons": {},
                },
            )
            for idx, mode in enumerate(modes):
                run_id = config.run_id_for_mode(mode)
                run_dir = root / "runs" / run_id
                analysis_dir = root / "results" / "analysis" / run_id
                run_dir.mkdir(parents=True)
                analysis_dir.mkdir(parents=True)
                (run_dir / "train.log").write_text(
                    "\n".join(
                        [
                            "epoch=0 step=1 train_loss=0.9 lr=1e-5",
                            "[eval] step=1 val_loss=1.1 val_ppl=3.0",
                            "epoch=0 step=2 train_loss=0.7 lr=0",
                            "training_done steps=2 val_loss=1.0 val_ppl=2.7",
                        ]
                    ),
                    encoding="utf-8",
                )
                write_json(analysis_dir / "checkpoint_eval_000000.json", eval_payload(0, "000000", 0.5, 0.1))
                write_json(analysis_dir / "checkpoint_eval_000100.json", eval_payload(100, "000100", 0.6 + idx * 0.05, 0.08))
                write_json(analysis_dir / "checkpoint_eval_final_checkpoint.json", eval_payload(None, "final_checkpoint", 0.55 + idx * 0.02, 0.05))
                write_json(analysis_dir / "final_eval.json", eval_payload(None, "final_checkpoint", 0.8 + idx * 0.01, 0.04, total=10.0))

            summary, curve_data = build_final_result_summary(config=config, modes=modes)
            outputs = write_final_result_summary(summary, curve_data, summary_dir)

            self.assertEqual(summary["group_run_id"], group_run_id)
            self.assertEqual(summary["modes"]["teacher_plain"]["final_eval"]["metrics"]["rollout_acc"], 0.8)
            self.assertEqual(summary["modes"]["teacher_token_soft"]["rollout_eval"]["best"]["checkpoint_step"], 100)
            self.assertTrue(outputs["json_file"].is_file())
            self.assertTrue(outputs["markdown_file"].is_file())
            self.assertTrue((summary_dir / "train_loss_curve.svg").is_file())
            self.assertTrue((summary_dir / "val_loss_ppl_curve.svg").is_file())
            self.assertTrue((summary_dir / "rollout_accuracy_curve.svg").is_file())
            self.assertIn("class=\"errorbar\"", (summary_dir / "rollout_accuracy_curve.svg").read_text(encoding="utf-8"))
            markdown = outputs["markdown_file"].read_text(encoding="utf-8")
            self.assertIn("# Final Result Summary", markdown)
            self.assertIn("teacher_token_hard", markdown)
            self.assertIn("rollout_accuracy_curve.svg", markdown)


if __name__ == "__main__":
    unittest.main()
