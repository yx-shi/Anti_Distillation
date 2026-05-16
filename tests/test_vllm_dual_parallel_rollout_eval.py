from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.vllm_dual_decoding.run_parallel_rollout_eval import (
    EvalJob,
    build_arg_parser,
    build_eval_command,
    build_plot_command,
    collect_eval_jobs,
    discover_eval_jobs_for_mode,
    gpu_batches,
)


class VllmDualParallelRolloutEvalTest(unittest.TestCase):
    def test_default_eval_max_num_seqs_is_128(self) -> None:
        args = build_arg_parser().parse_args([])

        self.assertEqual(args.max_num_seqs, 128)

    def test_discover_eval_jobs_includes_base_retained_checkpoints_and_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mode_dir = root / "teacher_plain"
            for name in (
                "checkpoint-step-000600",
                "checkpoint-step-000800",
                "checkpoint-step-001000",
                "checkpoint-step-001200",
                "checkpoint-step-001400",
                "final_checkpoint",
            ):
                (mode_dir / name).mkdir(parents=True)
            (mode_dir / "train_config.json").write_text('{"max_steps": 1600}\n', encoding="utf-8")

            jobs = discover_eval_jobs_for_mode(
                mode="teacher_plain",
                mode_dir=mode_dir,
                base_model="/models/student",
                analysis_dir=root / "analysis",
            )

        self.assertEqual(
            [(job.label, job.step, job.model_path.name if job.step else str(job.model_path)) for job in jobs],
            [
                ("000000", 0, "/models/student"),
                ("000600", 600, "checkpoint-step-000600"),
                ("000800", 800, "checkpoint-step-000800"),
                ("001000", 1000, "checkpoint-step-001000"),
                ("001200", 1200, "checkpoint-step-001200"),
                ("001400", 1400, "checkpoint-step-001400"),
                ("final_checkpoint", 1600, "final_checkpoint"),
            ],
        )
        self.assertEqual(jobs[0].output_file, root / "analysis" / "eval" / "checkpoint_eval_teacher_plain_000000.json")

    def test_gpu_batches_assigns_one_gpu_per_job(self) -> None:
        jobs = [
            EvalJob("teacher_plain", f"{idx:06d}", idx, Path(f"/m/{idx}"), Path(f"/o/{idx}.json"))
            for idx in range(5)
        ]

        batches = gpu_batches(jobs, ["0", "1"])

        self.assertEqual(
            [[(gpu, job.label) for gpu, job in batch] for batch in batches],
            [[("0", "000000"), ("1", "000001")], [("0", "000002"), ("1", "000003")], [("0", "000004")]],
        )

    def test_build_eval_command_sets_checkpoint_label_and_step_override(self) -> None:
        job = EvalJob(
            mode="teacher_token_hard",
            label="000600",
            step=600,
            model_path=Path("/runs/hard/checkpoint-step-000600"),
            output_file=Path("/analysis/checkpoint_eval_teacher_token_hard_000600.json"),
        )

        command = build_eval_command(
            conda_run=["conda", "run", "-n", "adistill-unified"],
            python_bin="python",
            job=job,
            dataset_name="agentica-org/DeepScaleR-Preview-Dataset",
            dataset_config_name="default",
            split="train",
            question_field="problem",
            answer_field="answer",
            max_samples=1024,
            exclude_subset_max_samples=8000,
            exclude_subset_seed=42,
            subset_seed=42,
            max_new_tokens=4096,
            num_rollouts=4,
            temperature=0.7,
            top_p=0.8,
            sampling_seed=42,
            tensor_parallel_size=1,
            max_model_len=8192,
            max_num_seqs=32,
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
        )

        self.assertIn("--model-name-or-path", command)
        self.assertIn("/runs/hard/checkpoint-step-000600", command)
        self.assertIn("--checkpoint-label-override", command)
        self.assertIn("000600", command)
        self.assertIn("--checkpoint-step-override", command)
        self.assertIn("600", command)
        self.assertIn("--output-file", command)
        self.assertIn("/analysis/checkpoint_eval_teacher_token_hard_000600.json", command)

    def test_collect_eval_jobs_flattens_modes_for_global_parallel_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs"
            analysis_dir = root / "analysis"
            for mode in ("teacher_plain", "teacher_token_hard"):
                mode_dir = run_dir / mode
                (mode_dir / "checkpoint-step-000200").mkdir(parents=True)
                (mode_dir / "final_checkpoint").mkdir()
                (mode_dir / "train_config.json").write_text('{"max_steps": 400}\n', encoding="utf-8")

            jobs_by_mode = collect_eval_jobs(
                modes=["teacher_plain", "teacher_token_hard"],
                run_dir=run_dir,
                base_model="/models/student",
                analysis_dir=analysis_dir,
            )

        flattened = [(job.mode, job.label, job.step) for jobs in jobs_by_mode.values() for job in jobs]
        self.assertEqual(
            flattened,
            [
                ("teacher_plain", "000000", 0),
                ("teacher_plain", "000200", 200),
                ("teacher_plain", "final_checkpoint", 400),
                ("teacher_token_hard", "000000", 0),
                ("teacher_token_hard", "000200", 200),
                ("teacher_token_hard", "final_checkpoint", 400),
            ],
        )

    def test_build_plot_command_writes_curves_under_analysis_dir(self) -> None:
        command = build_plot_command(
            conda_run=["conda", "run", "-n", "adistill-unified"],
            python_bin="python",
            run_dir=Path("/runs/exp"),
            analysis_dir=Path("/analysis/exp"),
            modes=["teacher_plain", "teacher_token_hard", "teacher_token_soft"],
        )

        self.assertEqual(
            command,
            [
                "conda",
                "run",
                "-n",
                "adistill-unified",
                "python",
                "src/pre_exp/plot_curves.py",
                "--run-dir",
                "/runs/exp",
                "--analysis-dir",
                "/analysis/exp/eval",
                "--output-dir",
                "/analysis/exp/curves",
                "--modes",
                "teacher_plain",
                "teacher_token_hard",
                "teacher_token_soft",
            ],
        )


if __name__ == "__main__":
    unittest.main()
