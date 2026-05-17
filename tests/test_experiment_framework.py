from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.experiment.config import load_experiment_config
from src.experiment.dataset_registry import get_dataset_spec
from src.experiment.launcher import ExperimentLauncher
from src.experiment.run_id import build_run_id


class DatasetRegistryTest(unittest.TestCase):
    def test_registry_defines_deepscaler_and_gsm8k_fields(self) -> None:
        deepscaler = get_dataset_spec("deepscaler")
        self.assertEqual(deepscaler.dataset_name, "agentica-org/DeepScaleR-Preview-Dataset")
        self.assertEqual(deepscaler.dataset_config_name, "default")
        self.assertEqual(deepscaler.train_split, "train")
        self.assertEqual(deepscaler.eval_split, "train")
        self.assertEqual(deepscaler.question_field, "problem")
        self.assertEqual(deepscaler.answer_field, "answer")
        self.assertEqual(deepscaler.completion_field, "solution")
        self.assertEqual(deepscaler.answer_extraction, "boxed")
        self.assertTrue(deepscaler.eval_exclude_train_subset)

        gsm8k = get_dataset_spec("gsm8k")
        self.assertEqual(gsm8k.dataset_name, "openai/gsm8k")
        self.assertEqual(gsm8k.dataset_config_name, "main")
        self.assertEqual(gsm8k.train_split, "train")
        self.assertEqual(gsm8k.eval_split, "test")
        self.assertEqual(gsm8k.question_field, "question")
        self.assertEqual(gsm8k.answer_field, "answer")
        self.assertEqual(gsm8k.completion_field, "answer")
        self.assertEqual(gsm8k.answer_extraction, "hash_fallback")
        self.assertFalse(gsm8k.eval_exclude_train_subset)


class RunIdTest(unittest.TestCase):
    def test_build_run_id_uses_na_for_missing_parameters(self) -> None:
        run_id = build_run_id(
            experiment_group="vllm_dual",
            dataset="deepscaler",
            mode="teacher_plain",
            hard_candidate_top_k=None,
            soft_student_weight=None,
            temperature=0.7,
            top_p=0.8,
            max_samples=8000,
            subset_seed=42,
        )

        self.assertEqual(
            run_id,
            "vllm_dual__ds-deepscaler__mode-teacher_plain__k-na__w-na"
            "__t-0.7__p-0.8__n-8000__seed-42",
        )

    def test_build_run_id_keeps_hard_and_soft_parameters_when_present(self) -> None:
        run_id = build_run_id(
            experiment_group="vllm_dual",
            dataset="gsm8k",
            mode="teacher_token_soft",
            hard_candidate_top_k=20,
            soft_student_weight=1.0,
            temperature=0.7,
            top_p=0.8,
            max_samples=128,
            subset_seed=7,
        )

        self.assertEqual(
            run_id,
            "vllm_dual__ds-gsm8k__mode-teacher_token_soft__k-20__w-1.0"
            "__t-0.7__p-0.8__n-128__seed-7",
        )


class ExperimentConfigTest(unittest.TestCase):
    def test_yaml_config_merges_dataset_defaults_and_mode_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "gsm8k.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment_group: smoke",
                        "dataset: gsm8k",
                        "modes:",
                        "  - teacher_plain",
                        "  - teacher_token_hard",
                        "paths:",
                        "  result_root: /tmp/adistill-results",
                        "  run_root: /tmp/adistill-runs",
                        "generation:",
                        "  max_samples: 12",
                        "  subset_seed: 7",
                        "  temperature: 0.5",
                        "  top_p: 0.75",
                        "  hard_candidate_top_k: 9",
                        "  soft_student_weight:",
                        "mode_overrides:",
                        "  teacher_token_hard:",
                        "    generation:",
                        "      max_num_seqs: 5",
                        "      prompt_batch_size: 3",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)

        self.assertEqual(config.dataset.key, "gsm8k")
        self.assertEqual(config.dataset.question_field, "question")
        self.assertEqual(config.dataset.answer_extraction, "hash_fallback")
        self.assertEqual(config.modes, ["teacher_plain", "teacher_token_hard"])
        self.assertEqual(config.paths["result_root"], "/tmp/adistill-results")

        plain_config = config.mode_config("teacher_plain")
        hard_config = config.mode_config("teacher_token_hard")
        self.assertEqual(plain_config["generation"]["max_samples"], 12)
        self.assertEqual(plain_config["generation"]["temperature"], 0.5)
        self.assertNotEqual(plain_config["generation"].get("max_num_seqs"), 5)
        self.assertEqual(hard_config["generation"]["max_num_seqs"], 5)
        self.assertEqual(hard_config["generation"]["prompt_batch_size"], 3)

        self.assertEqual(
            config.run_id_for_mode("teacher_token_hard"),
            "smoke__ds-gsm8k__mode-teacher_token_hard__k-9__w-na"
            "__t-0.5__p-0.75__n-12__seed-7",
        )


class LauncherDryRunPlanTest(unittest.TestCase):
    def test_teacher_generate_plan_uses_run_id_paths_and_existing_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "deepscaler.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment_group: smoke",
                        "dataset: deepscaler",
                        "modes: [teacher_plain]",
                        "paths:",
                        "  result_root: /tmp/adistill-results",
                        "  run_root: /tmp/adistill-runs",
                        "generation:",
                        "  max_samples: 4",
                        "  subset_seed: 42",
                        "  num_shards: 2",
                        "  gpu_ids: '0 1'",
                        "  temperature: 0.7",
                        "  top_p: 0.8",
                        "  hard_candidate_top_k:",
                        "  soft_student_weight:",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_experiment_config(config_path)

        launcher = ExperimentLauncher(config)
        plan = launcher.build_stage_plan("teacher_generate", "teacher_plain")
        run_id = config.run_id_for_mode("teacher_plain")

        self.assertEqual(
            plan.candidate_file,
            Path("/tmp/adistill-results") / "candidates" / run_id / "candidate_pool.jsonl",
        )
        self.assertEqual(len(plan.commands), 2)
        first = plan.commands[0]
        self.assertEqual(first.env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertIn("src/vllm_dual_decoding/teacher_generate.py", first.argv)
        self.assertIn("--dataset-key", first.argv)
        self.assertIn("deepscaler", first.argv)
        self.assertIn("--run-id", first.argv)
        self.assertIn(run_id, first.argv)
        self.assertIn("--generation-mode", first.argv)
        self.assertIn("plain", first.argv)
        self.assertNotIn("None", first.argv)
        self.assertNotIn("--hard-candidate-top-k", first.argv)
        self.assertNotIn("--hard-candidate-top-p", first.argv)
        self.assertNotIn("--soft-student-weight", first.argv)
        self.assertNotIn("--soft-temperature", first.argv)

    def test_plot_plan_wires_token_level_paths_to_curve_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "gsm8k.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment_group: smoke",
                        "dataset: gsm8k",
                        "modes: [teacher_plain]",
                        "paths:",
                        "  result_root: /tmp/adistill-results",
                        "  run_root: /tmp/adistill-runs",
                        "generation:",
                        "  max_samples: 8",
                        "  subset_seed: 123",
                        "  temperature: 0.7",
                        "  top_p: 0.8",
                        "  hard_candidate_top_k:",
                        "  soft_student_weight:",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_experiment_config(config_path)

        launcher = ExperimentLauncher(config)
        plan = launcher.build_stage_plan("curves", "teacher_plain")
        run_id = config.run_id_for_mode("teacher_plain")
        command = plan.commands[0]

        self.assertEqual(plan.stage, "plot")
        self.assertEqual(plan.paths.curve_dir, Path("/tmp/adistill-results") / "analysis" / run_id / "curves")
        self.assertIn("src/pre_exp/plot_curves.py", command.argv)
        self.assertIn("--mode-dir", command.argv)
        self.assertIn(f"teacher_plain=/tmp/adistill-runs/{run_id}", command.argv)
        self.assertIn("--analysis-file", command.argv)
        self.assertIn(
            f"teacher_plain=/tmp/adistill-results/analysis/{run_id}/final_eval.json",
            command.argv,
        )

    def test_rollout_eval_plan_includes_initial_intermediate_final_and_parallel_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "gsm8k.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment_group: smoke",
                        "dataset: gsm8k",
                        "modes: [teacher_plain]",
                        "models:",
                        "  student: /models/base-student",
                        "paths:",
                        f"  result_root: {root / 'results'}",
                        f"  run_root: {root / 'runs'}",
                        "generation:",
                        "  max_samples: 8",
                        "  subset_seed: 123",
                        "  temperature: 0.7",
                        "  top_p: 0.8",
                        "  hard_candidate_top_k:",
                        "  soft_student_weight:",
                        "rollout_eval:",
                        "  max_samples: 5",
                        "  checkpoint_glob: checkpoint-step-*",
                        "  include_initial: true",
                        "  include_final: true",
                        "  gpu_ids: '0 1'",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_experiment_config(config_path)
            run_id = config.run_id_for_mode("teacher_plain")
            run_dir = root / "runs" / run_id
            (run_dir / "checkpoint-step-200").mkdir(parents=True)
            (run_dir / "final_checkpoint").mkdir(parents=True)

            launcher = ExperimentLauncher(config)
            plan = launcher.build_stage_plan("rollout_eval", "teacher_plain")

        self.assertEqual(plan.stage, "rollout_eval")
        self.assertEqual(plan.paths.rollout_eval_summary_file, root / "results" / "analysis" / run_id / "checkpoint_eval.json")
        self.assertEqual(len(plan.commands), 3)
        output_args = [command.argv[command.argv.index("--output-file") + 1] for command in plan.commands]
        self.assertEqual(
            output_args,
            [
                str(root / "results" / "analysis" / run_id / "checkpoint_eval_000000.json"),
                str(root / "results" / "analysis" / run_id / "checkpoint_eval_200.json"),
                str(root / "results" / "analysis" / run_id / "checkpoint_eval_final_checkpoint.json"),
            ],
        )
        self.assertEqual([command.env["CUDA_VISIBLE_DEVICES"] for command in plan.commands], ["0", "1", "0"])
        first = plan.commands[0]
        self.assertIn("src/evaluation/rollout_eval.py", first.argv)
        self.assertIn("--checkpoint-label-override", first.argv)
        self.assertIn("000000", first.argv)
        self.assertIn("--checkpoint-step-override", first.argv)
        self.assertIn("0", first.argv)
        self.assertIn("/models/base-student", first.argv)

    def test_plot_plan_prefers_checkpoint_eval_files_over_final_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "gsm8k.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment_group: smoke",
                        "dataset: gsm8k",
                        "modes: [teacher_plain]",
                        "paths:",
                        f"  result_root: {root / 'results'}",
                        f"  run_root: {root / 'runs'}",
                        "generation:",
                        "  max_samples: 8",
                        "  subset_seed: 123",
                        "  temperature: 0.7",
                        "  top_p: 0.8",
                        "  hard_candidate_top_k:",
                        "  soft_student_weight:",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_experiment_config(config_path)
            run_id = config.run_id_for_mode("teacher_plain")
            analysis_dir = root / "results" / "analysis" / run_id
            analysis_dir.mkdir(parents=True)
            (analysis_dir / "checkpoint_eval_000000.json").write_text("{}", encoding="utf-8")
            (analysis_dir / "checkpoint_eval_200.json").write_text("{}", encoding="utf-8")
            (analysis_dir / "final_eval.json").write_text("{}", encoding="utf-8")

            launcher = ExperimentLauncher(config)
            plan = launcher.build_stage_plan("plot", "teacher_plain")
            argv = plan.commands[0].argv

        analysis_file_values = [
            argv[index + 1] for index, item in enumerate(argv) if item == "--analysis-file"
        ]
        self.assertEqual(len(analysis_file_values), 2)
        self.assertTrue(all("checkpoint_eval_" in value for value in analysis_file_values))
        self.assertTrue(all("final_eval.json" not in value for value in analysis_file_values))


if __name__ == "__main__":
    unittest.main()
