from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    MODE_TO_GENERATION_MODE,
    ExperimentConfig,
    normalize_mode,
    normalize_stage,
)
from .paths import ExperimentPaths, paths_for_mode
from src.vllm_dual_decoding.build_distill_dataset import (
    build_distill_records,
    summarize_distill_build,
)
from src.vllm_dual_decoding.common import read_jsonl, write_json, write_jsonl


@dataclass(frozen=True)
class CommandSpec:
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    log_file: Path | None = None


@dataclass(frozen=True)
class StagePlan:
    stage: str
    mode: str
    run_id: str
    paths: ExperimentPaths
    commands: list[CommandSpec]

    @property
    def candidate_file(self) -> Path:
        return self.paths.candidate_file

    @property
    def scored_file(self) -> Path:
        return self.paths.scored_file


def parse_gpu_ids(raw_gpu_ids: Any) -> list[str]:
    gpu_ids = [item.strip() for item in str(raw_gpu_ids).replace(",", " ").split() if item.strip()]
    if not gpu_ids:
        raise ValueError("GPU id list must contain at least one id.")
    return gpu_ids


def command_prefix(raw_conda_run: str) -> list[str]:
    return shlex.split(raw_conda_run) if raw_conda_run.strip() else []


def printable_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def guard_output_file(path: Path, *, allow_overwrite: bool) -> None:
    if allow_overwrite:
        return
    if path.exists() and path.stat().st_size > 0:
        raise FileExistsError(
            f"Refusing to overwrite non-empty file: {path}. "
            "Pass --allow-overwrite to rerun intentionally."
        )


def _append_bool_flag(argv: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        argv.append(flag)


def _append_option(argv: list[str], flag: str, value: Any) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _base_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "VLLM_USE_V1": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
    }
    library_path = os.environ.get("LIBRARY_PATH")
    env["LIBRARY_PATH"] = (
        "/usr/lib/x86_64-linux-gnu/stubs"
        + (f":{library_path}" if library_path else "")
    )
    if extra:
        env.update(extra)
    return env


class ExperimentLauncher:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def build_stage_plan(self, stage: str, mode: str) -> StagePlan:
        normalized_stage = normalize_stage(stage)
        normalized_mode = normalize_mode(mode)
        if normalized_stage == "teacher_generate":
            return self._teacher_generate_plan(normalized_mode)
        if normalized_stage == "student_score":
            return self._student_score_plan(normalized_mode)
        if normalized_stage == "build_distill":
            return self._build_distill_plan(normalized_mode)
        if normalized_stage == "train":
            return self._train_plan(normalized_mode)
        if normalized_stage == "eval":
            return self._eval_plan(normalized_mode)
        if normalized_stage == "plot":
            return self._plot_plan(normalized_mode)
        raise ValueError(f"Unsupported stage: {stage}")

    def run_stage(
        self,
        stage: str,
        mode: str,
        *,
        dry_run: bool = False,
        allow_overwrite: bool = False,
    ) -> StagePlan:
        plan = self.build_stage_plan(stage, mode)
        if dry_run:
            self.print_plan(plan)
            return plan

        if plan.stage == "teacher_generate":
            self._run_teacher_generate(plan, allow_overwrite=allow_overwrite)
        elif plan.stage == "student_score":
            self._run_single_command_stage(plan, allow_overwrite=allow_overwrite, outputs=[plan.paths.scored_file])
        elif plan.stage == "build_distill":
            self._run_build_distill(plan, allow_overwrite=allow_overwrite)
        elif plan.stage == "train":
            self._run_single_command_stage(plan, allow_overwrite=True, outputs=[])
        elif plan.stage == "eval":
            self._run_single_command_stage(plan, allow_overwrite=allow_overwrite, outputs=[plan.paths.final_eval_file])
        elif plan.stage == "plot":
            self._run_single_command_stage(
                plan,
                allow_overwrite=allow_overwrite,
                outputs=[
                    plan.paths.curve_dir / "curve_data.json",
                    plan.paths.curve_dir / "train_loss_curve.svg",
                    plan.paths.curve_dir / "val_loss_ppl_curve.svg",
                    plan.paths.curve_dir / "rollout_accuracy_curve.svg",
                ],
            )
        else:
            raise ValueError(f"Unsupported stage: {plan.stage}")
        return plan

    def print_plan(self, plan: StagePlan) -> None:
        print(
            "experiment_stage_plan "
            f"stage={plan.stage} mode={plan.mode} run_id={plan.run_id} "
            f"candidate_file={plan.paths.candidate_file} "
            f"scored_file={plan.paths.scored_file} "
            f"distill_file={plan.paths.distill_file} "
            f"run_dir={plan.paths.run_dir} "
            f"analysis_dir={plan.paths.analysis_dir} "
            f"curve_dir={plan.paths.curve_dir}",
            flush=True,
        )
        for command in plan.commands:
            env_text = " ".join(f"{key}={value}" for key, value in sorted(command.env.items()))
            log_text = f" log_file={command.log_file}" if command.log_file else ""
            print(
                f"dry_run_command env='{env_text}' command={printable_command(command.argv)}{log_text}",
                flush=True,
            )

    def _command_base(self) -> list[str]:
        return [
            *command_prefix(str(self.config.execution.get("conda_run", ""))),
            str(self.config.execution.get("python_bin", "python")),
        ]

    def _teacher_generate_plan(self, mode: str) -> StagePlan:
        mode_cfg = self.config.mode_config(mode)
        generation = mode_cfg["generation"]
        paths = paths_for_mode(self.config, mode)
        num_shards = int(generation["num_shards"])
        gpu_ids = parse_gpu_ids(generation["gpu_ids"])
        if num_shards < 1:
            raise ValueError("generation.num_shards must be >= 1.")
        if num_shards > len(gpu_ids):
            raise ValueError("generation.num_shards cannot exceed generation.gpu_ids count.")

        commands: list[CommandSpec] = []
        for shard_index in range(num_shards):
            shard_file = paths.shard_dir / f"candidate_pool.shard_{shard_index:02d}_of_{num_shards:02d}.jsonl"
            argv = [
                *self._command_base(),
                "src/vllm_dual_decoding/teacher_generate.py",
                "--generation-mode",
                MODE_TO_GENERATION_MODE[mode],
                "--model-name-or-path",
                str(mode_cfg["models"]["teacher"]),
                "--student-model-name-or-path",
                str(mode_cfg["models"]["student"]),
                "--dataset-name",
                self.config.dataset.dataset_name,
                "--dataset-config-name",
                self.config.dataset.dataset_config_name,
                "--split",
                self.config.dataset.train_split,
                "--question-field",
                self.config.dataset.question_field,
                "--answer-field",
                self.config.dataset.answer_field,
                "--dataset-key",
                self.config.dataset.key,
                "--run-id",
                paths.run_id,
                "--answer-extraction-mode",
                self.config.dataset.answer_extraction,
                "--output-file",
                str(shard_file),
                "--max-samples",
                str(generation["max_samples"]),
                "--subset-seed",
                str(generation["subset_seed"]),
                "--num-shards",
                str(num_shards),
                "--shard-index",
                str(shard_index),
                "--num-candidates",
                str(generation["num_candidates"]),
                "--temperature",
                str(generation["temperature"]),
                "--top-p",
                str(generation["top_p"]),
                "--max-new-tokens",
                str(generation["max_new_tokens"]),
                "--seed",
                str(generation["seed"]),
                "--prompt-batch-size",
                str(generation["prompt_batch_size"]),
                "--save-every-prompts",
                str(generation["save_every_prompts"]),
                "--tensor-parallel-size",
                str(generation["tensor_parallel_size"]),
                "--small-tensor-parallel-size",
                str(generation["small_tensor_parallel_size"]),
                "--dtype",
                str(generation["dtype"]),
                "--max-model-len",
                str(generation["max_model_len"]),
                "--max-num-seqs",
                str(generation["max_num_seqs"]),
                "--gpu-memory-utilization",
                str(generation["gpu_memory_utilization"]),
                "--small-gpu-memory-utilization",
                str(generation["small_gpu_memory_utilization"]),
            ]
            if mode in {"teacher_token_hard", "teacher_token_soft"}:
                _append_option(argv, "--hard-candidate-top-k", generation.get("hard_candidate_top_k"))
                _append_option(argv, "--hard-candidate-top-p", generation.get("hard_candidate_top_p"))
                _append_option(argv, "--debug-log-interval", generation.get("debug_log_interval"))
            if mode == "teacher_token_soft":
                _append_option(argv, "--soft-student-weight", generation.get("soft_student_weight"))
                _append_option(argv, "--soft-temperature", generation.get("soft_temperature"))
            _append_bool_flag(argv, "--enforce-eager", bool(generation.get("enforce_eager")))
            _append_bool_flag(argv, "--trust-remote-code", bool(generation.get("trust_remote_code")))
            _append_bool_flag(argv, "--allow-remote-model-files", bool(generation.get("allow_remote_model_files")))
            _append_bool_flag(argv, "--use-tqdm", bool(generation.get("use_tqdm")))
            commands.append(
                CommandSpec(
                    argv=argv,
                    env=_base_env({"CUDA_VISIBLE_DEVICES": gpu_ids[shard_index]}),
                    log_file=paths.log_dir / f"teacher_generate_shard_{shard_index:02d}.log",
                )
            )
        return StagePlan("teacher_generate", mode, paths.run_id, paths, commands)

    def _student_score_plan(self, mode: str) -> StagePlan:
        mode_cfg = self.config.mode_config(mode)
        scoring = mode_cfg["scoring"]
        paths = paths_for_mode(self.config, mode)
        gpu_id = parse_gpu_ids(scoring["gpu_ids"])[0]
        argv = [
            *self._command_base(),
            "src/vllm_dual_decoding/student_score.py",
            "--model-name-or-path",
            str(mode_cfg["models"]["student"]),
            "--input-file",
            str(paths.candidate_file),
            "--output-file",
            str(paths.scored_file),
            "--batch-size",
            str(scoring["batch_size"]),
            "--max-length",
            str(scoring["max_length"]),
            "--device",
            str(scoring["device"]),
            "--attn-implementation",
            str(scoring["attn_implementation"]),
        ]
        _append_bool_flag(argv, "--allow-remote-model-files", bool(scoring.get("allow_remote_model_files")))
        _append_bool_flag(argv, "--allow-hash-answer-fallback", self.config.allows_hash_answer_fallback())
        return StagePlan(
            "student_score",
            mode,
            paths.run_id,
            paths,
            [
                CommandSpec(
                    argv=argv,
                    env={"CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"},
                    log_file=paths.log_dir / "student_score.log",
                )
            ],
        )

    def _build_distill_plan(self, mode: str) -> StagePlan:
        mode_cfg = self.config.mode_config(mode)
        generation = mode_cfg["generation"]
        eval_cfg = mode_cfg["eval"]
        paths = paths_for_mode(self.config, mode)
        exclude_max_samples = generation["max_samples"] if self.config.dataset.eval_exclude_train_subset else 0
        exclude_seed = generation["subset_seed"]
        argv = [
            *self._command_base(),
            "src/pre_exp/build_holdout_eval_dataset.py",
            "--dataset-name",
            self.config.dataset.dataset_name,
            "--dataset-config-name",
            self.config.dataset.dataset_config_name,
            "--split",
            self.config.dataset.eval_split,
            "--question-field",
            self.config.dataset.question_field,
            "--answer-field",
            self.config.dataset.answer_field,
            "--completion-field",
            self.config.dataset.completion_field,
            "--output-file",
            str(paths.eval_file),
            "--summary-file",
            str(paths.eval_summary_file),
            "--max-samples",
            str(eval_cfg["holdout_max_samples"]),
            "--subset-seed",
            str(generation["subset_seed"]),
            "--exclude-subset-max-samples",
            str(exclude_max_samples),
            "--exclude-subset-seed",
            str(exclude_seed),
        ]
        return StagePlan(
            "build_distill",
            mode,
            paths.run_id,
            paths,
            [CommandSpec(argv=argv, env={"PYTHONUNBUFFERED": "1"}, log_file=paths.log_dir / "build_eval_dataset.log")],
        )

    def _train_plan(self, mode: str) -> StagePlan:
        mode_cfg = self.config.mode_config(mode)
        training = mode_cfg["training"]
        paths = paths_for_mode(self.config, mode)
        argv = [
            *command_prefix(str(self.config.execution.get("conda_run", ""))),
            "torchrun",
            "--standalone",
            f"--nproc_per_node={training['nproc_per_node']}",
            f"--master_port={training['master_port']}",
            "src/train_sft.py",
            "--model-name-or-path",
            str(mode_cfg["models"]["student"]),
            "--dataset-format",
            "distill_jsonl",
            "--train-file",
            str(paths.distill_file),
            "--eval-file",
            str(paths.eval_file),
            "--output-dir",
            str(paths.run_dir),
            "--max-steps",
            str(training["max_steps"]),
            "--eval-every",
            str(training["eval_every"]),
            "--checkpoint-every",
            str(training["checkpoint_every"]),
            "--max-checkpoints-to-keep",
            str(training["max_checkpoints_to_keep"]),
            "--max-length",
            str(training["max_length"]),
            "--train-batch-size",
            str(training["train_batch_size"]),
            "--eval-batch-size",
            str(training["eval_batch_size"]),
            "--learning-rate",
            str(training["learning_rate"]),
            "--weight-decay",
            str(training["weight_decay"]),
            "--warmup-ratio",
            str(training["warmup_ratio"]),
            "--seed",
            str(training["seed"]),
            "--attn-implementation",
            str(training["attn_implementation"]),
            "--disable-rollout-eval",
            "--disable-eval-preview",
        ]
        _append_bool_flag(argv, "--allow-remote-model-files", bool(training.get("allow_remote_model_files")))
        return StagePlan(
            "train",
            mode,
            paths.run_id,
            paths,
            [
                CommandSpec(
                    argv=argv,
                    env={"CUDA_VISIBLE_DEVICES": str(training["gpu_ids"]), "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"},
                    log_file=paths.run_dir / "train.log",
                )
            ],
        )

    def _eval_plan(self, mode: str) -> StagePlan:
        mode_cfg = self.config.mode_config(mode)
        eval_cfg = mode_cfg["eval"]
        generation = mode_cfg["generation"]
        paths = paths_for_mode(self.config, mode)
        exclude_max_samples = generation["max_samples"] if self.config.dataset.eval_exclude_train_subset else 0
        gpu_id = parse_gpu_ids(eval_cfg["gpu_ids"])[0]
        argv = [
            *self._command_base(),
            "src/pre_exp/final_eval.py",
            "--engine",
            str(eval_cfg["engine"]),
            "--model-name-or-path",
            str(paths.run_dir / "final_checkpoint"),
            "--dataset-name",
            self.config.dataset.dataset_name,
            "--dataset-config-name",
            self.config.dataset.dataset_config_name,
            "--split",
            self.config.dataset.eval_split,
            "--question-field",
            self.config.dataset.question_field,
            "--answer-field",
            self.config.dataset.answer_field,
            "--exclude-subset-max-samples",
            str(exclude_max_samples),
            "--exclude-subset-seed",
            str(generation["subset_seed"]),
            "--subset-seed",
            str(generation["subset_seed"]),
            "--max-samples",
            str(eval_cfg["max_samples"]),
            "--max-new-tokens",
            str(eval_cfg["max_new_tokens"]),
            "--num-rollouts",
            str(eval_cfg["num_rollouts"]),
            "--temperature",
            str(eval_cfg["temperature"]),
            "--top-p",
            str(eval_cfg["top_p"]),
            "--sampling-seed",
            str(eval_cfg["sampling_seed"]),
            "--tensor-parallel-size",
            str(eval_cfg["tensor_parallel_size"]),
            "--max-model-len",
            str(eval_cfg["max_model_len"]),
            "--max-num-seqs",
            str(eval_cfg["max_num_seqs"]),
            "--gpu-memory-utilization",
            str(eval_cfg["gpu_memory_utilization"]),
            "--output-file",
            str(paths.final_eval_file),
        ]
        _append_bool_flag(argv, "--trust-remote-code", bool(eval_cfg.get("trust_remote_code")))
        _append_bool_flag(argv, "--allow-remote-model-files", bool(eval_cfg.get("allow_remote_model_files")))
        _append_bool_flag(argv, "--allow-hash-answer-fallback", self.config.allows_hash_answer_fallback())
        return StagePlan(
            "eval",
            mode,
            paths.run_id,
            paths,
            [CommandSpec(argv=argv, env=_base_env({"CUDA_VISIBLE_DEVICES": gpu_id}), log_file=paths.log_dir / "final_eval.log")],
        )

    def _plot_plan(self, mode: str) -> StagePlan:
        paths = paths_for_mode(self.config, mode)
        argv = [
            *self._command_base(),
            "src/pre_exp/plot_curves.py",
            "--run-dir",
            str(paths.run_dir.parent),
            "--analysis-dir",
            str(paths.analysis_dir),
            "--output-dir",
            str(paths.curve_dir),
            "--modes",
            mode,
            "--mode-dir",
            f"{mode}={paths.run_dir}",
            "--analysis-file",
            f"{mode}={paths.final_eval_file}",
        ]
        return StagePlan(
            "plot",
            mode,
            paths.run_id,
            paths,
            [
                CommandSpec(
                    argv=argv,
                    env={"PYTHONUNBUFFERED": "1"},
                    log_file=paths.log_dir / "plot_curves.log",
                )
            ],
        )

    def _run_teacher_generate(self, plan: StagePlan, *, allow_overwrite: bool) -> None:
        plan.paths.shard_dir.mkdir(parents=True, exist_ok=True)
        plan.paths.log_dir.mkdir(parents=True, exist_ok=True)
        guard_output_file(plan.paths.candidate_file, allow_overwrite=allow_overwrite)
        for command in plan.commands:
            if command.log_file is None:
                raise RuntimeError("teacher_generate commands must define log_file.")
            guard_output_file(command.log_file, allow_overwrite=allow_overwrite)
            output_idx = command.argv.index("--output-file") + 1
            guard_output_file(Path(command.argv[output_idx]), allow_overwrite=allow_overwrite)

        processes: list[tuple[CommandSpec, subprocess.Popen[Any], Any]] = []
        for command in plan.commands:
            assert command.log_file is not None
            env = os.environ.copy()
            env.update(command.env)
            command.log_file.parent.mkdir(parents=True, exist_ok=True)
            log_handle = command.log_file.open("w", encoding="utf-8")
            log_handle.write(f"command={printable_command(command.argv)}\n\n")
            log_handle.flush()
            process = subprocess.Popen(
                command.argv,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((command, process, log_handle))

        failed: list[str] = []
        for command, process, log_handle in processes:
            try:
                return_code = process.wait()
                if return_code != 0:
                    failed.append(f"{command.log_file}:exit={return_code}")
            finally:
                log_handle.close()
        if failed:
            raise SystemExit("At least one teacher_generate shard failed: " + ", ".join(failed))

        with plan.paths.candidate_file.open("w", encoding="utf-8") as output:
            for command in plan.commands:
                output_idx = command.argv.index("--output-file") + 1
                shard_file = Path(command.argv[output_idx])
                if not shard_file.is_file() or shard_file.stat().st_size == 0:
                    raise FileNotFoundError(f"Missing or empty shard file: {shard_file}")
                with shard_file.open("r", encoding="utf-8") as shard:
                    for line in shard:
                        output.write(line)
        print(f"teacher_generate_stage_done run_id={plan.run_id} output_file={plan.paths.candidate_file}", flush=True)

    def _run_single_command_stage(
        self,
        plan: StagePlan,
        *,
        allow_overwrite: bool,
        outputs: list[Path],
    ) -> None:
        command = plan.commands[0]
        for output in outputs:
            guard_output_file(output, allow_overwrite=allow_overwrite)
        if command.log_file is not None:
            guard_output_file(command.log_file, allow_overwrite=allow_overwrite)
            command.log_file.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(command.env)
        if command.log_file is None:
            subprocess.run(command.argv, check=True, env=env)
            return
        with command.log_file.open("w", encoding="utf-8") as f:
            f.write(f"command={printable_command(command.argv)}\n\n")
            f.flush()
            subprocess.run(command.argv, check=True, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f"{plan.stage}_stage_done run_id={plan.run_id}", flush=True)

    def _run_build_distill(self, plan: StagePlan, *, allow_overwrite: bool) -> None:
        guard_output_file(plan.paths.distill_file, allow_overwrite=allow_overwrite)
        guard_output_file(plan.paths.distill_summary_file, allow_overwrite=allow_overwrite)
        records = read_jsonl(plan.paths.scored_file)
        distill_records = build_distill_records(records, mode=plan.mode)
        plan.paths.dataset_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(plan.paths.distill_file, distill_records)
        summary = summarize_distill_build(
            mode=plan.mode,
            input_file=plan.paths.scored_file,
            output_file=plan.paths.distill_file,
            input_records=records,
            distill_records=distill_records,
        )
        write_json(plan.paths.distill_summary_file, summary)
        self._run_single_command_stage(
            plan,
            allow_overwrite=allow_overwrite,
            outputs=[plan.paths.eval_file, plan.paths.eval_summary_file],
        )
        print(
            "build_distill_stage_done "
            f"run_id={plan.run_id} distill_file={plan.paths.distill_file} eval_file={plan.paths.eval_file}",
            flush=True,
        )
