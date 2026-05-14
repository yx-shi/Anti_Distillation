from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vllm_dual_decoding.build_distill_dataset import (
    build_distill_records,
    infer_mode,
    summarize_distill_build,
)
from vllm_dual_decoding.common import read_jsonl, write_json, write_jsonl


DEFAULT_EXPERIMENT_NAME = "vllm_dual_deepscaler8000_k1_t0.7_p0.8_len4096"
DEFAULT_RESULT_ROOT = "result/vllm_dual_decoding"
DEFAULT_RUN_ROOT = (
    "/home/disk2/shiyixuan/Anti_Distillation/result/vllm_dual_decoding/runs"
)
DEFAULT_STUDENT_MODEL = "/home/disk1/public_checkpoint/Qwen3-1.7B"
DEFAULT_CONDA_RUN = (
    f"{Path.home()}/miniconda3/bin/conda run --no-capture-output -n adistill-unified"
)

MODE_ORDER = ("teacher_plain", "teacher_token_hard", "teacher_token_soft")
MODE_ALIASES = {
    "plain": "teacher_plain",
    "teacher_plain": "teacher_plain",
    "hard": "teacher_token_hard",
    "teacher_token_hard": "teacher_token_hard",
    "soft": "teacher_token_soft",
    "teacher_token_soft": "teacher_token_soft",
}


@dataclass(frozen=True)
class ModeInput:
    mode: str
    scored_candidates_file: Path


@dataclass(frozen=True)
class PipelinePaths:
    result_root: Path
    experiment_name: str
    dataset_dir: Path
    analysis_dir: Path
    log_dir: Path
    run_dir: Path
    holdout_eval_file: Path
    holdout_eval_summary_file: Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build SFT-ready token-level vLLM-dual datasets and optionally run "
            "training plus offline rollout eval for plain/hard/soft groups."
        )
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--mode-file",
        action="append",
        default=[],
        metavar="MODE=PATH",
        help=(
            "Scored candidates for a mode. Repeat for teacher_plain, teacher_token_hard "
            "and teacher_token_soft. If omitted, files are auto-discovered under "
            "RESULT_ROOT/candidates/EXPERIMENT_NAME."
        ),
    )
    parser.add_argument(
        "--stages",
        default="build_distill,build_holdout",
        help=(
            "Comma-separated stages: build_distill,build_holdout,train,checkpoint_eval,"
            "final_eval. Default is safe data preparation only."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--conda-run", default=DEFAULT_CONDA_RUN)
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--student-model", default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--train-gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--train-nproc", type=int, default=8)
    parser.add_argument("--train-master-port-base", type=int, default=29631)
    parser.add_argument("--eval-gpu-ids", default="0")

    parser.add_argument("--max-steps", type=int, default=1600)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--max-checkpoints-to-keep", type=int, default=8)
    parser.add_argument("--train-max-length", type=int, default=5120)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="flash_attention_2")

    parser.add_argument("--dataset-name", default="agentica-org/DeepScaleR-Preview-Dataset")
    parser.add_argument("--dataset-config-name", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--question-field", default="problem")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--completion-field", default="solution")
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument(
        "--train-subset-max-samples",
        type=int,
        default=8000,
        help="Training subset size to exclude from DeepScaleR holdout eval.",
    )
    parser.add_argument("--holdout-eval-max-samples", type=int, default=1024)

    parser.add_argument("--eval-engine", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--checkpoint-eval-max-samples", type=int, default=1024)
    parser.add_argument("--final-eval-max-samples", type=int, default=2048)
    parser.add_argument("--rollout-max-new-tokens", type=int, default=4096)
    parser.add_argument("--eval-num-rollouts", type=int, default=4)
    parser.add_argument("--eval-temperature", type=float, default=0.7)
    parser.add_argument("--eval-top-p", type=float, default=0.8)
    parser.add_argument("--eval-sampling-seed", type=int, default=42)
    parser.add_argument("--eval-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--eval-max-model-len", type=int, default=8192)
    parser.add_argument("--eval-max-num-seqs", type=int, default=32)
    parser.add_argument("--eval-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def normalize_mode(raw_mode: str) -> str:
    mode = raw_mode.strip()
    if mode not in MODE_ALIASES:
        raise ValueError(
            f"Unsupported mode `{raw_mode}`. Expected one of: {', '.join(sorted(MODE_ALIASES))}."
        )
    return MODE_ALIASES[mode]


def parse_mode_files(raw_mode_files: list[str]) -> list[ModeInput]:
    mode_inputs: list[ModeInput] = []
    for raw_item in raw_mode_files:
        if "=" not in raw_item:
            raise ValueError(f"--mode-file must use MODE=PATH format, got: {raw_item}")
        raw_mode, raw_path = raw_item.split("=", maxsplit=1)
        mode_inputs.append(ModeInput(normalize_mode(raw_mode), Path(raw_path)))
    return sort_mode_inputs(mode_inputs)


def sort_mode_inputs(mode_inputs: list[ModeInput]) -> list[ModeInput]:
    order = {mode: idx for idx, mode in enumerate(MODE_ORDER)}
    return sorted(mode_inputs, key=lambda item: order.get(item.mode, len(order)))


def infer_mode_from_dirname(dirname: str) -> str | None:
    lowered = dirname.lower()
    if lowered in MODE_ALIASES:
        return MODE_ALIASES[lowered]
    if "plain" in lowered:
        return "teacher_plain"
    if "hard" in lowered:
        return "teacher_token_hard"
    if "soft" in lowered:
        return "teacher_token_soft"
    return None


def discover_mode_inputs(result_root: Path, experiment_name: str) -> list[ModeInput]:
    candidate_root = result_root / "candidates" / experiment_name
    if not candidate_root.is_dir():
        raise FileNotFoundError(f"Candidate directory does not exist: {candidate_root}")

    discovered: dict[str, Path] = {}
    for mode_dir in sorted(path for path in candidate_root.iterdir() if path.is_dir()):
        mode = infer_mode_from_dirname(mode_dir.name)
        scored_file = mode_dir / "scored_candidates.jsonl"
        if mode is None or not scored_file.is_file():
            continue
        discovered.setdefault(mode, scored_file)

    missing_modes = [mode for mode in MODE_ORDER if mode not in discovered]
    if missing_modes:
        raise FileNotFoundError(
            f"Could not discover scored_candidates.jsonl for modes {missing_modes} under {candidate_root}. "
            "Pass explicit --mode-file MODE=PATH values if your layout is custom."
        )
    return [ModeInput(mode, discovered[mode]) for mode in MODE_ORDER]


def parse_stages(raw_stages: str) -> set[str]:
    stages = {stage.strip() for stage in raw_stages.split(",") if stage.strip()}
    allowed = {"build_distill", "build_holdout", "train", "checkpoint_eval", "final_eval"}
    unknown = sorted(stages - allowed)
    if unknown:
        raise ValueError(f"Unknown stage(s): {', '.join(unknown)}")
    return stages


def build_paths(args: argparse.Namespace) -> PipelinePaths:
    result_root = Path(args.result_root)
    dataset_dir = result_root / "datasets" / args.experiment_name
    analysis_dir = result_root / "analysis" / args.experiment_name
    log_dir = analysis_dir / "logs"
    run_dir = Path(args.run_root) / args.experiment_name
    holdout_eval_file = (
        dataset_dir
        / f"deepscaler_holdout_eval_{args.holdout_eval_max_samples}_seed{args.subset_seed}.jsonl"
    )
    return PipelinePaths(
        result_root=result_root,
        experiment_name=args.experiment_name,
        dataset_dir=dataset_dir,
        analysis_dir=analysis_dir,
        log_dir=log_dir,
        run_dir=run_dir,
        holdout_eval_file=holdout_eval_file,
        holdout_eval_summary_file=holdout_eval_file.with_suffix(".summary.json"),
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def guard_output(path: Path, *, allow_overwrite: bool) -> None:
    if allow_overwrite:
        return
    if path.exists() and path.stat().st_size > 0:
        raise FileExistsError(
            f"Refusing to overwrite non-empty file: {path}. Pass --allow-overwrite to rerun."
        )


def command_prefix(raw_conda_run: str) -> list[str]:
    return shlex.split(raw_conda_run) if raw_conda_run.strip() else []


def printable_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print(f"run_command {printable_command(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True, env=env)


def mode_to_safe_port_offset(mode: str) -> int:
    return {
        "teacher_plain": 0,
        "teacher_token_hard": 2,
        "teacher_token_soft": 4,
    }[mode]


def distill_file_for(paths: PipelinePaths, mode: str) -> Path:
    return paths.dataset_dir / f"distill_{mode}.jsonl"


def build_distill_stage(
    mode_inputs: list[ModeInput],
    paths: PipelinePaths,
    *,
    allow_overwrite: bool,
    dry_run: bool,
) -> None:
    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    summary_output_file = paths.dataset_dir / "distill_summary.json"
    if not dry_run:
        guard_output(summary_output_file, allow_overwrite=allow_overwrite)
    for mode_input in mode_inputs:
        records = read_jsonl(mode_input.scored_candidates_file)
        file_mode = infer_mode(records, "")
        if file_mode != mode_input.mode:
            raise ValueError(
                f"Mode mismatch for {mode_input.scored_candidates_file}: "
                f"CLI/discovery mode={mode_input.mode}, file mode={file_mode}."
            )
        mode = mode_input.mode
        output_file = distill_file_for(paths, mode)
        summary_file = output_file.with_suffix(".summary.json")
        if not dry_run:
            guard_output(output_file, allow_overwrite=allow_overwrite)
            guard_output(summary_file, allow_overwrite=allow_overwrite)
        distill_records = build_distill_records(records, mode=mode)
        summary = summarize_distill_build(
            mode=mode,
            input_file=mode_input.scored_candidates_file,
            output_file=output_file,
            input_records=records,
            distill_records=distill_records,
        )
        if not dry_run:
            write_jsonl(output_file, distill_records)
            write_json(summary_file, summary)
        summaries[mode] = summary
        print(
            "vllm_dual_pipeline_build_distill_done "
            f"mode={mode} records={len(distill_records)} output_file={output_file} "
            f"dry_run={dry_run}",
            flush=True,
        )
    if not dry_run:
        write_json(summary_output_file, {"modes": summaries})


def build_holdout_command(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    return [
        *command_prefix(args.conda_run),
        args.python_bin,
        "src/pre_exp/build_holdout_eval_dataset.py",
        "--dataset-name",
        args.dataset_name,
        "--dataset-config-name",
        args.dataset_config_name,
        "--split",
        args.split,
        "--question-field",
        args.question_field,
        "--answer-field",
        args.answer_field,
        "--completion-field",
        args.completion_field,
        "--output-file",
        str(paths.holdout_eval_file),
        "--summary-file",
        str(paths.holdout_eval_summary_file),
        "--max-samples",
        str(args.holdout_eval_max_samples),
        "--subset-seed",
        str(args.subset_seed),
        "--exclude-subset-max-samples",
        str(args.train_subset_max_samples),
        "--exclude-subset-seed",
        str(args.subset_seed),
    ]


def build_train_command(args: argparse.Namespace, paths: PipelinePaths, mode: str) -> list[str]:
    output_dir = paths.run_dir / mode
    port = args.train_master_port_base + mode_to_safe_port_offset(mode)
    return [
        *command_prefix(args.conda_run),
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.train_nproc}",
        f"--master_port={port}",
        "src/train_sft.py",
        "--model-name-or-path",
        args.student_model,
        "--dataset-format",
        "distill_jsonl",
        "--train-file",
        str(distill_file_for(paths, mode)),
        "--eval-file",
        str(paths.holdout_eval_file),
        "--output-dir",
        str(output_dir),
        "--max-steps",
        str(args.max_steps),
        "--eval-every",
        str(args.eval_every),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--max-checkpoints-to-keep",
        str(args.max_checkpoints_to_keep),
        "--max-length",
        str(args.train_max_length),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--seed",
        str(args.train_seed),
        "--attn-implementation",
        args.attn_implementation,
        "--rollout-max-new-tokens",
        str(args.rollout_max_new_tokens),
        "--disable-rollout-eval",
        "--disable-eval-preview",
    ]


def eval_common_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--engine",
        args.eval_engine,
        "--dataset-name",
        args.dataset_name,
        "--dataset-config-name",
        args.dataset_config_name,
        "--split",
        args.split,
        "--question-field",
        args.question_field,
        "--answer-field",
        args.answer_field,
        "--exclude-subset-max-samples",
        str(args.train_subset_max_samples),
        "--exclude-subset-seed",
        str(args.subset_seed),
        "--subset-seed",
        str(args.subset_seed),
        "--max-new-tokens",
        str(args.rollout_max_new_tokens),
        "--num-rollouts",
        str(args.eval_num_rollouts),
        "--temperature",
        str(args.eval_temperature),
        "--top-p",
        str(args.eval_top_p),
        "--sampling-seed",
        str(args.eval_sampling_seed),
        "--tensor-parallel-size",
        str(args.eval_tensor_parallel_size),
        "--max-model-len",
        str(args.eval_max_model_len),
        "--max-num-seqs",
        str(args.eval_max_num_seqs),
        "--gpu-memory-utilization",
        str(args.eval_gpu_memory_utilization),
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def build_checkpoint_eval_command(args: argparse.Namespace, paths: PipelinePaths, mode: str) -> list[str]:
    return [
        *command_prefix(args.conda_run),
        args.python_bin,
        "src/pre_exp/final_eval.py",
        *eval_common_args(args),
        "--checkpoint-root",
        str(paths.run_dir / mode),
        "--max-samples",
        str(args.checkpoint_eval_max_samples),
        "--output-file",
        str(paths.analysis_dir / f"checkpoint_eval_{mode}.json"),
        "--output-prefix",
        f"checkpoint_eval_{mode}",
    ]


def build_final_eval_command(args: argparse.Namespace, paths: PipelinePaths, mode: str) -> list[str]:
    return [
        *command_prefix(args.conda_run),
        args.python_bin,
        "src/pre_exp/final_eval.py",
        *eval_common_args(args),
        "--model-name-or-path",
        str(paths.run_dir / mode / "final_checkpoint"),
        "--max-samples",
        str(args.final_eval_max_samples),
        "--output-file",
        str(paths.analysis_dir / f"final_eval_{mode}.json"),
    ]


def run_external_stage(
    command: list[str],
    *,
    log_file: Path,
    env_updates: dict[str, str] | None,
    dry_run: bool,
) -> None:
    ensure_parent(log_file)
    env = os.environ.copy()
    env.setdefault("VLLM_USE_V1", "0")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    if env_updates:
        env.update(env_updates)

    if dry_run:
        run_command(command, env=env, dry_run=True)
        print(f"dry_run_log_file {log_file}", flush=True)
        return

    with log_file.open("w", encoding="utf-8") as f:
        print(f"run_command {printable_command(command)}", flush=True)
        subprocess.run(command, check=True, env=env, stdout=f, stderr=subprocess.STDOUT)


def eval_gpu_for_mode(eval_gpu_ids: str, mode: str) -> str:
    gpu_ids = [item.strip() for item in eval_gpu_ids.replace(",", " ").split() if item.strip()]
    if not gpu_ids:
        return "0"
    idx = MODE_ORDER.index(mode) if mode in MODE_ORDER else 0
    return gpu_ids[idx % len(gpu_ids)]


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = build_paths(args)
    stages = parse_stages(args.stages)
    mode_inputs = (
        parse_mode_files(args.mode_file)
        if args.mode_file
        else discover_mode_inputs(Path(args.result_root), args.experiment_name)
    )

    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    paths.analysis_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    print(
        "vllm_dual_full_pipeline_start "
        f"experiment={args.experiment_name} "
        f"stages={','.join(sorted(stages))} "
        f"modes={','.join(mode_input.mode for mode_input in mode_inputs)} "
        f"dataset_dir={paths.dataset_dir} "
        f"run_dir={paths.run_dir} "
        f"analysis_dir={paths.analysis_dir}",
        flush=True,
    )

    if "build_distill" in stages:
        build_distill_stage(
            mode_inputs,
            paths,
            allow_overwrite=args.allow_overwrite,
            dry_run=args.dry_run,
        )

    if "build_holdout" in stages:
        guard_output(paths.holdout_eval_file, allow_overwrite=args.allow_overwrite)
        guard_output(paths.holdout_eval_summary_file, allow_overwrite=args.allow_overwrite)
        run_external_stage(
            build_holdout_command(args, paths),
            log_file=paths.log_dir / "build_holdout_eval.log",
            env_updates=None,
            dry_run=args.dry_run,
        )

    if "train" in stages:
        for mode_input in mode_inputs:
            run_external_stage(
                build_train_command(args, paths, mode_input.mode),
                log_file=paths.run_dir / mode_input.mode / "train.log",
                env_updates={"CUDA_VISIBLE_DEVICES": args.train_gpu_ids},
                dry_run=args.dry_run,
            )

    if "checkpoint_eval" in stages:
        for mode_input in mode_inputs:
            run_external_stage(
                build_checkpoint_eval_command(args, paths, mode_input.mode),
                log_file=paths.log_dir / f"checkpoint_eval_{mode_input.mode}.log",
                env_updates={"CUDA_VISIBLE_DEVICES": eval_gpu_for_mode(args.eval_gpu_ids, mode_input.mode)},
                dry_run=args.dry_run,
            )

    if "final_eval" in stages:
        for mode_input in mode_inputs:
            run_external_stage(
                build_final_eval_command(args, paths, mode_input.mode),
                log_file=paths.log_dir / f"final_eval_{mode_input.mode}.log",
                env_updates={"CUDA_VISIBLE_DEVICES": eval_gpu_for_mode(args.eval_gpu_ids, mode_input.mode)},
                dry_run=args.dry_run,
            )

    print(
        "vllm_dual_full_pipeline_done "
        f"dataset_dir={paths.dataset_dir} "
        f"run_dir={paths.run_dir} "
        f"analysis_dir={paths.analysis_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
