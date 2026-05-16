from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pre_exp.common import write_json
from pre_exp.collect_eval_summary import read_json


DEFAULT_EXPERIMENT_NAME = "vllm_dual_deepscaler8000_k1_t0.7_p0.8_len4096"
DEFAULT_RESULT_ROOT = "result/vllm_dual_decoding"
DEFAULT_RUN_ROOT = (
    "/home/disk2/shiyixuan/Anti_Distillation/result/vllm_dual_decoding/runs"
)
DEFAULT_BASE_MODEL = "/home/disk1/public_checkpoint/Qwen3-1.7B"
DEFAULT_CONDA_RUN = (
    f"{Path.home()}/miniconda3/bin/conda run --no-capture-output -n adistill-unified"
)
MODES = ("teacher_plain", "teacher_token_hard", "teacher_token_soft")


@dataclass(frozen=True)
class EvalJob:
    mode: str
    label: str
    step: int
    model_path: Path
    output_file: Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run rollout eval for token-level vLLM-dual checkpoints with one TP=1 vLLM "
            "process per GPU. Jobs from all modes share one global GPU worker queue."
        )
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--conda-run", default=DEFAULT_CONDA_RUN)
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Do not run plot_curves.py after all rollout eval jobs finish.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=10.0,
        help="Seconds between process status checks while the GPU worker queue is running.",
    )

    parser.add_argument("--dataset-name", default="agentica-org/DeepScaleR-Preview-Dataset")
    parser.add_argument("--dataset-config-name", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--question-field", default="problem")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--exclude-subset-max-samples", type=int, default=8000)
    parser.add_argument("--exclude-subset-seed", type=int, default=42)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-tqdm", action="store_true")
    return parser


def parse_gpu_ids(raw_gpu_ids: str) -> list[str]:
    gpu_ids = [item.strip() for item in raw_gpu_ids.replace(",", " ").split() if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU id.")
    return gpu_ids


def command_prefix(raw_conda_run: str) -> list[str]:
    return shlex.split(raw_conda_run) if raw_conda_run.strip() else []


def checkpoint_step_from_dir(path: Path) -> int:
    name = path.name
    if not name.startswith("checkpoint-step-"):
        raise ValueError(f"Not a checkpoint-step directory: {path}")
    return int(name.removeprefix("checkpoint-step-"))


def final_step_from_train_config(mode_dir: Path) -> int:
    config_path = mode_dir / "train_config.json"
    if config_path.is_file():
        payload = read_json(config_path)
        if isinstance(payload, dict) and payload.get("max_steps") is not None:
            return int(payload["max_steps"])

    log_path = mode_dir / "train.log"
    if log_path.is_file():
        for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
            if "training_done" in line and "steps=" in line:
                for token in line.split():
                    if token.startswith("steps="):
                        return int(token.removeprefix("steps="))
    raise FileNotFoundError(f"Unable to infer final step from {mode_dir}.")


def discover_eval_jobs_for_mode(
    *,
    mode: str,
    mode_dir: Path,
    base_model: str,
    analysis_dir: Path,
) -> list[EvalJob]:
    if not mode_dir.is_dir():
        raise FileNotFoundError(f"Mode run directory does not exist: {mode_dir}")
    eval_dir = analysis_dir / "eval"

    jobs = [
        EvalJob(
            mode=mode,
            label="000000",
            step=0,
            model_path=Path(base_model),
            output_file=eval_dir / f"checkpoint_eval_{mode}_000000.json",
        )
    ]

    for checkpoint_dir in sorted(mode_dir.glob("checkpoint-step-*")):
        if not checkpoint_dir.is_dir():
            continue
        step = checkpoint_step_from_dir(checkpoint_dir)
        label = f"{step:06d}"
        jobs.append(
            EvalJob(
                mode=mode,
                label=label,
                step=step,
                model_path=checkpoint_dir,
                output_file=eval_dir / f"checkpoint_eval_{mode}_{label}.json",
            )
        )

    final_checkpoint = mode_dir / "final_checkpoint"
    if final_checkpoint.is_dir():
        jobs.append(
            EvalJob(
                mode=mode,
                label="final_checkpoint",
                step=final_step_from_train_config(mode_dir),
                model_path=final_checkpoint,
                output_file=eval_dir / f"checkpoint_eval_{mode}_final_checkpoint.json",
            )
        )
    return jobs


def collect_eval_jobs(
    *,
    modes: list[str],
    run_dir: Path,
    base_model: str,
    analysis_dir: Path,
) -> dict[str, list[EvalJob]]:
    return {
        mode: discover_eval_jobs_for_mode(
            mode=mode,
            mode_dir=run_dir / mode,
            base_model=base_model,
            analysis_dir=analysis_dir,
        )
        for mode in modes
    }


def gpu_batches(jobs: list[EvalJob], gpu_ids: list[str]) -> list[list[tuple[str, EvalJob]]]:
    batches: list[list[tuple[str, EvalJob]]] = []
    for start in range(0, len(jobs), len(gpu_ids)):
        batch_jobs = jobs[start : start + len(gpu_ids)]
        batches.append([(gpu_ids[idx], job) for idx, job in enumerate(batch_jobs)])
    return batches


def build_eval_command(
    *,
    conda_run: list[str],
    python_bin: str,
    job: EvalJob,
    dataset_name: str,
    dataset_config_name: str,
    split: str,
    question_field: str,
    answer_field: str,
    max_samples: int,
    exclude_subset_max_samples: int,
    exclude_subset_seed: int,
    subset_seed: int,
    max_new_tokens: int,
    num_rollouts: int,
    temperature: float,
    top_p: float,
    sampling_seed: int,
    tensor_parallel_size: int,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    trust_remote_code: bool,
) -> list[str]:
    command = [
        *conda_run,
        python_bin,
        "src/pre_exp/final_eval.py",
        "--engine",
        "vllm",
        "--model-name-or-path",
        str(job.model_path),
        "--dataset-name",
        dataset_name,
        "--dataset-config-name",
        dataset_config_name,
        "--split",
        split,
        "--question-field",
        question_field,
        "--answer-field",
        answer_field,
        "--exclude-subset-max-samples",
        str(exclude_subset_max_samples),
        "--exclude-subset-seed",
        str(exclude_subset_seed),
        "--subset-seed",
        str(subset_seed),
        "--max-samples",
        str(max_samples),
        "--max-new-tokens",
        str(max_new_tokens),
        "--num-rollouts",
        str(num_rollouts),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--sampling-seed",
        str(sampling_seed),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--checkpoint-label-override",
        job.label,
        "--checkpoint-step-override",
        str(job.step),
        "--output-file",
        str(job.output_file),
    ]
    if trust_remote_code:
        command.append("--trust-remote-code")
    return command


def shell_quote(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def build_plot_command(
    *,
    conda_run: list[str],
    python_bin: str,
    run_dir: Path,
    analysis_dir: Path,
    modes: list[str],
) -> list[str]:
    return [
        *conda_run,
        python_bin,
        "src/pre_exp/plot_curves.py",
        "--run-dir",
        str(run_dir),
        "--analysis-dir",
        str(analysis_dir / "eval"),
        "--output-dir",
        str(analysis_dir / "curves"),
        "--modes",
        *modes,
    ]


def guard_job_outputs(job: EvalJob, *, allow_overwrite: bool) -> None:
    if allow_overwrite:
        return
    records_file = job.output_file.with_name(job.output_file.stem + ".records.jsonl")
    for path in (job.output_file, records_file):
        if path.exists() and path.stat().st_size > 0:
            raise FileExistsError(
                f"Refusing to overwrite non-empty eval output: {path}. "
                "Pass --allow-overwrite to rerun."
            )


def run_one_job(
    *,
    gpu_id: str,
    command: list[str],
    log_file: Path,
    dry_run: bool,
) -> subprocess.Popen[Any] | None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    printable = shell_quote(command)
    if dry_run:
        print(f"dry_run gpu={gpu_id} log={log_file} command={printable}", flush=True)
        return None

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_id,
            "VLLM_USE_V1": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "LIBRARY_PATH": (
                "/usr/lib/x86_64-linux-gnu/stubs"
                + (f":{env['LIBRARY_PATH']}" if env.get("LIBRARY_PATH") else "")
            ),
        }
    )
    with log_file.open("w", encoding="utf-8") as f:
        f.write(f"gpu={gpu_id}\ncommand={printable}\n\n")
    log_handle = log_file.open("a", encoding="utf-8")
    return subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT)


def wait_for_batch(processes: list[tuple[EvalJob, subprocess.Popen[Any] | None]]) -> None:
    failed: list[str] = []
    for job, process in processes:
        if process is None:
            continue
        return_code = process.wait()
        if return_code != 0:
            failed.append(f"{job.mode}:{job.label}:exit={return_code}")
    if failed:
        raise SystemExit("At least one eval job failed: " + ", ".join(failed))


def run_job_queue(
    *,
    jobs: list[EvalJob],
    gpu_ids: list[str],
    args: argparse.Namespace,
    conda_run: list[str],
    log_dir: Path,
) -> None:
    if args.dry_run:
        for idx, job in enumerate(jobs):
            gpu_id = gpu_ids[idx % len(gpu_ids)]
            command = build_eval_command(
                conda_run=conda_run,
                python_bin=args.python_bin,
                job=job,
                dataset_name=args.dataset_name,
                dataset_config_name=args.dataset_config_name,
                split=args.split,
                question_field=args.question_field,
                answer_field=args.answer_field,
                max_samples=args.max_samples,
                exclude_subset_max_samples=args.exclude_subset_max_samples,
                exclude_subset_seed=args.exclude_subset_seed,
                subset_seed=args.subset_seed,
                max_new_tokens=args.max_new_tokens,
                num_rollouts=args.num_rollouts,
                temperature=args.temperature,
                top_p=args.top_p,
                sampling_seed=args.sampling_seed,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                gpu_memory_utilization=args.gpu_memory_utilization,
                trust_remote_code=args.trust_remote_code,
            )
            log_file = log_dir / f"{job.mode}_{job.label}.log"
            print(
                "parallel_rollout_eval_launch "
                f"mode={job.mode} label={job.label} step={job.step} gpu={gpu_id} "
                f"output={job.output_file}",
                flush=True,
            )
            run_one_job(gpu_id=gpu_id, command=command, log_file=log_file, dry_run=True)
        return

    pending: deque[EvalJob] = deque(jobs)
    available_gpus: deque[str] = deque(gpu_ids)
    running: list[tuple[str, EvalJob, subprocess.Popen[Any]]] = []
    failed: list[str] = []
    completed = 0

    while pending or running:
        while pending and available_gpus:
            gpu_id = available_gpus.popleft()
            job = pending.popleft()
            command = build_eval_command(
                conda_run=conda_run,
                python_bin=args.python_bin,
                job=job,
                dataset_name=args.dataset_name,
                dataset_config_name=args.dataset_config_name,
                split=args.split,
                question_field=args.question_field,
                answer_field=args.answer_field,
                max_samples=args.max_samples,
                exclude_subset_max_samples=args.exclude_subset_max_samples,
                exclude_subset_seed=args.exclude_subset_seed,
                subset_seed=args.subset_seed,
                max_new_tokens=args.max_new_tokens,
                num_rollouts=args.num_rollouts,
                temperature=args.temperature,
                top_p=args.top_p,
                sampling_seed=args.sampling_seed,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                gpu_memory_utilization=args.gpu_memory_utilization,
                trust_remote_code=args.trust_remote_code,
            )
            log_file = log_dir / f"{job.mode}_{job.label}.log"
            print(
                "parallel_rollout_eval_launch "
                f"mode={job.mode} label={job.label} step={job.step} gpu={gpu_id} "
                f"output={job.output_file}",
                flush=True,
            )
            process = run_one_job(
                gpu_id=gpu_id,
                command=command,
                log_file=log_file,
                dry_run=False,
            )
            if process is None:
                raise RuntimeError("run_one_job returned no process outside dry-run.")
            running.append((gpu_id, job, process))

        next_running: list[tuple[str, EvalJob, subprocess.Popen[Any]]] = []
        for gpu_id, job, process in running:
            return_code = process.poll()
            if return_code is None:
                next_running.append((gpu_id, job, process))
                continue
            available_gpus.append(gpu_id)
            completed += 1
            if return_code != 0:
                failed.append(f"{job.mode}:{job.label}:exit={return_code}")
                print(
                    "parallel_rollout_eval_job_failed "
                    f"mode={job.mode} label={job.label} gpu={gpu_id} exit={return_code}",
                    flush=True,
                )
            else:
                print(
                    "parallel_rollout_eval_job_done "
                    f"mode={job.mode} label={job.label} gpu={gpu_id} completed={completed}/{len(jobs)}",
                    flush=True,
                )
        running = next_running

        if failed:
            for _, _, process in running:
                process.terminate()
            raise SystemExit("At least one eval job failed: " + ", ".join(failed))
        if running and not (pending and available_gpus):
            time.sleep(max(args.poll_interval_seconds, 0.1))


def collect_mode_summary(mode: str, jobs: list[EvalJob], output_file: Path) -> None:
    checkpoints: list[dict[str, Any]] = []
    shared: dict[str, Any] = {}
    for job in jobs:
        payload = read_json(job.output_file)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object JSON in {job.output_file}.")
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
                "output_file": str(job.output_file),
                "records_file": str(job.output_file.with_name(job.output_file.stem + ".records.jsonl")),
                "metrics": payload.get("metrics", {}),
            }
        )
    write_json(output_file, {**shared, "mode": mode, "checkpoints": checkpoints})


def main() -> None:
    args = build_arg_parser().parse_args()
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if args.tensor_parallel_size != 1:
        raise SystemExit(
            "run_parallel_rollout_eval assigns one TP=1 vLLM process per visible GPU. "
            "Use --tensor-parallel-size 1."
        )
    conda_run = command_prefix(args.conda_run)
    run_dir = Path(args.run_root) / args.experiment_name
    analysis_dir = Path(args.result_root) / "analysis" / args.experiment_name
    log_dir = analysis_dir / "logs" / "parallel_rollout_eval"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "eval").mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    modes = list(args.modes)
    jobs_by_mode = collect_eval_jobs(
        modes=modes,
        run_dir=run_dir,
        base_model=args.base_model,
        analysis_dir=analysis_dir,
    )
    jobs = [job for mode in modes for job in jobs_by_mode[mode]]

    for job in jobs:
        guard_job_outputs(job, allow_overwrite=args.allow_overwrite or args.dry_run)

    print(
        "parallel_rollout_eval_start "
        f"experiment={args.experiment_name} modes={','.join(modes)} "
        f"gpus={','.join(gpu_ids)} tensor_parallel_size={args.tensor_parallel_size} "
        f"jobs={len(jobs)} max_samples={args.max_samples}",
        flush=True,
    )

    for mode in modes:
        mode_jobs = jobs_by_mode[mode]
        print(
            "parallel_rollout_eval_mode_plan "
            f"mode={mode} jobs={len(mode_jobs)} labels={','.join(job.label for job in mode_jobs)}",
            flush=True,
        )

    run_job_queue(
        jobs=jobs,
        gpu_ids=gpu_ids,
        args=args,
        conda_run=conda_run,
        log_dir=log_dir,
    )

    for mode in modes:
        if not args.dry_run:
            summary_file = analysis_dir / "eval" / f"checkpoint_eval_{mode}.json"
            collect_mode_summary(mode, jobs_by_mode[mode], summary_file)
            print(
                "parallel_rollout_eval_mode_done "
                f"mode={mode} summary={summary_file}",
                flush=True,
            )

    if not args.skip_plot:
        plot_command = build_plot_command(
            conda_run=conda_run,
            python_bin=args.python_bin,
            run_dir=run_dir,
            analysis_dir=analysis_dir,
            modes=modes,
        )
        if args.dry_run:
            print(f"dry_run plot_command={shell_quote(plot_command)}", flush=True)
        else:
            print(f"parallel_rollout_eval_plot_start command={shell_quote(plot_command)}", flush=True)
            subprocess.run(plot_command, check=True)
            print(f"parallel_rollout_eval_plot_done output_dir={analysis_dir / 'curves'}", flush=True)

    print("parallel_rollout_eval_done", flush=True)


if __name__ == "__main__":
    main()
