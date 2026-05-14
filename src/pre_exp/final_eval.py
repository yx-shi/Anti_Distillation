from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pre_exp.common import choose_holdout_indices, write_json, write_jsonl
from sft.hf_cache import ensure_writable_hf_datasets_cache
from sft.rollout_eval import build_rollout_eval_samples, grade_multi_rollout_predictions


DEFAULT_MODEL_PATH = "result/pre_exp/runs/smoke/teacher_baseline/final_checkpoint"
DEFAULT_OUTPUT_FILE = "result/pre_exp/analysis/smoke/final_eval_teacher_baseline.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline rollout grading on one or more checkpoints.")
    parser.add_argument("--engine", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint-root", default="")
    parser.add_argument("--checkpoint-glob", default="checkpoint-step-*")
    parser.add_argument("--dataset-name", default="agentica-org/DeepScaleR-Preview-Dataset")
    parser.add_argument("--dataset-config-name", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--question-field", default="problem")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument(
        "--exclude-subset-max-samples",
        type=int,
        default=0,
        help=(
            "If positive, exclude the deterministic subset chosen from the eval dataset with "
            "this max_samples value before sampling eval examples. Used for DeepScaleR holdout."
        ),
    )
    parser.add_argument(
        "--exclude-subset-seed",
        type=int,
        default=42,
        help="Seed paired with --exclude-subset-max-samples when defining the excluded subset.",
    )
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument(
        "--output-prefix",
        default="",
        help=(
            "Prefix used for per-checkpoint files when --checkpoint-root evaluates multiple "
            "checkpoints. Defaults to the --output-file stem, so different mode summary files "
            "produce different per-checkpoint names. Single-checkpoint output is unchanged."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-tqdm", action="store_true")
    parser.add_argument("--allow-remote-model-files", action="store_true")
    parser.add_argument("--checkpoint-label-override", default="")
    parser.add_argument("--checkpoint-step-override", type=int, default=None)
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def checkpoint_label(checkpoint_path: Path) -> str:
    name = checkpoint_path.name
    if name.startswith("checkpoint-step-"):
        return name.removeprefix("checkpoint-step-")
    return name


def checkpoint_step(checkpoint_path: Path) -> int | None:
    name = checkpoint_path.name
    if not name.startswith("checkpoint-step-"):
        return None
    try:
        return int(name.removeprefix("checkpoint-step-"))
    except ValueError:
        return None


def resolved_checkpoint_label(checkpoint_path: Path, args: argparse.Namespace) -> str:
    return args.checkpoint_label_override or checkpoint_label(checkpoint_path)


def resolved_checkpoint_step(checkpoint_path: Path, args: argparse.Namespace) -> int | None:
    if args.checkpoint_step_override is not None:
        return args.checkpoint_step_override
    return checkpoint_step(checkpoint_path)


def safe_stem_component(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    return cleaned.strip("._-") or "final_eval"


def per_checkpoint_output_file(
    output_file: Path,
    output_prefix: str,
    checkpoint_path: Path,
) -> Path:
    prefix = safe_stem_component(output_prefix or output_file.stem)
    label = safe_stem_component(checkpoint_label(checkpoint_path))
    return output_file.parent / f"{prefix}_{label}.json"


def resolve_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    if not args.checkpoint_root:
        return [Path(args.model_name_or_path)]

    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_paths = sorted(path for path in checkpoint_root.glob(args.checkpoint_glob) if path.is_dir())

    final_checkpoint = checkpoint_root / "final_checkpoint"
    if final_checkpoint.is_dir() and final_checkpoint not in checkpoint_paths:
        checkpoint_paths.append(final_checkpoint)

    if not checkpoint_paths:
        raise SystemExit(
            f"No checkpoints found under {checkpoint_root} with glob `{args.checkpoint_glob}`."
        )
    return checkpoint_paths


def shutdown_llm(llm: object) -> None:
    """尽量显式回收 vLLM engine。"""

    engine = getattr(llm, "llm_engine", None)
    executor = getattr(engine, "model_executor", None)
    if executor is not None and hasattr(executor, "shutdown"):
        executor.shutdown()
    del llm
    gc.collect()


def load_eval_dataset(args: argparse.Namespace) -> Any:
    ensure_writable_hf_datasets_cache()
    return load_dataset(args.dataset_name, args.dataset_config_name, split=args.split)


def resolve_candidate_indices(dataset: Any, args: argparse.Namespace) -> list[int] | None:
    if args.exclude_subset_max_samples <= 0:
        return None
    return choose_holdout_indices(
        len(dataset),
        exclude_max_samples=args.exclude_subset_max_samples,
        exclude_seed=args.exclude_subset_seed,
        max_samples=0,
        subset_seed=args.subset_seed,
    )


def run_hf_rollout_eval(
    checkpoint_path: Path,
    dataset: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        local_files_only=not args.allow_remote_model_files,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype="auto",
        local_files_only=not args.allow_remote_model_files,
    ).to(device)
    model.eval()

    samples = build_rollout_eval_samples(
        tokenizer=tokenizer,
        eval_source_dataset=dataset,
        max_samples=args.max_samples,
        subset_seed=args.subset_seed,
        question_field=args.question_field,
        answer_field=args.answer_field,
        candidate_indices=resolve_candidate_indices(dataset, args),
    )
    prompts = [sample["prompt_text"] for sample in samples]

    torch.manual_seed(args.sampling_seed)
    generated_texts_by_sample: list[list[str]] = []
    with torch.inference_mode():
        for prompt_text in prompts:
            batch = tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            output_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0.0,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.num_rollouts,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            sample_texts: list[str] = []
            for sequence_ids in output_ids:
                generated_ids = sequence_ids[batch["input_ids"].size(1):]
                sample_texts.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
            generated_texts_by_sample.append(sample_texts)

    metrics, eval_records = grade_multi_rollout_predictions(samples, generated_texts_by_sample)
    return metrics, eval_records


def run_vllm_rollout_eval(
    checkpoint_path: Path,
    dataset: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        local_files_only=not args.allow_remote_model_files,
    )
    samples = build_rollout_eval_samples(
        tokenizer=tokenizer,
        eval_source_dataset=dataset,
        max_samples=args.max_samples,
        subset_seed=args.subset_seed,
        question_field=args.question_field,
        answer_field=args.answer_field,
        candidate_indices=resolve_candidate_indices(dataset, args),
    )
    prompts = [sample["prompt_text"] for sample in samples]

    os.environ.setdefault("VLLM_USE_V1", "0")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError as exc:
        raise SystemExit("未检测到 vllm。请确认当前环境已安装 vllm==0.8.5。") from exc

    sampling_params = SamplingParams(
        n=args.num_rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.sampling_seed,
        max_tokens=args.max_new_tokens,
    )

    llm = None
    try:
        llm = LLM(
            model=str(checkpoint_path),
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            generation_config="vllm",
        )
        outputs = llm.generate(prompts, sampling_params, use_tqdm=args.use_tqdm)
        generated_texts_by_sample = [
            [completion.text.strip() for completion in output.outputs]
            for output in outputs
        ]
    finally:
        if llm is not None:
            shutdown_llm(llm)

    metrics, eval_records = grade_multi_rollout_predictions(samples, generated_texts_by_sample)
    return metrics, eval_records


def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if args.engine == "hf":
        metrics, eval_records = run_hf_rollout_eval(
            checkpoint_path=checkpoint_path,
            dataset=dataset,
            args=args,
            device=device,
        )
    else:
        metrics, eval_records = run_vllm_rollout_eval(
            checkpoint_path=checkpoint_path,
            dataset=dataset,
            args=args,
        )

    payload = {
        "engine": args.engine,
        "model_name_or_path": str(checkpoint_path),
        "checkpoint_label": resolved_checkpoint_label(checkpoint_path, args),
        "checkpoint_step": resolved_checkpoint_step(checkpoint_path, args),
        "dataset_name": args.dataset_name,
        "dataset_config_name": args.dataset_config_name,
        "split": args.split,
        "question_field": args.question_field,
        "answer_field": args.answer_field,
        "max_new_tokens": args.max_new_tokens,
        "num_rollouts": args.num_rollouts,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "sampling_seed": args.sampling_seed,
        "max_samples": args.max_samples,
        "subset_seed": args.subset_seed,
        "exclude_subset_max_samples": args.exclude_subset_max_samples,
        "exclude_subset_seed": args.exclude_subset_seed,
        "eligible_sample_count": (
            len(resolve_candidate_indices(dataset, args))
            if args.exclude_subset_max_samples > 0
            else len(dataset)
        ),
        "metrics": metrics,
    }
    return payload, eval_records


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.num_rollouts <= 0:
        raise SystemExit("--num-rollouts must be positive.")
    device = resolve_device(args.device)
    dataset = load_eval_dataset(args)
    checkpoint_paths = resolve_checkpoint_paths(args)

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        checkpoint_payload, eval_records = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            dataset=dataset,
            args=args,
            device=device,
        )

        if len(checkpoint_paths) == 1:
            checkpoint_output_file = output_file
        else:
            checkpoint_output_file = per_checkpoint_output_file(
                output_file=output_file,
                output_prefix=args.output_prefix,
                checkpoint_path=checkpoint_path,
            )

        records_output_file = checkpoint_output_file.with_name(
            checkpoint_output_file.stem + ".records.jsonl"
        )
        write_json(checkpoint_output_file, checkpoint_payload)
        write_jsonl(records_output_file, eval_records)

        summary_rows.append(
            {
                "checkpoint_label": checkpoint_payload["checkpoint_label"],
                "checkpoint_step": checkpoint_payload["checkpoint_step"],
                "model_name_or_path": checkpoint_payload["model_name_or_path"],
                "output_file": str(checkpoint_output_file),
                "records_file": str(records_output_file),
                "metrics": checkpoint_payload["metrics"],
            }
        )

        print(
            "final_eval_done "
            f"engine={args.engine} "
            f"checkpoint={checkpoint_path} "
            f"rollout_acc={checkpoint_payload['metrics']['rollout_acc']:.4f} "
            f"rollout_acc_std={checkpoint_payload['metrics'].get('rollout_acc_std', 0.0):.4f} "
            f"output_file={checkpoint_output_file}",
            flush=True,
        )

    if len(checkpoint_paths) > 1:
        write_json(
            output_file,
            {
                "engine": args.engine,
                "dataset_name": args.dataset_name,
                "dataset_config_name": args.dataset_config_name,
                "split": args.split,
                "question_field": args.question_field,
                "answer_field": args.answer_field,
                "max_new_tokens": args.max_new_tokens,
                "num_rollouts": args.num_rollouts,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "sampling_seed": args.sampling_seed,
                "max_samples": args.max_samples,
                "subset_seed": args.subset_seed,
                "exclude_subset_max_samples": args.exclude_subset_max_samples,
                "exclude_subset_seed": args.exclude_subset_seed,
                "checkpoints": summary_rows,
            },
        )


if __name__ == "__main__":
    main()
