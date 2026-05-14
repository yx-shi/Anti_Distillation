from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sft.hf_cache import ensure_writable_hf_datasets_cache
from sft.prompting import build_qwen3_messages, render_qwen3_prompt
from sft.rollout_eval import extract_gsm8k_final_answer
from vllm_dual_decoding.common import append_jsonl, choose_subset_indices


DEFAULT_TEACHER_MODEL = "/home/disk1/public_checkpoint/Qwen3-8B"
DEFAULT_STUDENT_MODEL = "/home/disk1/public_checkpoint/Qwen3-1.7B"
DEFAULT_OUTPUT_FILE = "result/vllm_dual_decoding/candidates/smoke/candidate_pool.jsonl"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate token-level vLLM-dual teacher outputs for data-side smoke."
    )
    parser.add_argument("--model-name-or-path", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument(
        "--generation-mode",
        choices=["plain", "hard", "soft"],
        default="plain",
        help="`plain` uses ordinary vLLM; `hard` and `soft` enable vLLM-dual.",
    )
    parser.add_argument("--student-model-name-or-path", default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config-name", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--save-every-prompts", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--small-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--small-gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--hard-candidate-top-k", type=int, default=20)
    parser.add_argument("--hard-candidate-top-p", type=float, default=0.95)
    parser.add_argument("--soft-student-weight", type=float, default=1.0)
    parser.add_argument("--soft-temperature", type=float, default=1.0)
    parser.add_argument("--debug-log-interval", type=int, default=16)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-remote-model-files", action="store_true")
    parser.add_argument("--use-tqdm", action="store_true")
    return parser


def generation_mode_label(mode: str) -> str:
    if mode == "plain":
        return "teacher_plain"
    if mode == "hard":
        return "teacher_token_hard"
    if mode == "soft":
        return "teacher_token_soft"
    raise ValueError(f"Unsupported generation mode: {mode}")


def build_dual_model_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.generation_mode == "plain":
        return None

    return {
        "small_model_engine_args": {
            "model": args.student_model_name_or_path,
            "dtype": args.dtype,
            "tensor_parallel_size": args.small_tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.small_gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "trust_remote_code": args.trust_remote_code,
        },
        "adversarial_mode": args.generation_mode,
        "hard_candidate_top_k": args.hard_candidate_top_k,
        "hard_candidate_top_p": args.hard_candidate_top_p,
        "soft_student_weight": args.soft_student_weight,
        "soft_temperature": args.soft_temperature,
        "debug_log_interval": args.debug_log_interval,
    }


def shutdown_llm(llm: object) -> None:
    engine = getattr(llm, "llm_engine", None)
    executor = getattr(engine, "model_executor", None)
    if executor is not None and hasattr(executor, "shutdown"):
        executor.shutdown()
    del llm
    gc.collect()


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}"
        )

    mode_label = generation_mode_label(args.generation_mode)
    adversarial_mode = None if args.generation_mode == "plain" else args.generation_mode
    dual_model_config = build_dual_model_config(args)
    record_dual_model_config = (
        json.loads(json.dumps(dual_model_config, ensure_ascii=False))
        if dual_model_config is not None
        else None
    )

    print(
        "vllm_dual_teacher_generate_start "
        f"model={args.model_name_or_path} "
        f"student={args.student_model_name_or_path} "
        f"dataset={args.dataset_name} "
        f"split={args.split} "
        f"generation_mode={mode_label} "
        f"max_samples={args.max_samples} "
        f"num_shards={args.num_shards} "
        f"shard_index={args.shard_index}",
        flush=True,
    )

    ensure_writable_hf_datasets_cache()
    dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.split)
    subset_indices = choose_subset_indices(len(dataset), args.max_samples, args.subset_seed)
    total_subset_samples = len(subset_indices)
    subset_indices = subset_indices[args.shard_index::args.num_shards]
    subset_samples = [dataset[int(dataset_idx)] for dataset_idx in subset_indices]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=not args.allow_remote_model_files,
    )
    prompt_payloads: list[dict[str, object]] = []
    prompts: list[str] = []
    for dataset_idx, sample in zip(subset_indices, subset_samples):
        if args.question_field not in sample or args.answer_field not in sample:
            available_fields = ", ".join(sorted(sample.keys()))
            raise KeyError(f"Missing question/answer field. Available fields: {available_fields}")
        question = str(sample[args.question_field]).strip()
        raw_answer = str(sample[args.answer_field]).strip()
        messages = build_qwen3_messages(question)
        prompt_text = render_qwen3_prompt(
            tokenizer=tokenizer,
            messages=messages,
            enable_thinking=False,
        )
        prompt_payloads.append(
            {
                "sample_id": int(dataset_idx),
                "question": question,
                "gold_answer": extract_gsm8k_final_answer(raw_answer),
                "gold_answer_raw": raw_answer,
                "messages": messages,
                "prompt_text": prompt_text,
            }
        )
        prompts.append(prompt_text)

    os.environ.setdefault("VLLM_USE_V1", "0")
    try:
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError as exc:
        raise SystemExit("未检测到 vllm。请确认当前环境已安装 vllm==0.8.5。") from exc

    sampling_params = SamplingParams(
        n=args.num_candidates,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    generation_config = {
        "generation_mode": mode_label,
        "adversarial_mode": adversarial_mode,
        "num_candidates": args.num_candidates,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "enable_thinking": False,
        "vllm_use_v1": os.environ.get("VLLM_USE_V1", ""),
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        tqdm = None

    llm = None
    progress_bar = None
    pending_records: list[dict[str, Any]] = []
    completed_prompts = 0
    completed_records = 0
    next_flush_prompts = args.save_every_prompts
    try:
        llm_kwargs: dict[str, Any] = {
            "model": args.model_name_or_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "trust_remote_code": args.trust_remote_code,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "generation_config": "vllm",
        }
        if dual_model_config is not None:
            llm_kwargs["dual_model_config"] = dual_model_config

        llm = LLM(**llm_kwargs)
        vllm_config = llm.llm_engine.vllm_config
        worker_cls = str(vllm_config.parallel_config.worker_cls)
        effective_dual_config = getattr(vllm_config, "dual_model_config", None)
        effective_adversarial_mode = (
            getattr(effective_dual_config, "adversarial_mode", None)
            if effective_dual_config is not None
            else None
        )
        print(
            "VLLM_DUAL_PIPELINE_EFFECTIVE "
            f"generation_mode={mode_label} "
            f"worker_cls={worker_cls} "
            f"adv_mode={effective_adversarial_mode}",
            flush=True,
        )

        total_prompts = len(prompts)
        if args.use_tqdm and tqdm is not None:
            progress_bar = tqdm(total=total_prompts, desc="vllm_dual_generate_prompts")

        for batch_start in range(0, total_prompts, args.prompt_batch_size):
            batch_end = min(batch_start + args.prompt_batch_size, total_prompts)
            outputs = llm.generate(
                prompts[batch_start:batch_end],
                sampling_params,
                use_tqdm=args.use_tqdm,
            )

            batch_records: list[dict[str, Any]] = []
            for sample_payload, output in zip(prompt_payloads[batch_start:batch_end], outputs):
                prompt_token_ids = getattr(output, "prompt_token_ids", None)
                for candidate_idx, candidate in enumerate(output.outputs):
                    completion_token_ids = getattr(candidate, "token_ids", None)
                    batch_records.append(
                        {
                            "sample_id": sample_payload["sample_id"],
                            "question": sample_payload["question"],
                            "gold_answer": sample_payload["gold_answer"],
                            "gold_answer_raw": sample_payload["gold_answer_raw"],
                            "messages": sample_payload["messages"],
                            "prompt_text": sample_payload["prompt_text"],
                            "candidate_id": candidate_idx,
                            "num_shards": args.num_shards,
                            "shard_index": args.shard_index,
                            "candidate_text": candidate.text.strip(),
                            "generation_mode": mode_label,
                            "adversarial_mode": adversarial_mode,
                            "dual_model_config": record_dual_model_config,
                            "worker_cls": worker_cls,
                            "generation_config": generation_config,
                            "finish_reason": getattr(candidate, "finish_reason", None),
                            "stop_reason": getattr(candidate, "stop_reason", None),
                            "num_prompt_tokens": (
                                len(prompt_token_ids) if prompt_token_ids is not None else None
                            ),
                            "num_generated_tokens": (
                                len(completion_token_ids) if completion_token_ids is not None else None
                            ),
                        }
                    )

            completed_prompts += len(outputs)
            pending_records.extend(batch_records)
            if progress_bar is not None:
                progress_bar.update(len(outputs))

            should_flush = (
                completed_prompts >= next_flush_prompts
                or completed_prompts >= total_prompts
            )
            if should_flush:
                append_jsonl(output_path, pending_records)
                completed_records += len(pending_records)
                print(
                    "vllm_dual_teacher_generate_flush "
                    f"completed_prompts={completed_prompts}/{total_prompts} "
                    f"flushed_records={len(pending_records)} "
                    f"written_records={completed_records} "
                    f"output_file={output_path}",
                    flush=True,
                )
                pending_records.clear()
                while next_flush_prompts <= completed_prompts:
                    next_flush_prompts += args.save_every_prompts
    finally:
        if progress_bar is not None:
            progress_bar.close()
        if llm is not None:
            shutdown_llm(llm)

    print(
        "vllm_dual_teacher_generate_done "
        f"prompts={len(prompt_payloads)} "
        f"total_subset_prompts={total_subset_samples} "
        f"num_shards={args.num_shards} "
        f"shard_index={args.shard_index} "
        f"records={completed_records} "
        f"output_file={args.output_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
