from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pre_exp.common import choose_subset_indices, write_jsonl
from sft.hf_cache import ensure_writable_hf_datasets_cache
from sft.prompting import build_qwen3_messages, render_qwen3_prompt
from sft.rollout_eval import extract_gsm8k_final_answer


DEFAULT_TEACHER_MODEL = "/data1/public_checkpoints/Qwen3-8B"
DEFAULT_OUTPUT_FILE = "result/pre_exp/candidates/smoke/candidate_pool.jsonl"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Qwen3 teacher candidate responses for the smoke pre-experiment.")
    parser.add_argument("--model-name-or-path", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config-name", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-remote-model-files", action="store_true")
    parser.add_argument("--use-tqdm", action="store_true")
    return parser


def shutdown_llm(llm: object) -> None:
    """尽量显式回收 vLLM engine 相关资源。"""

    engine = getattr(llm, "llm_engine", None)
    executor = getattr(engine, "model_executor", None)
    if executor is not None and hasattr(executor, "shutdown"):
        executor.shutdown()
    del llm
    gc.collect()


def main() -> None:
    args = build_arg_parser().parse_args()

    print(
        "teacher_generate_start "
        f"model={args.model_name_or_path} "
        f"dataset={args.dataset_name} "
        f"split={args.split} "
        f"max_samples={args.max_samples} "
        f"num_candidates={args.num_candidates}",
        flush=True,
    )

    ensure_writable_hf_datasets_cache()
    dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.split)
    subset_indices = choose_subset_indices(len(dataset), args.max_samples, args.subset_seed)
    subset_samples = [dataset[int(dataset_idx)] for dataset_idx in subset_indices]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=not args.allow_remote_model_files,
    )
    prompt_payloads: list[dict[str, object]] = []
    prompts: list[str] = []

    for dataset_idx, sample in zip(subset_indices, subset_samples):
        question = sample["question"].strip()
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
                "gold_answer": extract_gsm8k_final_answer(sample["answer"]),
                "gold_answer_raw": sample["answer"].strip(),
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
        "num_candidates": args.num_candidates,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "enable_thinking": False,
        "vllm_use_v1": os.environ.get("VLLM_USE_V1", ""),
    }

    llm = None
    records: list[dict[str, object]] = []
    try:
        llm = LLM(
            model=args.model_name_or_path,
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

        for sample_payload, output in zip(prompt_payloads, outputs):
            prompt_token_ids = getattr(output, "prompt_token_ids", None)
            for candidate_idx, candidate in enumerate(output.outputs):
                completion_token_ids = getattr(candidate, "token_ids", None)
                records.append(
                    {
                        "sample_id": sample_payload["sample_id"],
                        "question": sample_payload["question"],
                        "gold_answer": sample_payload["gold_answer"],
                        "gold_answer_raw": sample_payload["gold_answer_raw"],
                        "messages": sample_payload["messages"],
                        "prompt_text": sample_payload["prompt_text"],
                        "candidate_id": candidate_idx,
                        "candidate_text": candidate.text.strip(),
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
    finally:
        if llm is not None:
            shutdown_llm(llm)

    write_jsonl(args.output_file, records)
    print(
        "teacher_generate_done "
        f"prompts={len(prompt_payloads)} "
        f"records={len(records)} "
        f"output_file={args.output_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
