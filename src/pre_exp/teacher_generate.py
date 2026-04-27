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

from pre_exp.common import choose_subset_indices
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
    parser.add_argument(
        "--question-field",
        default="question",
        help=(
            "数据集中题目文本所在的列名。"
            "Hugging Face datasets 会把每条样本表示成 dict；"
            "把列名做成参数，是适配不同数学数据集 schema 的常见范式。"
            "例如 GSM8K 用 question，DeepScaleR-Preview-Dataset 用 problem。"
        ),
    )
    parser.add_argument(
        "--answer-field",
        default="answer",
        help=(
            "数据集中标准答案所在的列名。"
            "GSM8K 的 answer 字段包含推理和 #### 最终答案，"
            "DeepScaleR 的 answer 字段通常已经是短答案；后续会统一抽成 gold_answer。"
        ),
    )
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help=(
            "Teacher 单条回答最多生成的新 token 数。"
            "这里把默认值设成 512，而不是 256，原因是 Qwen3 在 GSM8K 上"
            "经常会输出较完整的步骤化推理 + \\boxed{} 最终答案；"
            "如果上限太小，容易在写到最终答案前被长度截断，进而显著拉低"
            "候选有效率。"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=256,
        help=(
            "Teacher 生成时一次送进 vLLM 的 prompt 数。"
            "这个值越大，单次调度的吞吐通常越高；但太大时，一次 generate 的等待时间也会更长，"
            "整体进度条更新不够频繁。"
        ),
    )
    parser.add_argument(
        "--save-every-prompts",
        type=int,
        default=1000,
        help=(
            "每处理多少条 prompt 就把当前批次结果追加写入 JSONL。"
            "这样即使中途手动停止，也能保住前面已经生成好的结果。"
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=128,
        help=(
            "vLLM scheduler 一次最多同时维护的序列数。"
            "这是 batch inference 的关键吞吐参数：值太小会让 GPU 大部分时间吃不满。"
            "vLLM 0.8.5 在未显式设置时也会回退到 128；这里显式写出，"
            "避免误用很小的调试值导致 teacher generate 看起来像卡住。"
        ),
    )
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


def append_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """把一批记录追加写入 JSONL。

    和 `write_jsonl` 的区别是：
    - `write_jsonl` 适合“一次性把全部结果整体写出”
    - 这里我们想支持长任务的增量落盘，所以需要 append 范式
    """

    if not records:
        return

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = build_arg_parser().parse_args()

    print(
        "teacher_generate_start "
        f"model={args.model_name_or_path} "
        f"dataset={args.dataset_name} "
        f"split={args.split} "
        f"max_samples={args.max_samples} "
        f"num_candidates={args.num_candidates} "
        f"prompt_batch_size={args.prompt_batch_size} "
        f"save_every_prompts={args.save_every_prompts}",
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
        if args.question_field not in sample:
            available_fields = ", ".join(sorted(sample.keys()))
            raise KeyError(
                f"Question field `{args.question_field}` is not present in dataset sample. "
                f"Available fields: {available_fields}"
            )
        if args.answer_field not in sample:
            available_fields = ", ".join(sorted(sample.keys()))
            raise KeyError(
                f"Answer field `{args.answer_field}` is not present in dataset sample. "
                f"Available fields: {available_fields}"
            )

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
        "num_candidates": args.num_candidates,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "enable_thinking": False,
        "vllm_use_v1": os.environ.get("VLLM_USE_V1", ""),
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 每次新运行都先清空旧文件，避免把不同实验的结果误追加到一起。
    output_path.write_text("", encoding="utf-8")

    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        tqdm = None

    llm = None
    pending_records: list[dict[str, object]] = []
    completed_prompts = 0
    completed_records = 0
    next_flush_prompts = args.save_every_prompts
    progress_bar = None
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

        total_prompts = len(prompts)
        if args.use_tqdm and tqdm is not None:
            progress_bar = tqdm(total=total_prompts, desc="teacher_generate_prompts")

        for batch_start in range(0, total_prompts, args.prompt_batch_size):
            batch_end = min(batch_start + args.prompt_batch_size, total_prompts)
            batch_prompt_payloads = prompt_payloads[batch_start:batch_end]
            batch_prompts = prompts[batch_start:batch_end]

            print(
                "teacher_generate_batch_start "
                f"prompt_range=[{batch_start},{batch_end}) "
                f"batch_prompts={len(batch_prompts)} "
                f"completed_prompts={completed_prompts}/{total_prompts}",
                flush=True,
            )

            # vLLM 的 LLM.generate 是阻塞调用：只有整个 batch 返回后，外层 Python
            # 才能继续执行。因此如果只更新外层 tqdm，大 batch 会长时间看不到进度。
            # 这里把 use_tqdm 透传给 vLLM，让 engine 在 batch 内部按完成的请求刷新进度。
            # 常用范式：外层 tqdm 负责“全局已完成多少 prompt”，库内部 tqdm 负责
            # “当前阻塞调用内部是否还在推进”。
            outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=args.use_tqdm)

            batch_records: list[dict[str, object]] = []
            for sample_payload, output in zip(batch_prompt_payloads, outputs):
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

            completed_prompts += len(batch_prompts)
            pending_records.extend(batch_records)

            if progress_bar is not None:
                progress_bar.update(len(batch_prompts))
                progress_bar.set_postfix(
                    completed_prompts=completed_prompts,
                    buffered_records=len(pending_records),
                )

            # 这里不能写成 `completed_prompts % save_every_prompts == 0`。
            # 原因是 prompt_batch_size 和 save_every_prompts 往往不是整除关系；
            # 例如 256 一批、每 1000 条落盘，那么 completed_prompts 会走
            # 256/512/768/1024/...，永远碰不到“恰好等于 1000”。
            #
            # 更稳的常用范式是“跨过阈值就 flush 一次”，而不是“只在精确命中阈值时 flush”。
            should_flush = (
                completed_prompts >= next_flush_prompts
                or completed_prompts >= total_prompts
            )
            if should_flush:
                append_jsonl(output_path, pending_records)
                completed_records += len(pending_records)
                print(
                    "teacher_generate_flush "
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
        "teacher_generate_done "
        f"prompts={len(prompt_payloads)} "
        f"records={completed_records} "
        f"output_file={args.output_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
