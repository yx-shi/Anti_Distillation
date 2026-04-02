"""A teaching-oriented vLLM offline inference demo for the Anti Distillation repo.

这个脚本刻意不依赖 `src/sft/` 里的训练逻辑，目的是把 vLLM 的“原生离线推理”
workflow 单独讲清楚：

1. 准备一批 prompt
2. 配置采样参数 `SamplingParams`
3. 初始化离线推理入口 `LLM`
4. 调用 `llm.generate(...)`
5. 打印结果，并把结果保存成 JSONL，便于后续评测/分析

设计取向：
- 尽量贴近真实科研/工程里的 batch inference 用法
- 用详尽注释说明常见参数、常用范式和调参思路
- 不和当前训练框架耦合，后续可以自然扩展到 online serving / benchmark
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path


# 默认直接指向你当前项目里已经在用的本地 Qwen3-1.7B 权重目录。
# 常见范式：
# - 开发阶段用本地路径，避免每次都从 HF Hub 下载
# - 后期切换到其他模型时，通过命令行 `--model` 覆盖，而不是改代码常量
DEFAULT_MODEL_PATH = "/data1/public_checkpoints/Qwen3-1.7B"

# 离线推理通常会“一次喂多条 prompt”，这里内置一组最小示例。
# 如果你传入 `--prompt-file` 或多个 `--prompt`，会覆盖/补充这组默认样例。
DEFAULT_PROMPTS = [
    "Please explain knowledge distillation in two short sentences.",
    "Finish the sentence: The capital of France is",
    "用一句话解释为什么 KL divergence 常被用于蒸馏目标。",
]

# JSONL 是科研工程里非常常见的落盘格式：
# - 一行一个样本，便于流式写入
# - 后续评测脚本、统计脚本更容易逐条读取
DEFAULT_OUTPUT_FILE = Path("result/vllm_offline_demo_outputs.jsonl")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    这里故意把 CLI 参数写得比较完整，因为“把实验配置参数化”本身就是科研代码里
    非常重要的范式：同一个脚本可以在不改源码的前提下重复做不同实验。
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal offline vLLM demo with a local Hugging Face-compatible model."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=(
            "模型路径或 Hugging Face Hub 模型名。默认使用本地 "
            f"{DEFAULT_MODEL_PATH}。"
        ),
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help=(
            "手动追加一条 prompt。这个参数可以重复写多次，形成一个 batch。"
        ),
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help=(
            "从文本文件读取 prompt；文件里每一行视为一条 prompt，空行和以 # 开头的行会被忽略。"
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="把生成结果保存成 JSONL 的路径。",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help=(
            "张量并行大小。单卡设为 1；双卡/多卡时设为可见 GPU 数。"
        ),
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help=(
            "模型权重/计算 dtype。Qwen3 默认常用 bfloat16；显存紧张时也可尝试 float16。"
        ),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help=(
            "vLLM 预留 KV cache 时采用的最大上下文长度。调小能显著省显存。"
        ),
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=4,
        help=(
            "同时处理的最大序列数。调小通常也能省显存，适合 first run。"
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.3,
        help=(
            "允许 vLLM 使用的 GPU 显存比例。first run 建议不要一开始就打满。"
        ),
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help=(
            "关闭 CUDA graph，退回 eager 模式。更省一部分显存，但速度可能更慢。"
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "是否允许加载模型仓库中的自定义 Python 代码。只有在你确认模型来源可信时再开启。"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "采样温度。0 表示 greedy decoding，更利于教学和复现实验。"
        ),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="nucleus sampling 的 top-p。temperature=0 时它通常不会起主要作用。",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="重复惩罚系数；1.0 表示不额外惩罚重复。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="每条 prompt 最多新生成多少个 token。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。对采样实验很有用；做可复现实验时建议显式固定。",
    )
    parser.add_argument(
        "--use-tqdm",
        action="store_true",
        help="是否显示 vLLM 内部的 tqdm 进度条。",
    )
    return parser.parse_args()


def load_prompts(prompt_file: Path | None, extra_prompts: list[str]) -> list[str]:
    """收集本次离线推理要跑的 prompt 列表。

    典型 offline inference 范式就是：
    - 先把一批输入准备好
    - 再一次性调用 `llm.generate(batch_prompts, sampling_params)`
    这样才能真正利用 vLLM 的批处理能力。
    """

    prompts: list[str] = []

    if prompt_file is not None:
        file_prompts = [
            line.strip()
            for line in prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        prompts.extend(file_prompts)

    prompts.extend(prompt.strip() for prompt in extra_prompts if prompt.strip())

    if not prompts:
        prompts.extend(DEFAULT_PROMPTS)

    return prompts


def print_run_summary(args: argparse.Namespace, prompts: list[str]) -> None:
    """打印本次运行配置。

    科研代码里，一个非常好的习惯是：在真正开跑前，把关键配置打印出来。
    这样你回头看日志时，能快速知道这次实验到底是怎么配的。
    """

    print("=== vLLM Offline Demo Configuration ===")
    print(f"model={args.model}")
    print(f"prompt_count={len(prompts)}")
    print(f"tensor_parallel_size={args.tensor_parallel_size}")
    print(f"dtype={args.dtype}")
    print(f"max_model_len={args.max_model_len}")
    print(f"max_num_seqs={args.max_num_seqs}")
    print(f"gpu_memory_utilization={args.gpu_memory_utilization}")
    print(f"enforce_eager={args.enforce_eager}")
    print(f"temperature={args.temperature}")
    print(f"top_p={args.top_p}")
    print(f"repetition_penalty={args.repetition_penalty}")
    print(f"max_tokens={args.max_tokens}")
    print(f"seed={args.seed}")
    print(f"output_file={args.output_file}")
    print(
        "CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', '(not set; vLLM will see default visible GPUs)')}"
    )
    print(f"VLLM_USE_V1={os.environ.get('VLLM_USE_V1')}")


def save_results(output_file: Path, records: list[dict[str, object]]) -> None:
    """把结果保存成 JSONL。

    这里用最朴素的写法，不依赖 pandas 等额外库，优点是：
    - 环境更干净
    - 结果格式足够通用
    - 后续无论你用 Python、jq 还是 shell 都很容易继续处理
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def shutdown_llm(llm: object) -> None:
    """尽量显式清理 vLLM 的多进程执行器。

    背景：
    - 多卡 tensor parallel 在 vLLM V0 里通常会走 multiprocessing + NCCL
    - 如果 Python 解释器直接退出，底层进程组和 shared memory 有时会在
      垃圾回收阶段才被动释放，于是 PyTorch / multiprocessing 可能打印
      `destroy_process_group` 或 `resource_tracker` 的清理警告

    这里采用“best effort cleanup”范式：
    - 如果底层 executor 暴露了 shutdown，就主动调用
    - 之后再 `del llm` + `gc.collect()`，尽量让资源更早释放

    这种写法的核心思想不是“依赖析构函数碰运气”，而是显式表达：
    “我这个推理引擎用完了，现在就开始清理。”
    """

    engine = getattr(llm, "llm_engine", None)
    executor = getattr(engine, "model_executor", None)

    if executor is not None and hasattr(executor, "shutdown"):
        executor.shutdown()

    del llm
    gc.collect()


def main() -> None:
    args = parse_args()

    if args.prompt_file is not None and not args.prompt_file.exists():
        raise SystemExit(
            f"Prompt file not found: {args.prompt_file}"
        )

    prompts = load_prompts(args.prompt_file, args.prompt)

    # 这个环境变量必须在 `import vllm` 之前设置。
    # 你在 AGENTS.md 里要求使用 v0 路径，所以这里用 setdefault：
    # - 如果你在 shell 里显式写了 `VLLM_USE_V1=0`，这里不会覆盖
    # - 如果你忘了写，脚本也会自动兜底成 0
    os.environ.setdefault("VLLM_USE_V1", "0")

    # 把 import 放到这里，是一个很实用的小技巧：
    # - 这样 `python script.py --help` 在没安装 vllm 时也能正常显示帮助
    # - 报错信息也能更贴近“安装缺失”这个真实原因
    try:
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "未检测到 vllm。请先按 examples/vllm_offline/README.md 安装 vllm==0.8.5。"
        ) from exc

    print_run_summary(args, prompts)

    # SamplingParams 控制“怎么生成”，而不是“加载什么模型”。
    # 这是一个重要抽象：模型本体和解码策略通常分离管理。
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # LLM 是 vLLM 离线推理的核心入口。
    # 这里传入的几个参数，基本就是 first run 最常调整的一批：
    # - tensor_parallel_size: 多卡张量并行
    # - max_model_len / max_num_seqs: 控制 KV cache 和 batch 上限，常用于“先跑通”
    # - gpu_memory_utilization: 给 vLLM 的显存预算
    # - enforce_eager: OOM 或 CUDA graph 相关问题时的常见排障开关
    # - generation_config="vllm": 不沿用 HF 仓库里的 generation_config.json，
    #   这样更适合教学，因为生成行为主要由你这里显式写出的 SamplingParams 决定
    llm = None
    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            generation_config="vllm",
        )

        # `llm.generate` 接收一个 prompt 列表，并返回与之对应的输出列表。
        # 这就是 offline inference 的最经典调用方式。
        outputs = llm.generate(prompts,
                               sampling_params,
                               use_tqdm=args.use_tqdm)

        records: list[dict[str, object]] = []
        print("\n=== Generation Results ===")

        for sample_idx, output in enumerate(outputs, start=1):
            # vLLM 默认每个 prompt 返回 n=1 个候选，这里取第一个即可。
            completion = output.outputs[0]

            prompt_token_ids = getattr(output, "prompt_token_ids", None)
            completion_token_ids = getattr(completion, "token_ids", None)

            record = {
                "sample_id": sample_idx,
                "prompt": output.prompt,
                "generated_text": completion.text,
                "finish_reason": getattr(completion, "finish_reason", None),
                "stop_reason": getattr(completion, "stop_reason", None),
                "num_prompt_tokens": (
                    len(prompt_token_ids) if prompt_token_ids is not None else None
                ),
                "num_generated_tokens": (
                    len(completion_token_ids) if completion_token_ids is not None else None
                ),
            }
            records.append(record)

            print(f"\n--- Sample {sample_idx} ---")
            print(f"Prompt: {record['prompt']}")
            print(f"Generated: {record['generated_text']}")
            print(f"finish_reason={record['finish_reason']}")
            print(f"num_prompt_tokens={record['num_prompt_tokens']}")
            print(f"num_generated_tokens={record['num_generated_tokens']}")

        save_results(args.output_file, records)
        print(f"\nSaved JSONL results to: {args.output_file}")
    finally:
        if llm is not None:
            shutdown_llm(llm)


if __name__ == "__main__":
    main()
