"""Standalone GSM8K rollout + grading script.

这个脚本的目标不是训练，而是把“模型生成 -> 提取答案 -> 提取 GSM8K gold -> 判分”
这条链路单独跑通。研究里常见的一个好习惯是：
先把评测链路做成独立脚本验证通过，再把它接回训练框架。
这样排错时更容易区分“模型没学会”和“评测管线接错了”。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 当我们用 `python src/run_grading_eval.py` 运行脚本时，Python 默认只把 `src/`
# 放进模块搜索路径。这里显式把项目根目录也加进去，这样才能 import 顶层的 `grading` 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI for standalone rollout grading."""

    parser = argparse.ArgumentParser(description="Generate one GSM8K sample and grade it with the local grading pipeline.")
    parser.add_argument("--model-name-or-path", default="/data1/public_checkpoints/Qwen3-1.7B")
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config-name", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-remote-model-files",
        action="store_true",
        help="If set, allow tokenizer/model files to be fetched remotely instead of requiring local files.",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Optional manual question. If set, the script skips dataset loading and uses this question directly.",
    )
    parser.add_argument(
        "--gold-answer",
        default=None,
        help="Optional manual gold answer. For GSM8K-style answers, both full `reasoning + #### answer` and plain final answers are accepted.",
    )
    parser.add_argument(
        "--model-output",
        default=None,
        help="Optional manual model output. If set, the script skips model loading/generation and only runs extraction + grading.",
    )
    return parser


def build_gsm8k_prompt(question: str) -> str:
    """Match the prompt template used in the current SFT pipeline.

    这里重复写一份模板，而不是 import 训练代码里的 helper，
    是为了让这个脚本保持“独立评测工具”的属性：
    它不需要先把训练模块都 import 成功，才能显示 `--help` 或做基础检查。
    """

    return f"### Question:\n{question.strip()}\n\n### Answer:\n"


def load_sample_from_args(args: argparse.Namespace) -> tuple[str, str, str]:
    """Resolve one evaluation sample either from the dataset or from manual CLI arguments."""

    from sft.rollout_eval import extract_gsm8k_final_answer

    if args.question is not None:
        if args.gold_answer is None:
            raise ValueError("`--gold-answer` must be provided when `--question` is used.")
        question = args.question.strip()
        gold_answer_raw = args.gold_answer.strip()
        gold_answer_final = extract_gsm8k_final_answer(gold_answer_raw)
        return question, gold_answer_raw, gold_answer_final

    from datasets import load_dataset

    dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.split)
    sample = dataset[int(args.sample_index)]
    question = sample["question"].strip()
    gold_answer_raw = sample["answer"].strip()
    gold_answer_final = extract_gsm8k_final_answer(gold_answer_raw)
    return question, gold_answer_raw, gold_answer_final


def generate_model_output(args: argparse.Namespace, question: str) -> str:
    """Generate one completion for the given question.

    这里用 Hugging Face `generate` 跑最简单的 greedy decoding。
    作为独立评测脚本，这是最常见、也最容易验证的 baseline 写法。
    """

    if args.model_output is not None:
        return args.model_output

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompt = build_gsm8k_prompt(question)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=not args.allow_remote_model_files,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        local_files_only=not args.allow_remote_model_files,
    ).to(device)
    model.eval()

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion_ids = generated[0, prompt_length:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return completion


def main() -> None:
    args = build_arg_parser().parse_args()
    from sft.rollout_eval import load_grading_functions, truncate_to_first_gsm8k_answer

    extract_final_ans, grade_answer = load_grading_functions()

    question, gold_answer_raw, gold_answer_final = load_sample_from_args(args)
    model_output = generate_model_output(args, question)
    model_output = truncate_to_first_gsm8k_answer(model_output)

    predicted_answer = extract_final_ans(model_output)
    is_correct = False
    if predicted_answer is not None:
        is_correct = grade_answer(predicted_answer, gold_answer_final)

    print("=== Sample ===")
    print(question)
    print()
    print("=== Gold Answer (raw) ===")
    print(gold_answer_raw)
    print()
    print("=== Gold Answer (final) ===")
    print(gold_answer_final)
    print()
    print("=== Model Output ===")
    print(model_output)
    print()
    print("=== Extracted Prediction ===")
    print(predicted_answer)
    print()
    print("=== Grading Result ===")
    print(f"is_correct={is_correct}")


if __name__ == "__main__":
    main()
