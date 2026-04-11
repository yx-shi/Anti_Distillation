from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Project-level configuration for SFT training."""

    model_name_or_path: str = "/data1/public_checkpoints/Qwen3-1.7B"
    dataset_format: str = "gsm8k_raw"
    dataset_backend: str = "huggingface"
    dataset_name: str = "openai/gsm8k"
    dataset_namespace: str = ""  # namespace指的是huggingface上数据集的组织方式，通常是"组织/数据集"，如果数据集没有组织则留空
    dataset_config_name: str = "main"
    modelscope_trust_remote_code: bool = True
    train_split: str = "train"
    eval_split: str = "test"
    train_file: str = ""
    eval_file: str = ""
    max_length: int = 256
    train_batch_size: int = 2
    eval_batch_size: int = 2
    num_epochs: int = 1
    learning_rate: float = 5e-5

    # weight decay: 训练过程中对模型参数进行L2正则化的强度。
    weight_decay: float = 0.01
    # warmup ratio: 训练开始时逐渐增加学习率的阶段占总训练步骤的比例。这个阶段有助于模型更稳定地收敛，避免一开始就使用过大的学习率导致训练不稳定。
    warmup_ratio: float = 0.03
    log_every: int = 20
    eval_every: int = 200
    num_workers: int = 0
    seed: int = 42
    max_steps: int = 0
    debug_fsdp: bool = False
    # eval preview: 每次验证时额外生成一道固定题，方便肉眼观察模型当前输出风格。
    eval_preview: bool = True
    # rollout eval: 直接让模型做题并用 grading 判对错，比 val_loss 更贴近真实任务表现。
    rollout_eval: bool = True
    # 默认不在每次 eval 上跑完整个 test split，而是先抽一个固定大小的子集做近似评估。
    # 这是训练工程里很常见的折中：降低评测开销，提升实验迭代速度。
    rollout_eval_max_samples: int = 64
    # rollout 最多生成多少个新 token。这个值太小会截断答案，太大会拖慢评测。
    rollout_max_new_tokens: int = 256
    ignore_index: int = -100
    train_on_prompt: bool = False
    local_files_only: bool = True
    output_dir: str = "result/sft_output"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build a small CLI so common hyperparameters can be overridden from the shell."""

    parser = argparse.ArgumentParser(description="Train a Qwen SFT model on GSM8K with PyTorch + FSDP.")
    parser.add_argument("--model-name-or-path", default="/data1/public_checkpoints/Qwen3-1.7B")
    parser.add_argument(
        "--dataset-format",
        choices=["gsm8k_raw", "distill_jsonl"],
        default="gsm8k_raw",
        help=(
            "训练数据格式。`gsm8k_raw` 表示直接读取原始 GSM8K；"
            "`distill_jsonl` 表示读取本地 prompt/completion JSONL。"
        ),
    )
    parser.add_argument("--dataset-backend", choices=["modelscope", "huggingface"], default="huggingface")
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-namespace", default="")
    parser.add_argument("--dataset-config-name", default="main")
    parser.add_argument("--disable-modelscope-trust-remote-code", action="store_true")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument(
        "--train-file",
        default="",
        help="当 `--dataset-format distill_jsonl` 时，训练集 JSONL 路径。",
    )
    parser.add_argument(
        "--eval-file",
        default="",
        help=(
            "可选的本地验证集 JSONL 路径。若留空，则继续使用 `--eval-split` "
            "指定的原始 GSM8K 验证集。"
        ),
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--debug-fsdp", action="store_true")
    parser.add_argument(
        "--disable-eval-preview",
        action="store_true",
        help="Disable the fixed GSM8K preview generation that otherwise runs during each evaluation.",
    )
    parser.add_argument(
        "--disable-rollout-eval",
        action="store_true",
        help="Disable rollout-based grading evaluation during validation.",
    )
    parser.add_argument(
        "--rollout-eval-max-samples",
        type=int,
        default=16,
        help="How many eval samples to score with rollout grading per evaluation. Use 0 to score the full eval split.",
    )
    parser.add_argument(
        "--rollout-max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens generated for each rollout evaluation sample.",
    )
    parser.add_argument("--train-on-prompt", action="store_true")
    parser.add_argument(
        "--allow-remote-model-files",
        action="store_true",
        help="If set, model/tokenizer files are allowed to be fetched remotely instead of local-only loading.",
    )
    parser.add_argument(
        "--output-dir",
        default="result/sft_output",
        help="训练输出目录。训练结束后会在其中保存最终 checkpoint 与配置摘要。",
    )
    return parser


def parse_args() -> TrainConfig:
    """Parse CLI arguments into a dataclass so the rest of the code only depends on one config object."""

    args = build_arg_parser().parse_args()
    return TrainConfig(
        model_name_or_path=args.model_name_or_path,
        dataset_format=args.dataset_format,
        dataset_backend=args.dataset_backend,
        dataset_name=args.dataset_name,
        dataset_namespace=args.dataset_namespace,
        dataset_config_name=args.dataset_config_name,
        modelscope_trust_remote_code=not args.disable_modelscope_trust_remote_code,
        train_split=args.train_split,
        eval_split=args.eval_split,
        train_file=args.train_file,
        eval_file=args.eval_file,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        log_every=args.log_every,
        eval_every=args.eval_every,
        num_workers=args.num_workers,
        seed=args.seed,
        max_steps=args.max_steps,
        debug_fsdp=args.debug_fsdp,
        eval_preview=not args.disable_eval_preview,
        rollout_eval=not args.disable_rollout_eval,
        rollout_eval_max_samples=args.rollout_eval_max_samples,
        rollout_max_new_tokens=args.rollout_max_new_tokens,
        train_on_prompt=args.train_on_prompt,
        local_files_only=not args.allow_remote_model_files,
        output_dir=args.output_dir,
    )
