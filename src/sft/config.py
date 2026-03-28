from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Project-level configuration for SFT training."""

    model_name_or_path: str = "/data1/public_checkpoints/Qwen3-1.7B"
    dataset_backend: str = "huggingface"
    dataset_name: str = "openai/gsm8k"
    dataset_namespace: str = ""
    dataset_config_name: str = "main"
    modelscope_trust_remote_code: bool = True
    train_split: str = "train"
    eval_split: str = "test"
    max_length: int = 256
    train_batch_size: int = 2
    eval_batch_size: int = 2
    num_epochs: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    log_every: int = 20
    eval_every: int = 200
    num_workers: int = 0
    seed: int = 42
    max_steps: int = 0
    debug_fsdp: bool = False
    ignore_index: int = -100
    train_on_prompt: bool = False
    local_files_only: bool = True


def build_arg_parser() -> argparse.ArgumentParser:
    """Build a small CLI so common hyperparameters can be overridden from the shell."""

    parser = argparse.ArgumentParser(description="Train a Qwen SFT model on GSM8K with PyTorch + FSDP.")
    parser.add_argument("--model-name-or-path", default="/data1/public_checkpoints/Qwen3-1.7B")
    parser.add_argument("--dataset-backend", choices=["modelscope", "huggingface"], default="huggingface")
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-namespace", default="")
    parser.add_argument("--dataset-config-name", default="main")
    parser.add_argument("--disable-modelscope-trust-remote-code", action="store_true")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
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
    parser.add_argument("--train-on-prompt", action="store_true")
    parser.add_argument(
        "--allow-remote-model-files",
        action="store_true",
        help="If set, model/tokenizer files are allowed to be fetched remotely instead of local-only loading.",
    )
    return parser


def parse_args() -> TrainConfig:
    """Parse CLI arguments into a dataclass so the rest of the code only depends on one config object."""

    args = build_arg_parser().parse_args()
    return TrainConfig(
        model_name_or_path=args.model_name_or_path,
        dataset_backend=args.dataset_backend,
        dataset_name=args.dataset_name,
        dataset_namespace=args.dataset_namespace,
        dataset_config_name=args.dataset_config_name,
        modelscope_trust_remote_code=not args.disable_modelscope_trust_remote_code,
        train_split=args.train_split,
        eval_split=args.eval_split,
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
        train_on_prompt=args.train_on_prompt,
        local_files_only=not args.allow_remote_model_files,
    )
