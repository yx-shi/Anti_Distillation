from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

from pre_exp.common import write_json
from sft.distributed import DistributedContext
from sft.hf_cache import ensure_writable_hf_datasets_cache
from sft.rollout_eval import evaluate_rollout_accuracy


DEFAULT_MODEL_PATH = "result/pre_exp/runs/smoke/teacher_baseline/final_checkpoint"
DEFAULT_OUTPUT_FILE = "result/pre_exp/analysis/smoke/final_eval_teacher_baseline.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full-test rollout grading on a trained checkpoint.")
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config-name", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-remote-model-files", action="store_true")
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=not args.allow_remote_model_files,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        local_files_only=not args.allow_remote_model_files,
    ).to(device)
    model.eval()

    ensure_writable_hf_datasets_cache()
    dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.split)
    runtime = DistributedContext(
        use_fsdp=False,
        world_size=1,
        rank=0,
        local_rank=0,
        device=device,
    )

    with torch.inference_mode():
        metrics = evaluate_rollout_accuracy(
            model=model,
            tokenizer=tokenizer,
            eval_source_dataset=dataset,
            runtime=runtime,
            max_new_tokens=args.max_new_tokens,
            max_samples=0,
        )

    write_json(
        args.output_file,
        {
            "model_name_or_path": args.model_name_or_path,
            "dataset_name": args.dataset_name,
            "dataset_config_name": args.dataset_config_name,
            "split": args.split,
            "max_new_tokens": args.max_new_tokens,
            "metrics": metrics,
        },
    )
    print(
        "final_eval_done "
        f"model={args.model_name_or_path} "
        f"rollout_acc={metrics['rollout_acc']:.4f} "
        f"output_file={args.output_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
