from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from grading.extract_ans import extract_final_ans
from grading.grader import grade_answer
from pre_exp.common import read_jsonl, write_jsonl
from sft.data import SupervisedFineTuningCollator
from sft.prompting import normalize_completion_text


DEFAULT_STUDENT_MODEL = "/data1/public_checkpoints/Qwen3-1.7B"
DEFAULT_INPUT_FILE = "result/pre_exp/candidates/smoke/candidate_pool.jsonl"
DEFAULT_OUTPUT_FILE = "result/pre_exp/candidates/smoke/scored_candidates.jsonl"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score teacher candidates with the student base model.")
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--model-name-or-path", default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="auto",
        help=(
            "Transformers attention backend. `auto` keeps the library default; "
            "`flash_attention_2` uses the flash_attn package when supported by the model."
        ),
    )
    parser.add_argument("--allow-remote-model-files", action="store_true")
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_scoring_feature(record: dict[str, Any], tokenizer, max_length: int) -> tuple[dict[str, list[int]], bool]:
    """把单条候选记录转成和训练时一致的 token 视图。

    返回值中的 bool 表示这条样本在打分阶段是否发生了长度截断。
    这里不因为过长就直接丢样本，而是保留一个显式标记，方便后续分析。
    """

    prompt_ids = tokenizer(
        record["prompt_text"],
        add_special_tokens=False,
    )["input_ids"]
    completion_ids = tokenizer(
        record["candidate_text"],
        add_special_tokens=False,
    )["input_ids"]
    full_ids = prompt_ids + completion_ids
    score_truncated = len(full_ids) > max_length
    return {
        "prompt_ids": prompt_ids,
        "input_ids": full_ids,
    }, score_truncated


def compute_batch_mean_completion_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
) -> tuple[list[float | None], list[int]]:
    """按样本返回 completion token 的平均 NLL 和 token 数。

    这里和训练代码保持同一套 masking 逻辑：
    - prompt 部分 label 已经在 collator 中被改成 `ignore_index`
    - 因此这里直接对非忽略位置求平均，就得到“completion-token mean NLL”
    """

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(shift_labels.shape)

    valid_mask = shift_labels.ne(ignore_index)
    token_counts = valid_mask.sum(dim=1)
    loss_sums = (token_losses * valid_mask).sum(dim=1)

    mean_losses: list[float | None] = []
    token_count_list: list[int] = []
    for loss_sum, token_count in zip(loss_sums.tolist(), token_counts.tolist()):
        token_count_int = int(token_count)
        token_count_list.append(token_count_int)
        if token_count_int <= 0:
            mean_losses.append(None)
        else:
            mean_losses.append(float(loss_sum) / token_count_int)
    return mean_losses, token_count_list


def main() -> None:
    args = build_arg_parser().parse_args()

    device = resolve_device(args.device)
    print(
        "student_score_start "
        f"input_file={args.input_file} "
        f"model={args.model_name_or_path} "
        f"device={device} "
        f"batch_size={args.batch_size}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=not args.allow_remote_model_files,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "torch_dtype": "auto",
        "local_files_only": not args.allow_remote_model_files,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs).to(device)
    model.config.use_cache = False
    model.eval()
    print(
        "student_score_model_loaded "
        f"requested_attn_implementation={args.attn_implementation} "
        f"actual_attn_implementation={getattr(model.config, '_attn_implementation', 'unknown')} "
        f"use_cache={model.config.use_cache}",
        flush=True,
    )

    collator = SupervisedFineTuningCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        ignore_index=-100,
        train_on_prompt=False,
    )

    raw_records = read_jsonl(args.input_file)
    scored_records: list[dict[str, Any]] = []
    scoreable_entries: list[tuple[int, dict[str, Any]]] = []

    for record_idx, record in enumerate(raw_records):
        candidate_text_raw = normalize_completion_text(str(record.get("candidate_text", "")))
        extracted_answer = extract_final_ans(candidate_text_raw) if candidate_text_raw else None

        finish_reason = str(record.get("finish_reason") or "")
        stop_reason = str(record.get("stop_reason") or "")
        is_empty = candidate_text_raw == ""
        is_generation_truncated = finish_reason == "length" or stop_reason == "length"
        is_extractable = extracted_answer is not None
        is_correct = is_extractable and grade_answer(extracted_answer, str(record["gold_answer"]))

        # “valid candidate” 仍然只作为候选质量分析字段：
        # - 不是空输出
        # - 不是因为长度被截断
        # - 能抽出最终答案
        #
        # NLL 打分不再只服务“正确/有效候选重排”，而是服务完整 Teacher
        # 分布蒸馏。因此只要 Teacher completion 非空，就计算 Student NLL；
        # 正误、截断、可抽取性都保留下来做分析，不参与是否打分。
        is_valid_candidate = (not is_empty) and (not is_generation_truncated) and is_extractable

        enriched_record = dict(record)
        enriched_record.update(
            {
                "candidate_text": candidate_text_raw,
                "extracted_answer": extracted_answer,
                "is_empty": is_empty,
                "is_generation_truncated": is_generation_truncated,
                "is_extractable": is_extractable,
                "is_valid_candidate": is_valid_candidate,
                "is_correct": bool(is_correct),
                "student_mean_nll": None,
                "student_token_count": 0,
                "score_truncated": False,
            }
        )
        scored_records.append(enriched_record)

        if not is_empty:
            feature, score_truncated = build_scoring_feature(enriched_record, tokenizer, args.max_length)
            enriched_record["score_truncated"] = score_truncated
            scoreable_entries.append((record_idx, feature))

    with torch.inference_mode():
        for batch_start in range(0, len(scoreable_entries), args.batch_size):
            batch_entries = scoreable_entries[batch_start: batch_start + args.batch_size]
            batch_indices = [record_idx for record_idx, _ in batch_entries]
            batch_features = [feature for _, feature in batch_entries]

            batch = collator(batch_features)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            mean_losses, token_counts = compute_batch_mean_completion_nll(
                logits=outputs.logits,
                labels=batch["labels"],
                ignore_index=-100,
            )

            for record_idx, mean_loss, token_count in zip(batch_indices, mean_losses, token_counts):
                scored_records[record_idx]["student_mean_nll"] = mean_loss
                scored_records[record_idx]["student_token_count"] = token_count

    write_jsonl(args.output_file, scored_records)
    print(
        "student_score_done "
        f"input_records={len(raw_records)} "
        f"scoreable_records={len(scoreable_entries)} "
        f"output_file={args.output_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
