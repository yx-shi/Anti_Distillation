from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from sft.distributed import DistributedContext
from sft.prompting import build_qwen3_prompt

# `grading/` 目前是项目根目录下的独立包，而训练入口通常是 `python src/train_sft.py`。
# 这时 Python 会把 `src/` 放进模块搜索路径，但不会自动把项目根目录也加进去。
# 因此这里显式补上 root path，这是一种很常见的“脚本式项目”兼容写法。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pre_exp.common import choose_subset_indices

EVAL_PREVIEW_QUESTION = (
    "James decides to run 3 sprints 3 times a week. "
    "He runs 60 meters each sprint. How many total meters does he run a week?"
)


def build_rollout_prompt(tokenizer, question: str) -> str:
    """Build the Qwen3 chat-template prompt used during rollout generation.

    rollout eval 不再维护独立的硬编码模板，而是直接复用共享的 Qwen3 helper。
    这样训练输入、teacher 输入和 rollout 输入使用的是同一套 prompt 协议。
    """

    return build_qwen3_prompt(
        tokenizer=tokenizer,
        question=question,
        enable_thinking=False,
    )


def load_grading_functions() -> tuple[Callable[[str], str | None], Callable[[str, str], bool]]:
    """Import grading helpers lazily so training can still start even if rollout eval is disabled.

    这是项目里很常见的一个范式：
    - 训练主流程尽量少依赖“可选评测模块”
    - 只有真正跑到 rollout grading 时，才去导入对应依赖

    这样做的好处是：
    1. 配置关闭该功能时，不会因为缺评测依赖而整个训练入口都 import 失败
    2. 报错位置更接近真实问题，更容易排查
    """

    try:
        from grading.extract_ans import extract_final_ans
        from grading.grader import grade_answer
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "unknown dependency"
        raise RuntimeError(
            "Failed to import the local grading pipeline for rollout evaluation. "
            "Make sure the grading dependencies are installed. "
            f"Missing module: `{missing_name}`."
        ) from exc

    return extract_final_ans, grade_answer


def extract_gsm8k_final_answer(answer_text: str) -> str:
    """Extract the final GSM8K answer from `reasoning + #### answer`.

    GSM8K 的原始 `answer` 字段通常不是纯答案，而是一段推理再加上最后一行：
    `#### 123`
    因此在做自动评分时，gold answer 一般需要先抽成最终答案，再与模型输出比较。
    """

    text = answer_text.strip()
    if "####" in text:
        return text.split("####")[-1].strip()
    return text


def build_rollout_eval_samples(
    tokenizer,
    eval_source_dataset: Any,
    max_samples: int,
    subset_seed: int,
) -> list[dict[str, Any]]:
    """把评测集样本转换成统一的 rollout 请求视图。

    这一步把“评测哪些题”“每道题的 gold answer 是什么”“真正送给模型的 prompt 长什么样”
    统一固化下来。这样无论你后面用 HF 还是 vLLM 生成，评测口径都能保持一致。
    """

    dataset_size = len(eval_source_dataset)
    selected_indices = choose_subset_indices(dataset_size, max_samples, subset_seed)

    prepared_samples: list[dict[str, Any]] = []
    for dataset_idx in selected_indices:
        sample = eval_source_dataset[int(dataset_idx)]
        question = sample["question"].strip()
        prepared_samples.append(
            {
                "sample_id": int(dataset_idx),
                "question": question,
                "gold_answer": extract_gsm8k_final_answer(sample["answer"]),
                "prompt_text": build_rollout_prompt(tokenizer, question),
            }
        )
    return prepared_samples


def grade_rollout_predictions(
    samples: list[dict[str, Any]],
    generated_texts: list[str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """用统一的 grading 规则评估一批 rollout 输出。"""

    if len(samples) != len(generated_texts):
        raise ValueError(
            "The number of prepared samples does not match the number of generated outputs."
        )

    extract_final_ans, grade_answer = load_grading_functions()
    eval_records: list[dict[str, Any]] = []
    correct = 0

    for sample, generated_text in zip(samples, generated_texts):
        completion = generated_text.strip()
        predicted_answer = extract_final_ans(completion)
        is_correct = predicted_answer is not None and grade_answer(predicted_answer, sample["gold_answer"])
        correct += int(is_correct)
        eval_records.append(
            {
                "sample_id": sample["sample_id"],
                "question": sample["question"],
                "gold_answer": sample["gold_answer"],
                "generated_text": generated_text,
                "predicted_answer": predicted_answer,
                "is_correct": bool(is_correct),
            }
        )

    total = len(samples)
    metrics = {
        "rollout_acc": correct / total if total > 0 else 0.0,
        "rollout_correct": float(correct),
        "rollout_total": float(total),
    }
    return metrics, eval_records


def greedy_generate_completion(
    model,
    tokenizer,
    runtime: DistributedContext,
    question: str,
    max_new_tokens: int,
) -> str:
    """Generate one answer with a small hand-written greedy decoding loop.

    这里不直接调用 `model.generate(...)`，而是保留一个显式的 token-by-token 循环。
    这样在 FSDP 场景里更容易控制每个 rank 的 forward 次数，并把生成逻辑和训练逻辑放在同一套
    低层 PyTorch workflow 里，排障时也更直观。

    一个重要细节是：这里故意不在遇到 EOS 后提前 `break`。
    原因是多卡 FSDP 下，不同 rank 如果在不同时间提早结束，容易让 collective 次序失配。
    固定迭代 `max_new_tokens` 次，是一种更稳妥的分布式范式。
    """

    prompt = build_rollout_prompt(tokenizer, question)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(runtime.device)
    attention_mask = encoded["attention_mask"].to(runtime.device)
    prompt_length = input_ids.size(1)

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

    generated_ids = input_ids[0, prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_eval_preview(
    model,
    tokenizer,
    runtime: DistributedContext,
    max_new_tokens: int,
) -> dict[str, str]:
    """Generate the fixed preview sample that is printed during evaluation."""

    completion = greedy_generate_completion(
        model=model,
        tokenizer=tokenizer,
        runtime=runtime,
        question=EVAL_PREVIEW_QUESTION,
        max_new_tokens=max_new_tokens,
    )
    return {
        "question": EVAL_PREVIEW_QUESTION,
        "completion": completion,
    }


def evaluate_rollout_accuracy(
    model,
    tokenizer,
    eval_source_dataset: Any,
    runtime: DistributedContext,
    max_new_tokens: int,
    max_samples: int,
) -> dict[str, float]:
    """Run rollout-based grading on the eval split.

    这里的工作流是：
    1. 取 eval 样本里的原始 question / answer
    2. 用当前模型实际生成答案
    3. 从生成结果里抽最终答案
    4. 从 GSM8K gold 里抽 `####` 后的最终答案
    5. 用 grading pipeline 判对错

    和 `val_loss` 相比，这个指标更接近“模型真正做题时是否答对”。
    """

    extract_final_ans, grade_answer = load_grading_functions()

    dataset_size = len(eval_source_dataset)
    total_target_samples = dataset_size if max_samples <= 0 else min(dataset_size, max_samples)

    if total_target_samples == 0:
        return {"rollout_acc": 0.0, "rollout_correct": 0.0, "rollout_total": 0.0}

    # 为了让 FSDP 下每个 rank 的 forward 次数一致，这里不是简单地“谁有多少样本就跑多少”，
    # 而是把总样本数按 slot 排开。每个 slot 里所有 rank 都会做一次生成；
    # 如果某个 rank 在该 slot 没有真实样本，就用一个 dummy question 占位参与 forward。
    slots_per_rank = math.ceil(total_target_samples / runtime.world_size)
    dummy_question = EVAL_PREVIEW_QUESTION

    local_correct = 0
    local_total = 0

    for slot_idx in range(slots_per_rank):
        global_sample_idx = slot_idx * runtime.world_size + runtime.rank
        has_real_sample = global_sample_idx < total_target_samples

        if has_real_sample:
            sample = eval_source_dataset[int(global_sample_idx)]
            question = sample["question"].strip()
            gold_answer = extract_gsm8k_final_answer(sample["answer"])
        else:
            question = dummy_question
            gold_answer = ""

        model_output = greedy_generate_completion(
            model=model,
            tokenizer=tokenizer,
            runtime=runtime,
            question=question,
            max_new_tokens=max_new_tokens,
        )

        if has_real_sample:
            predicted_answer = extract_final_ans(model_output)
            is_correct = predicted_answer is not None and grade_answer(predicted_answer, gold_answer)
            local_correct += int(is_correct)
            local_total += 1

    correct_tensor = torch.tensor(float(local_correct), device=runtime.device)
    total_tensor = torch.tensor(float(local_total), device=runtime.device)

    if runtime.use_fsdp:
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    rollout_correct = correct_tensor.item()
    rollout_total = total_tensor.item()
    rollout_acc = rollout_correct / rollout_total if rollout_total > 0 else 0.0

    return {
        "rollout_acc": rollout_acc,
        "rollout_correct": rollout_correct,
        "rollout_total": rollout_total,
    }
