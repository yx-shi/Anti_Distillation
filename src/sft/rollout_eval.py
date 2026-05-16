from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable

from sft.prompting import build_qwen3_prompt

# `grading/` 目前是项目根目录下的独立包，而训练入口通常是 `python src/train_sft.py`。
# 这时 Python 会把 `src/` 放进模块搜索路径，但不会自动把项目根目录也加进去。
# 因此这里显式补上 root path，这是一种很常见的“脚本式项目”兼容写法。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grading.gold_answer import normalize_gold_answer
from pre_exp.common import choose_subset_indices, choose_subset_indices_from_pool


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


def build_rollout_eval_samples(
    tokenizer,
    eval_source_dataset: Any,
    max_samples: int,
    subset_seed: int,
    *,
    question_field: str = "question",
    answer_field: str = "answer",
    candidate_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    """把评测集样本转换成统一的 rollout 请求视图。

    这一步把“评测哪些题”“每道题的 gold answer 是什么”“真正送给模型的 prompt 长什么样”
    统一固化下来。这样无论你后面用 HF 还是 vLLM 生成，评测口径都能保持一致。
    """

    dataset_size = len(eval_source_dataset)
    if candidate_indices is None:
        selected_indices = choose_subset_indices(dataset_size, max_samples, subset_seed)
    else:
        selected_indices = choose_subset_indices_from_pool(candidate_indices, max_samples, subset_seed)

    prepared_samples: list[dict[str, Any]] = []
    for dataset_idx in selected_indices:
        sample = eval_source_dataset[int(dataset_idx)]
        if question_field not in sample:
            available_fields = ", ".join(sorted(sample.keys()))
            raise KeyError(
                f"Question field `{question_field}` is not present in eval sample. "
                f"Available fields: {available_fields}"
            )
        if answer_field not in sample:
            available_fields = ", ".join(sorted(sample.keys()))
            raise KeyError(
                f"Answer field `{answer_field}` is not present in eval sample. "
                f"Available fields: {available_fields}"
            )

        question = str(sample[question_field]).strip()
        raw_answer = sample[answer_field]
        prepared_samples.append(
            {
                "sample_id": int(dataset_idx),
                "question": question,
                "gold_answer": normalize_gold_answer(raw_answer),
                "prompt_text": build_rollout_prompt(tokenizer, question),
            }
        )
    return prepared_samples


def _population_variance(values: list[float], mean_value: float) -> float:
    if not values:
        return 0.0
    return sum((value - mean_value) ** 2 for value in values) / len(values)


def grade_multi_rollout_predictions(
    samples: list[dict[str, Any]],
    generated_texts_by_sample: list[list[str]],
    *,
    allow_hash_answer_fallback: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """评估每题多次 rollout，并按独立样本方差传播得到整体 acc 方差。

    对第 i 道题，先用 R 次 rollout 的 0/1 correctness 估计该题随机变量的
    mean/variance；整体 accuracy 是 N 道题随机变量的平均，因此整体方差为
    sum(var_i) / N^2。
    """

    if len(samples) != len(generated_texts_by_sample):
        raise ValueError(
            "The number of prepared samples does not match the number of generated outputs."
        )

    extract_final_ans, grade_answer = load_grading_functions()
    eval_records: list[dict[str, Any]] = []
    sample_means: list[float] = []
    sample_variances: list[float] = []
    expected_correct = 0.0
    num_rollouts = 0

    for sample, generated_texts in zip(samples, generated_texts_by_sample):
        if not generated_texts:
            raise ValueError("Each sample must have at least one rollout output.")
        if num_rollouts == 0:
            num_rollouts = len(generated_texts)
        elif len(generated_texts) != num_rollouts:
            raise ValueError("All samples must have the same number of rollout outputs.")

        rollout_records: list[dict[str, Any]] = []
        correctness_values: list[float] = []
        for rollout_id, generated_text in enumerate(generated_texts):
            completion = generated_text.strip()
            if allow_hash_answer_fallback:
                predicted_answer = extract_final_ans(
                    completion,
                    allow_hash_fallback=True,
                )
            else:
                predicted_answer = extract_final_ans(completion)
            is_correct = predicted_answer is not None and grade_answer(
                predicted_answer,
                sample["gold_answer"],
            )
            correctness_values.append(float(is_correct))
            rollout_records.append(
                {
                    "rollout_id": rollout_id,
                    "generated_text": generated_text,
                    "predicted_answer": predicted_answer,
                    "is_correct": bool(is_correct),
                }
            )

        sample_mean = sum(correctness_values) / len(correctness_values)
        sample_variance = _population_variance(correctness_values, sample_mean)
        sample_means.append(sample_mean)
        sample_variances.append(sample_variance)
        expected_correct += sample_mean

        first_rollout = rollout_records[0]
        eval_records.append(
            {
                "sample_id": sample["sample_id"],
                "question": sample["question"],
                "gold_answer": sample["gold_answer"],
                "generated_text": first_rollout["generated_text"],
                "predicted_answer": first_rollout["predicted_answer"],
                "is_correct": first_rollout["is_correct"],
                "sample_rollout_acc_mean": sample_mean,
                "sample_rollout_acc_variance": sample_variance,
                "rollouts": rollout_records,
            }
        )

    total = len(samples)
    rollout_acc = sum(sample_means) / total if total > 0 else 0.0
    rollout_variance = sum(sample_variances) / (total * total) if total > 0 else 0.0
    metrics = {
        "rollout_acc": rollout_acc,
        "rollout_acc_variance": rollout_variance,
        "rollout_acc_std": math.sqrt(rollout_variance),
        "mean_sample_rollout_acc_variance": (
            sum(sample_variances) / total if total > 0 else 0.0
        ),
        "rollout_correct": float(expected_correct),
        "rollout_total": float(total),
        "num_rollouts": float(num_rollouts),
    }
    return metrics, eval_records
