"""Gold-answer normalization helpers shared by data build and eval.

DeepScaleR 的 `answer` 字段已经是短标准答案，因此主链路这里只做 strip。
为了不让历史 GSM8K JSONL 或手动调试样本立即失效，仍在统一边界兼容
`reasoning #### final_answer` 形式；不要在主实验代码里继续暴露 GSM8K 专名。
"""

from __future__ import annotations


def normalize_gold_answer(answer_text: object) -> str:
    """Return the short gold answer used by grading.

    Current DeepScaleR records pass through unchanged after whitespace trimming.
    Legacy records that contain a GSM8K-style `####` marker are normalized once at
    ingestion time, so downstream scoring can treat `gold_answer` as a plain answer.
    """

    text = str(answer_text).strip()
    if "####" in text:
        return text.rsplit("####", maxsplit=1)[-1].strip()
    return text
