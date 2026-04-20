from __future__ import annotations

from typing import Any


GSM8K_QWEN3_FORMAT_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def build_qwen3_messages(question: str) -> list[dict[str, str]]:
    """把 GSM8K 问题包装成 Qwen3 单轮对话消息。

    这里固定采用最简单、也是最容易和预实验对齐的形式：
    - 只构造一条 user message
    - 不额外塞 system prompt

    这样做的目的是把变量尽量压到最少，避免 teacher / student / eval
    分别用不同的消息结构，最后很难解释实验差异到底来自哪里。

    这里额外把 Qwen3 模型卡里推荐的数学输出规范直接拼到 user message 中，
    即 `Please reason step by step, and put your final answer within \boxed{}.`。

    这样做有两个现实收益：
    1. 让 teacher 更稳定地产出带 `\boxed{}` 的最终答案，减少 grader 的抽取失败。
    2. 让训练、rollout eval、teacher data generation 用完全同一套 prompt 约定，
       避免“teacher 用一种格式、student/eval 用另一种格式”的隐性分布偏移。
    """

    normalized_question = question.strip()
    user_content = (
        f"{normalized_question}\n\n"
        f"{GSM8K_QWEN3_FORMAT_INSTRUCTION}"
    )
    return [{"role": "user", "content": user_content}]


def render_qwen3_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool = False,
) -> str:
    """用 tokenizer 的 chat template 渲染真正送入模型的 prompt 文本。

    一个容易忽略但非常重要的细节是：
    这里故意“原样相信 tokenizer 的实际输出”，不手动改写模板。

    你的本地 Qwen3 tokenizer 在 `enable_thinking=False` 下，仍然会渲染一个空的
    `<think>...</think>` 块。虽然这和一些文档示例里的“完全不出现 think 块”不同，
    但它代表的是“当前本地权重 + tokenizer 组合的真实输入协议”。

    为了保证训练、teacher 生成、rollout eval 完全一致，这里不手动剥离该块。
    """

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError as exc:
        raise RuntimeError(
            "Current tokenizer does not support `enable_thinking` in `apply_chat_template`. "
            "The smoke pre-experiment is designed around Qwen3 chat templates, so please "
            "check the installed `transformers` version and tokenizer files."
        ) from exc


def build_qwen3_prompt(
    tokenizer: Any,
    question: str,
    *,
    enable_thinking: bool = False,
) -> str:
    """给单个问题生成 Qwen3 chat-template prompt。"""

    messages = build_qwen3_messages(question)
    return render_qwen3_prompt(
        tokenizer=tokenizer,
        messages=messages,
        enable_thinking=enable_thinking,
    )


def normalize_completion_text(completion: str) -> str:
    """统一 completion 的最小清洗规则。

    这里故意只做 `strip()`，不做更激进的文本改写。
    原因是 SFT/蒸馏训练通常希望尽量保留 teacher 原始输出分布；
    如果在进入训练前做太多“美化”，后面就很难判断模型到底学到了什么。
    """

    return completion.strip()


def build_prompt_completion_text(
    tokenizer: Any,
    question: str,
    completion: str,
    *,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    """把 question/completion 变成统一的训练样本视图。

    返回值里同时保留：
    - `messages`：结构化 chat 输入，便于数据审计
    - `prompt`：真正喂给模型的文本 prompt
    - `completion`：只包含答案部分
    - `full_text`：训练时拼接后的完整序列
    """

    messages = build_qwen3_messages(question)
    prompt = render_qwen3_prompt(
        tokenizer=tokenizer,
        messages=messages,
        enable_thinking=enable_thinking,
    )
    normalized_completion = normalize_completion_text(completion)
    return {
        "messages": messages,
        "prompt": prompt,
        "completion": normalized_completion,
        "full_text": prompt + normalized_completion,
    }
