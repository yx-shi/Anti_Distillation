"""Utilities for extracting the model's final answer from a long solution string.

这类函数通常放在“判分器”的最前面。原因很简单：
1. 模型输出往往是一整段推理过程，而不是单独的最终答案。
2. 判分器真正关心的，通常只是最后那个答案片段。

这一层只负责“从长文本里找到答案”，不负责判断答案是否正确。
真正的正确性判断在 `grader.py` 或 `deepscaler/rewards/math_reward.py` 里做。
"""

def extract_final_boxed(text):
    """提取最后一个 `\\boxed{...}` 里的内容。

    这是数学数据里最常见的“最终答案”格式之一。
    这里不用简单正则，而是手动维护栈，目的是正确处理嵌套花括号。
    例如 `\\boxed{\\frac{1}{2}}` 这种情况，直接用贪心/非贪心正则都容易写错。
    """
    stack = []
    start_pos = -1

    # 从最后一个 `\boxed{` 附近开始扫，避免把前面的中间结果误当成最终答案。
    last_formula = None
    last_pos = -1
    box_substr = '\\boxed{'
    box_len = len(box_substr)
    i = len(text.rsplit(box_substr, maxsplit=1)[0])
    while i < len(text):
        if text[i:i+box_len] == box_substr:
            # 进入一个 boxed 区域。
            start_pos = i + box_len
            stack.append(i)
            i += box_len
        elif text[i] == '{':
            # 普通左括号也入栈，这样 boxed 内部的 LaTeX 结构也能正确配对。
            stack.append(i)
            i += 1
        elif text[i] == '}' and stack:
            # 遇到右括号就出栈；当栈清空时，说明当前 boxed 完整闭合了。
            start = stack.pop()
            if not stack:
                end_pos = i
                last_formula = text[start_pos:end_pos]
                last_pos = start_pos
                break
        i += 1
    
    return last_formula, last_pos


# 当前预实验 prompt 统一要求最终答案写入 `\boxed{}`。
# 因此判分侧也只抽取 boxed 答案，避免自然语言/普通公式兜底把中间推导
# 或非标准最终答案误当作可判分答案。
extract_fns = [extract_final_boxed]


def extract_final_ans(text):
    """返回最后一个 `\\boxed{...}` 中的答案文本。"""
    max_result = None
    max_pos = -1
    for fn in extract_fns:
        try:
            result, pos = fn(text)
            assert result
            if pos > max_pos:
                max_result = result
                max_pos = pos
        except AssertionError:
            pass
        except Exception as e:
            print('function {} error'.format(fn.__name__))
            import traceback
            traceback.print_exc()
            continue
    return max_result
            

if __name__ == '__main__':
    # Example usage
    text = r"### ✅ Final Answer:\n\n$$\n\\boxed{4}\n$$\n\nThere are **4** positive integers $ b $ such that $ \\log_b 729 $ is a positive integer."
    result = extract_final_ans(text)
    print(f"The last boxed formula is: {result}")
