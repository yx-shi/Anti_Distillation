"""Utilities for extracting the model's final answer from a long solution string.

这类函数通常放在“判分器”的最前面。原因很简单：
1. 模型输出往往是一整段推理过程，而不是单独的最终答案。
2. 判分器真正关心的，通常只是最后那个答案片段。

这一层只负责“从长文本里找到答案”，不负责判断答案是否正确。
真正的正确性判断在 `grader.py` 或 `deepscaler/rewards/math_reward.py` 里做。
"""

import re

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


# def extract_final_formula(text):
#     # Define regex patterns for $...$ and $$...$$
#     formula_pattern = r'\$+(.*?)\$+|\\\[(.*?)\\\]'
#     last_formula = re.findall('({})'.format(formula_pattern), text)[-1]
#     if not last_formula:
#         return None
#     print('last formula: {}'.format(last_formula))
#     last_formula = last_formula[0]
#     print('last formula: {}'.format(last_formula))
#     last_formula = re.search(formula_pattern, last_formula).group(1)
#     print('last formula: {}'.format(last_formula))
#     return last_formula

def extract_final_formula(text):
    """提取文本里最后一个 LaTeX 公式。

    支持三种常见写法：
    - `$...$`
    - `$$...$$`
    - `\\[ ... \\]`

    这属于“兜底策略”：当模型没用 `\\boxed{}`，但最后一句里写了公式时，仍有机会抽出答案。
    """
    # Define regex pattern to capture $...$, $$...$$, and \[...\]
    formula_pattern = r'\$+(.*?)\$+|\\\[(.*?)\\\]'
    
    # Use finditer to get match objects
    matches = re.finditer(formula_pattern, text)
    
    # Initialize variables to track the last match and its position
    last_match_content = None
    last_match_position = None
    
    # Iterate through the matches and store the last one
    for match in matches:
        for i in range(1, 3):
            if match.group(i):
                last_match_content = match.group(i)
                last_match_position = (match.start(i), match.end(i))
    
    if last_match_content and last_match_position:
        # print(f"Last formula: {last_match_content}")
        # print(f"Position: {last_match_position}")
        return last_match_content, last_match_position[0]
    return None, None


def extract_NL(text):
    """从自然语言模板里抽答案。

    这里故意匹配 `'he final answer is:'` 而不是完整的 `'The final answer is:'`，
    是个很实用的小技巧：这样既能匹配 `The final answer is:`，
    也能匹配 `the final answer is:`，少写一个大小写分支。
    """
    final_answer_prompts = ['he final answer is:', 'he answer is:']
    for prompt in final_answer_prompts:
        if prompt in text:
            return text.split(prompt)[-1].strip(), text.index(prompt) + len(prompt)
    return None, None


def extract_gsm8k_hash_answer(text):
    """提取 GSM8K 常见的 `#### final_answer` 格式。

    这里故意取“第一个” `####` 行，而不是最后一个。
    原因是生成式模型在没有及时停止时，可能会在正确答案后继续重复输出同一格式。
    对 GSM8K 来说，第一次出现的 `#### ...` 往往才是模型真正的最终答案位置。
    """

    match = re.search(r"####\s*([^\n\r]+)", text)
    if match is None:
        return None, None
    return match.group(1).strip(), match.start(1)


# 抽取函数按“候选来源”组织在一起，方便后续继续扩展。
# 常见范式：先写多个 extractor，再统一做“谁在文本里出现得更靠后，就选谁”。
extract_fns = [extract_NL, extract_gsm8k_hash_answer, extract_final_boxed, extract_final_formula]


def extract_final_ans(text):
    """综合多种规则，返回最像“最终答案”的那一段文本。

    核心策略不是“谁优先级最高”，而是“谁在原文里出现得最靠后”。
    这是因为数学解题输出里，后出现的答案通常更接近真正的最终作答。
    """
    max_result = None
    max_pos = -1
    max_fn = None
    for fn in extract_fns:
        try:
            result, pos = fn(text)
            assert result
            if pos > max_pos:
                max_result = result
                max_pos = pos
                max_fn = fn
        except AssertionError:
            pass
        except Exception as e:
            print('function {} error'.format(fn.__name__))
            import traceback
            traceback.print_exc()
            continue
    # 这里最终只返回答案文本本身；如果你后面想调试，也可以把 `max_fn.__name__` 一起返回。
    return max_result
            

if __name__ == '__main__':
    # Example usage
    text = r"First, we can simplify the left side of the equation by factoring out $16^{16}$:\n\\[16^{16}+16^{16}+16^{16}+16^{16} = 4 \\cdot 16^{16}.\\]\nNext, we can express $16$ as $2^4$:\n\\[4 \\cdot 16^{16} = 4 \\cdot (2^4)^{16} = 4 \\cdot 2^{64}.\\]\nAnd finally, we can express $4$ as $2^2$:\n\\[4 \\cdot 2^{64} = 2^2 \\cdot 2^{64} = 2^{66}.\\]\nSo, the equation becomes $2^{66}=2^x$.\nTherefore, the value of $x$ is $66$.\nThe answer is: $66$"
    # result = extract_boxed(text)
    # result = extract_final_formula(text)
    result = extract_final_ans(text)
    print(f"The last boxed formula is: {result}")
