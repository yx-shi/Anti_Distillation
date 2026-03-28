"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.

和 `grader.py` 相比，这个文件不是单纯返回 True/False，
而是把“判对结果”包装成训练/评测更常用的 reward 接口。
"""
from typing import List, Union

from src.grading.deepscaler.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END, OAI_RM_MODEL
from src.grading.deepscaler.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from src.grading.deepscaler.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd
from src.grading.deepscaler.system_prompts import ORM_PROMPT
from src.grading.deepscaler.utils import call_gemini_llm, call_oai_rm_llm

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        
        # DeepScaler 这套格式假设模型输出像：
        # <think> ...中间推理... </think> \boxed{final_answer}
        #
        # 也就是说，reward 函数不只是看“答案内容”，还检查输出是否符合约定格式。
        if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        
        model_answer = extract_answer(model_solution)
        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # ground truth 允许是单个答案，也允许是多个等价答案。
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)
        
        # 统一转成 list，是很常见的数据处理范式：
        # 后面循环逻辑会更简单，不需要单独维护“单答案/多答案”两套分支。
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # 参考答案如果自带 `\boxed{}`，就只取盒子内部。
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        
        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # 第一层：本地规则判分。
        # 先走便宜的 heuristics，只有都失败了才考虑更重的 LLM judge。
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                return RewardOutput(reward=self.config.correct_reward, is_correct=True)

        # 第二层：可选的 LLM-as-a-judge / ORM。
        # 只有配置打开 `use_math_orm` 时才会触发，因为这一步更贵、更慢，也更依赖外部 API。
        if self.config.use_math_orm:
            for ground_truth in processed_ground_truths:
                try:
                    orm_response = call_gemini_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                    )

                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                except Exception as e:
                    print ("Error calling Gemini ORM, trying OAI RM")
                    orm_response = call_oai_rm_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                        model_id=OAI_RM_MODEL,
                    )
                    
                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                    continue
                
        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

def deepscaler_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], solution_length: int=None, baseline_length: int=None, enable_llm = False):
    """DeepScaler 风格的外部便捷入口。

    返回一个字典：
    - `acc`: 是否答对
    - `score`: reward 分数

    这里的 length shaping 体现了 RL / reward engineering 里常见的一个思路：
    正确的前提下，解得更短，可以再拿到额外奖励。
    """
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    reward = reward_response.reward
    is_correct = reward_response.is_correct
    # print('reward: {}, iscorrect: {} baseline_length: {}'.format(reward, is_correct, baseline_length))
    # print('reward: {} baseline_length: {} solution_length: {} answer: {}, ground_truth: {}'.format(reward, baseline_length, solution_length, solution_str[-100:], ground_truth))
    if baseline_length is not None and is_correct:
        reward *= max(0, baseline_length - solution_length) / baseline_length
        reward = reward * 2 - 1
    return {
        'acc': is_correct,
        'score': reward
    }

def deepscaler_reward_fn_new(solution_str: str, ground_truth: Union[str, List[str]], solution_length: int=None, baseline_length: int=None, enable_llm = False):
    """和 `deepscaler_reward_fn` 类似，但长度奖励的映射方式更直接。"""
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    reward = reward_response.reward
    is_correct = reward_response.is_correct
    # print('reward: {}, iscorrect: {} baseline_length: {}'.format(reward, is_correct, baseline_length))
    # print('reward: {} baseline_length: {} solution_length: {} answer: {}, ground_truth: {}'.format(reward, baseline_length, solution_length, solution_str[-100:], ground_truth))
    if baseline_length is not None and is_correct:
        reward *= max(0, baseline_length - solution_length) / baseline_length
    return {
        'acc': is_correct,
        'score': reward
    }

def deepscaler_reward_fn_no_length(solution_str: str, ground_truth: Union[str, List[str]], solution_length: int=None, baseline_length: int=None, enable_llm = False):
    """最简单的版本：只看对错，不做长度 shaping。"""
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    reward = reward_response.reward
    is_correct = reward_response.is_correct
    # print('reward: {} answer: {}, ground_truth: {}'.format(reward, solution_str[-100:], ground_truth))
    return {
        'acc': is_correct,
        'score': reward
    }


if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig)
    import json
    with open('/home/lixujun/AITA-o1/experiments/over_thinking/dump_1751528078291.json', 'r', encoding='utf8') as file:
        data = json.load(file)
    input = RewardInput(problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.", problem_type=RewardType.MATH, model_response=data['solution_str'], ground_truth={"answer": data['ground_truth']})
    output = reward(input)
    print(output)
