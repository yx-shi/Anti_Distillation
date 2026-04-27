# Data And Evaluation Spec

本文档汇总当前数据集、评分器和实验产物口径。具体预实验流程见 `pre_exp.md`。

## Datasets

- GSM8K：`openai/gsm8k`，历史预实验使用的数据集，当前认为难度偏低。
- DeepScaleR：`agentica-org/DeepScaleR-Preview-Dataset`，当前后续预实验优先数据集。

DeepScaleR 字段：

- `problem`：题目文本。
- `answer`：短标准答案。
- `solution`：参考解答。

当前 `teacher_generate.py` 支持通过参数指定字段：

- `--question-field problem`
- `--answer-field answer`

## Grading

- 本项目主要使用 `grading/` 中的答案抽取与数学等价判分。
- 候选答案先由 `grading.extract_ans.extract_final_ans` 抽取最终答案。
- 再由 `grading.grader.grade_answer` 与 gold answer 比较。
- 对数学数据，prompt 统一要求模型把最终答案放入 `\boxed{}`，以提升抽取稳定性。

## Candidate Quality Fields

预实验候选常用字段：

- `is_empty`：候选是否为空。
- `is_generation_truncated`：生成是否因长度上限截断。
- `is_extractable`：是否能抽出最终答案。
- `is_valid_candidate`：非空、未截断、可抽答案。
- `is_correct`：抽取答案与 gold 判分正确。
- `student_mean_nll`：Student 对 completion token 的平均 NLL。
- `student_token_count`：用于 NLL 的 completion token 数。

注意：后续正式实验中，selection policy 应优先使用“正确且未截断”的候选，避免截断候选污染训练数据。

## Main Outputs

- `candidate_pool.jsonl`：Teacher 生成候选池。
- `scored_candidates.jsonl`：候选质量字段和 Student NLL 打分。
- `teacher_baseline.selected.jsonl`：baseline 选择结果。
- `teacher_adversarial.selected.jsonl`：adversarial 选择结果。
- `distill_teacher_baseline.jsonl`：可用于 SFT 的 baseline 数据。
- `distill_teacher_adversarial.jsonl`：可用于 SFT 的 adversarial 数据。
- `dataset_summary.json`：候选质量、fallback、长度和 NLL gap 统计。
