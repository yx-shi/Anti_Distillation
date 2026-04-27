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

当前 response-level selection policy 不再使用这些质量字段做过滤。baseline 始终选择 Teacher 第一条候选；adversarial 始终选择可计算 NLL 的候选中 `student_mean_nll` 最大的一条。质量字段保留用于解释 Teacher 分布、截断率、正确率和训练后结果，而不是把训练数据裁成“正确且有效”的子分布。

## Token-Level Decoding Outputs

vLLM-dual hard/soft 接入 full smoke 时，优先复用现有 pre-exp 产物规范，但需要额外记录生成模式维度。

建议模式名：

- `teacher_plain`：普通 vLLM Teacher 生成，不传 `dual_model_config`。
- `teacher_token_hard`：`dual_model_config.adversarial_mode=hard`。
- `teacher_token_soft`：`dual_model_config.adversarial_mode=soft`。

token-level full smoke 的 raw generation/candidate 记录应尽量包含：

- `generation_mode`：上述模式名。
- `dual_model_config`：hard/soft 的关键参数摘要；plain 可为 `null`。
- `worker_cls` 或等价 marker：用于证明 plain 没走 dual worker、hard/soft 走了 dual worker。
- `adversarial_mode`：plain 为 `null`，hard/soft 为实际模式。
- `candidate_text`、`finish_reason`、`is_generation_truncated`：沿用现有候选质量分析字段。
- `intervention_summary`：如果当前 pipeline 能稳定获得 hard/soft intervention 统计则写入；否则在 run summary 中记录“仅日志可见”。

SFT-ready JSONL 仍应兼容 `src/train_sft.py --dataset-format distill_jsonl`，避免为 token-level smoke 另起训练数据格式。

## Main Outputs

- `candidate_pool.jsonl`：Teacher 生成候选池。
- `scored_candidates.jsonl`：候选质量字段和 Student NLL 打分。
- `teacher_baseline.selected.jsonl`：baseline 选择结果。
- `teacher_adversarial.selected.jsonl`：adversarial 选择结果。
- `distill_teacher_baseline.jsonl`：可用于 SFT 的 baseline 数据。
- `distill_teacher_adversarial.jsonl`：可用于 SFT 的 adversarial 数据。
- `dataset_summary.json`：候选质量、选中样本质量、长度和 NLL gap 统计。
