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
- gold answer 在数据入口统一通过 `grading.gold_answer.normalize_gold_answer`
  规范化。DeepScaleR `answer` 字段已经是短答案，因此这里只做 `strip()`；
  历史 `####` 形式只作为兼容逻辑保留，不再以 GSM8K 专名暴露到主链路。
- 候选答案先由 `grading.extract_ans.extract_final_ans` 抽取最终答案；当前实现优先接受最后一个 `\boxed{...}`，缺少 boxed 时回退到 `Final Answer:`。
- 再由 `grading.grader.grade_answer` 与 gold answer 比较。
- 对数学数据，prompt 统一要求模型把最终答案放入 `\boxed{}`。模型输出抽取不再使用 GSM8K `#### ...`、泛自然语言或普通公式兜底。

## Rollout Eval Metrics

`src/sft/rollout_eval.py` 只保留离线 rollout grading 共享函数：构造 Qwen3
rollout prompt、准备 eval samples、聚合多 rollout correctness。训练主循环不再
内嵌 greedy rollout / eval preview；checkpoint 与 final rollout eval 由
`src/pre_exp/final_eval.py` 离线消费 checkpoint。

当前离线 rollout eval 对每道题采样 4 次，默认使用 `temperature=0.7`、`top_p=0.8`。每题先把 4 个 correctness 结果计算成：

- `sample_rollout_acc_mean`
- `sample_rollout_acc_variance`，使用 population variance

整体指标按独立随机变量的均值/方差可加性聚合：

- `rollout_acc = mean(sample_rollout_acc_mean)`
- `rollout_acc_variance = sum(sample_rollout_acc_variance) / N^2`
- `rollout_acc_std = sqrt(rollout_acc_variance)`

曲线图中的点使用 `rollout_acc`，误差线使用 `rollout_acc ± rollout_acc_std`。

## Candidate Quality Fields

预实验候选常用字段：

- `is_empty`：候选是否为空。
- `is_generation_truncated`：生成是否因长度上限截断。
- `is_extractable`：是否能抽出最终答案。
- `is_valid_candidate`：非空、未截断、可抽答案。
- `is_correct`：抽取答案与 gold 判分正确。
- `student_mean_nll`：Student 对 completion token 的平均 NLL。
- `student_token_count`：用于 NLL 的 completion token 数。

当前 response-level selection policy 使用 `is_correct` 对齐 baseline/adversarial 的 selected correctness：baseline 始终选择 Teacher 第一条候选；adversarial 先匹配 baseline 所选候选的 `is_correct`，再在同正确性候选中选择可计算 NLL 且 `student_mean_nll` 最大的一条。`is_valid_candidate`、截断状态和其它质量字段保留用于解释 Teacher 分布、截断率、正确率和训练后结果，不作为 selection filter。

## Token-Level Decoding Outputs

vLLM-dual hard/soft 使用独立 token-level 链路，目录为 `src/vllm_dual_decoding/`。它不接入 response-level 的 `src/pre_exp/` 主入口，但产物字段应尽量兼容现有评分、SFT-ready 数据和 eval 消费约定。当前链路已覆盖 data-side smoke：生成、Student NLL 打分和数据质量分析；并新增 SFT-ready 数据构建和 full pipeline 串联入口，后续可直接启动三组训练、checkpoint eval 和 final eval。

建议模式名：

- `teacher_plain`：普通 vLLM Teacher 生成，不传 `dual_model_config`。
- `teacher_token_hard`：`dual_model_config.adversarial_mode=hard`。
- `teacher_token_soft`：`dual_model_config.adversarial_mode=soft`。

这三个名字暂时表示三种独立生成模式，不是 response-level 的 `teacher_baseline` / `teacher_adversarial` selection mode。

token-level data-side smoke 的 raw generation/candidate 记录应尽量包含：

- `generation_mode`：上述模式名。
- `dual_model_config`：hard/soft 的关键参数摘要；plain 可为 `null`。
- `worker_cls` 或等价 marker：用于证明 plain 没走 dual worker、hard/soft 走了 dual worker。
- `adversarial_mode`：plain 为 `null`，hard/soft 为实际模式。
- `candidate_text`、`finish_reason`、`is_generation_truncated`：沿用现有候选质量分析字段。
- `intervention_summary`：如果当前 pipeline 能稳定获得 hard/soft intervention 统计则写入；否则在 run summary 中记录“仅日志可见”。

SFT-ready JSONL 仍应兼容 `src/train_sft.py --dataset-format distill_jsonl`，避免为 token-level smoke 另起训练数据格式。token-level 模式下 `selection_mode` 直接写入 `teacher_plain` / `teacher_token_hard` / `teacher_token_soft`，表示独立 generation mode，不表示 response-level candidate selection。

当前 token-level data-side 正式数据构建默认参数：

- dataset：`agentica-org/DeepScaleR-Preview-Dataset`
- samples：10000
- candidates per sample：1
- generation modes：`teacher_plain`、`teacher_token_hard`、`teacher_token_soft`
- `temperature=0.7`、`top_p=0.8`
- `max_new_tokens=4096`、`max_model_len=8192`、`score_max_length=8192`
- 三种模式顺序运行；每种模式内部按 8 个 TP=1 shard 并发生成和打分。

当前 data-side 输出布局：

```text
/home/disk2/shiyixuan/Anti_Distillation/result/vllm_dual_decoding/candidates/vllm_dual_deepscaler10000_k1_t0.7_p0.8_len4096_8shard/
  teacher_plain/
    shards/
    candidate_pool.jsonl
    scored_candidates.jsonl
  teacher_token_hard/
    shards/
    candidate_pool.jsonl
    scored_candidates.jsonl
  teacher_token_soft/
    shards/
    candidate_pool.jsonl
    scored_candidates.jsonl
/home/disk2/shiyixuan/Anti_Distillation/result/vllm_dual_decoding/analysis/vllm_dual_deepscaler10000_k1_t0.7_p0.8_len4096_8shard/
  data_quality_summary.json
  logs/
```

data-side 阶段不要求产出 checkpoint 或 eval JSON；这些属于 full smoke。SFT-ready distill JSONL 由 `src/vllm_dual_decoding/build_distill_dataset.py` 或 `src/vllm_dual_decoding/run_full_pipeline.py --stages build_distill` 构建。

2026-05-14 已处理的 DeepScaleR 8000 条 token-level 数据布局：

```text
result/vllm_dual_decoding/candidates/vllm_dual_deepscaler8000_k1_t0.7_p0.8_len4096/
  plain_bs128_s128/scored_candidates.jsonl
  hard_bs96_s80/scored_candidates.jsonl
  soft_bs96_s80/scored_candidates.jsonl
result/vllm_dual_decoding/datasets/vllm_dual_deepscaler8000_k1_t0.7_p0.8_len4096/
  distill_teacher_plain.jsonl
  distill_teacher_token_hard.jsonl
  distill_teacher_token_soft.jsonl
  deepscaler_holdout_eval_1024_seed42.jsonl
```

三份 distill JSONL 均为 8000 条，holdout eval JSONL 为 1024 条，排除同 seed-42 的 8000 条训练子集。

## Main Outputs

- `candidate_pool.jsonl`：Teacher 生成候选池。
- `scored_candidates.jsonl`：候选质量字段和 Student NLL 打分。
- `teacher_baseline.selected.jsonl`：baseline 选择结果。
- `teacher_adversarial.selected.jsonl`：adversarial 选择结果。
- `distill_teacher_baseline.jsonl`：可用于 SFT 的 baseline 数据。
- `distill_teacher_adversarial.jsonl`：可用于 SFT 的 adversarial 数据。
- `dataset_summary.json`：候选质量、选中样本质量、长度和 NLL gap 统计。
