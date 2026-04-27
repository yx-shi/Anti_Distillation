# Task: DeepScaleR Main Pre-Experiment

## Goal

基于 DeepScaleR smoke 结果，准备并运行更可靠的 response-level anti-distillation 主实验。

## Required Reading

- `AGENTS.md`
- `docs/README.md`
- `docs/spec/pre_exp.md`
- `docs/spec/data_and_eval.md`
- `docs/plan/pre_exp_next_run.md`

## Context

GSM8K 难度偏低，Student base 正确率较高，adversarial 差异不明显。DeepScaleR smoke 显示数据更难，但 `max_new_tokens=1024` 截断过重，当前 smoke 结果不适合直接训练。

2026-04-28 更新：selection policy 已按新的实验口径切换为完整 Teacher 分布蒸馏，不再优先过滤“未截断且正确”的候选。baseline 选 Teacher 第一条候选，adversarial 选 Student NLL 最大候选；质量字段只用于分析。

## Implementation List

- [completed] 检查并修正 selection policy，使 baseline/adversarial 学完整 Teacher 分布，而不是只学正确候选子分布。
- 增强 dataset analysis，报告 sample-level valid/correct、截断率、选中样本质量和 NLL gap。
- 准备 DeepScaleR 主实验脚本，使用 `docs/plan/pre_exp_next_run.md` 的建议参数。
- 先只跑数据侧；数据质量达标后再启动训练。
- 训练前确认 baseline/adversarial 数据长度分布接近。

## Acceptance Criteria

- 数据侧产物包含 candidate pool、scored candidates、selection、distill jsonl 和 summary。
- summary 明确显示截断率、valid/correct sample rate、selected-candidate quality 和 NLL gap。
- 如果数据质量不达标，停止在数据侧分析，不启动训练。

## Do Not

- 不要在数据质量不明时直接跑 SFT。
- 不要覆盖历史 smoke/main 结果目录。
- 不要删除 `/home/disk2/shiyixuan` 下已有 checkpoint。
