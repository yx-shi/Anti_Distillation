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

## Implementation List

- 检查并修正 selection policy，使训练候选优先来自“未截断且正确”的集合。
- 增强 dataset analysis，报告 sample-level usable-correct、截断率、fallback 和 NLL gap。
- 准备 DeepScaleR 主实验脚本，使用 `docs/plan/pre_exp_next_run.md` 的建议参数。
- 先只跑数据侧；数据质量达标后再启动训练。
- 训练前确认 baseline/adversarial 数据长度分布接近。

## Acceptance Criteria

- 数据侧产物包含 candidate pool、scored candidates、selection、distill jsonl 和 summary。
- summary 明确显示截断率、fallback rate、usable-correct sample rate 和 NLL gap。
- 如果数据质量不达标，停止在数据侧分析，不启动训练。

## Do Not

- 不要在数据质量不明时直接跑 SFT。
- 不要覆盖历史 smoke/main 结果目录。
- 不要删除 `/home/disk2/shiyixuan` 下已有 checkpoint。
