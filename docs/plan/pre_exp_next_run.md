# DeepScaleR Pre-Experiment Next Run Plan

本文档记录 256-sample DeepScaleR data-only smoke 后的实验建议。

## Smoke Result Summary

运行配置：

- dataset：`agentica-org/DeepScaleR-Preview-Dataset`
- samples：256
- candidates per sample：10
- temperature：1.0
- top_p：0.95
- max_new_tokens：1024

核心观察：

- 候选总数：2560
- valid candidate rate：45.3%
- candidate-level correct rate：26.7%
- 至少 1 个正确候选的题目：132/256
- 至少 1 个未截断且正确候选的题目：121/256
- selection fallback rate：48.4%
- 被长度截断的候选：1400/2560
- 10 个候选全被截断的题目：97/256

结论：DeepScaleR 比 GSM8K 更适合后续实验，但 `max_new_tokens=1024` 截断过重，不适合直接进入训练。

## Recommended Main Settings

建议下一次正式数据侧/训练实验从以下参数开始：

```bash
NUM_CANDIDATES=10
TEMPERATURE=0.9
TOP_P=0.9
GEN_MAX_NEW_TOKENS=2048
SCORE_MAX_LENGTH=6144
MAX_MODEL_LEN=6144
```

理由：

- 保留 `k=10`，提高每题至少一个正确候选的概率。
- 将 `temperature/top_p` 从 `1.0/0.95` 收回到 `0.9/0.9`，减少过长和错误候选。
- 将生成长度提高到 2048，缓解 DeepScaleR 推理链截断。
- Student score 长度同步提高，避免 teacher 能生成但 scorer 又截断。

## Required Code/Policy Check Before Training

- selection policy 应严格优先选择“未截断且正确”的候选。
- `is_correct` 不能单独代表可训练质量，因为少量截断候选也可能提前写出正确答案。
- 数据分析应报告 candidate-level、sample-level、usable-correct 和 fallback 指标。
- 如果 fallback 仍接近或超过 40%，不建议直接进入主训练。

## Success Criteria For Main Data

- 截断候选比例显著低于 smoke。
- 至少 1 个未截断且正确候选的题目比例足够高。
- baseline/adversarial 的 NLL gap 保持正值，且两组选中不同候选的样本占比不太低。
- baseline/adversarial 的长度分布接近，避免 adversarial 差异主要由长度解释。
