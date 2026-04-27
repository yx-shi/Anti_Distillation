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

## Smoke128 Len4096 Result Summary

2026-04-28 运行了新的 data-only smoke：

- experiment：`deepscaler_smoke128_k8_t0.9_p0.85_len4096`
- samples：128
- candidates per sample：8
- temperature：0.9
- top_p：0.85
- max_new_tokens：4096
- score_max_length / max_model_len：8192

核心观察：

- 候选总数：1024
- empty candidate rate：0.0%
- extractable candidate rate：100.0%
- valid candidate rate：93.4%
- candidate-level correct rate：36.2%
- generation truncated rate：6.6%
- 至少 1 个 valid candidate 的题目：128/128
- 至少 1 个正确候选的题目：82/128
- 至少 1 个 valid 且正确候选的题目：82/128
- 有任一截断候选的题目：29/128
- 8 个候选全被截断的题目：0/128

在完整 Teacher 分布 selection policy 下：

- baseline selected correct rate：39.1%
- adversarial selected correct rate：25.8%
- baseline selected truncated rate：3.9%
- adversarial selected truncated rate：6.2%
- baseline/adversarial 选中不同候选的样本占比：91.4%
- 平均 NLL gap：0.0946
- baseline 平均 completion token 数：1084.4
- adversarial 平均 completion token 数：814.6
- Student score 阶段 `score_truncated`：0/1024

结论：`max_new_tokens=4096` 基本解决了上一轮 smoke 的严重截断问题，且 adversarial selection 能稳定选到与 baseline 不同、Student NLL 更高的候选。需要注意的是，在完整 Teacher 分布蒸馏口径下，adversarial 选中样本正确率低于 baseline，后续训练曲线差异必须结合 selected-candidate correctness / truncation 一起解释。

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

- selection policy 已从“正确且未截断候选内重排”改为“完整 Teacher 分布蒸馏”：baseline 选第一条 Teacher 候选，adversarial 选 Student NLL 最大候选。
- `is_correct`、`is_valid_candidate`、截断状态仍需报告，但只作为解释变量，不作为 selection filter。
- 数据分析应报告 candidate-level、sample-level、valid/correct、截断率、selected-candidate 正确率和 NLL gap。
- 如果截断率仍过高，训练结论需要谨慎解释，因为 Student 学到的是 Teacher 的完整输出分布，其中包含截断与错误样本。

## Success Criteria For Main Data

- 截断候选比例显著低于 smoke。
- baseline/adversarial 的 NLL gap 保持正值，且两组选中不同候选的样本占比不太低。
- baseline/adversarial 的长度分布接近，避免 adversarial 差异主要由长度解释。
- 两组选中样本的正确率、截断率差异可解释，避免把训练差异误读为纯 anti-distillation 信号。
