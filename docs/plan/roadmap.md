# Roadmap

本文档记录项目阶段性方向。具体实现不要直接写在这里，应拆到 `docs/task/*`。

## Long-Term Goal

实现并评估 anti-distillation decoding：Teacher 输出仍保持数学任务质量，但生成分布对 Student 更难拟合，从而降低黑盒蒸馏效率。

## Completed

- 熟悉 Hugging Face Transformers 推理和 tokenizer 机制。
- 手写并整理 SFT 训练流程。
- 将 SFT 训练接入 GSM8K。
- 用 FSDP 跑通多卡训练。
- 用 vLLM 0.8.5 跑通 offline/online inference。
- 完成 response-level anti-distillation 预实验框架。
- 完成 GSM8K 预实验并发现任务偏简单。
- 完成 DeepScaleR 256-sample data-only smoke，确认数据更难，但 1024 生成长度不足。
- 完成 vLLM-dual hard/soft token-level adversarial decoding 首版，并通过 hard、soft、普通 vLLM 三类 smoke。

## Current Phase

1. 用 DeepScaleR 重跑更高质量的 response-level 预实验。
2. 根据数据质量与训练结果判断 response-level 方法是否有可观测 anti-distillation 信号。
3. 将 vLLM-dual hard/soft decoding 接入现有 `src/pre_exp/` 框架，跑完整 smoke：生成、质量分析、SFT、小规模 eval。

## Next Decisions

- DeepScaleR 主实验的样本规模、生成长度和采样参数。
- selection policy 是否严格排除截断候选。
- vLLM-dual full smoke 的最小数据规模、Teacher/Student 组合、输出 schema 和训练预算。
- token-level hard/soft 结果如何与 response-level baseline/adversarial 结果做可比分析。
