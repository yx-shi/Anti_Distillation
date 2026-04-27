# Project State

本文档记录当前项目事实，供 agent 快速建立上下文。它不是计划，也不是任务清单；发生事实变化时再更新。

## Research Focus

项目目标是研究 anti-distillation：在 Teacher 输出质量基本不下降的前提下，让小模型更难通过蒸馏学习 Teacher 的能力。

当前采用两条路线：

- response-level 预实验：Teacher 对同一题生成多个候选，选择 Student 平均 NLL 更高的正确候选作为 adversarial distillation 数据。
- token-level adversarial decoding：修改 vLLM，使 Teacher 和 Student 在同一 prefix 上共同 forward，并在 token 级别选择对 Student 更难的 next token；hard/soft 首版已在 vLLM-dual 路径实现并通过 smoke。

## Current Code Layout

- `src/pre_exp/`：预实验数据侧、候选打分、候选选择、数据分析和最终评测脚本。
- `src/sft/`：SFT 训练框架，使用 Transformers + PyTorch/FSDP。
- `src/train_sft.py`：SFT 训练入口。
- `grading/`：数学题答案抽取和判分逻辑。
- `examples/vllm_offline/`：vLLM offline inference 学习示例，不是正式实验主入口。
- `scripts/`：实验串联脚本和辅助脚本。

## Environment

- Conda 环境：`adistill-unified`
- vLLM：`0.8.5`
- vLLM engine：运行时设置 `VLLM_USE_V1=0`
- Teacher：`/home/disk1/public_checkpoint/Qwen3-8B`
- Student：`/home/disk1/public_checkpoint/Qwen3-1.7B`
- 大体量 checkpoint 与训练产物目录：`/home/disk2/shiyixuan`

## Completed Learning/Engineering Milestones

- 已熟悉基础 Hugging Face Transformers 推理流程。
- 已实现并跑通过 SFT 训练框架。
- 已用 FSDP 跑通多卡训练。
- 已用 vLLM 跑通 offline 和 online inference。
- 已了解 vLLM V0 的关键概念：batch inference、prefill、decoding、scheduler、paged attention、swap 等。
- 已启动 vLLM-dual 修改方向，重点文件包括 worker、arg_utils 和 config 相关逻辑。
- 已实现 vLLM-dual hard/soft token-level adversarial decoding 首版：显式 `dual_model_config` 才启用，普通 vLLM 路径保持默认 worker。

## Current Experimental State

- GSM8K 预实验显示任务偏简单，Student base 正确率较高，蒸馏提升和 adversarial 差异不明显。
- 已开始切换到 DeepScaleR 数据集方向。
- 256-sample DeepScaleR data-only smoke 已完成，结论记录在 `plan/pre_exp_next_run.md`。

## Maintenance Rule

如果代码或实验事实变化，优先更新本文件或对应 `docs/spec/*`。不要把临时日志、长命令输出或一次性分析堆进 `AGENTS.md`。
