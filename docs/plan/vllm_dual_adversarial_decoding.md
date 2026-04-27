# vLLM-Dual Adversarial Decoding Plan

本文档描述 token-level adversarial decoding 的实现方向。具体任务拆分见 `docs/task/vllm_dual_token_level_decoding.md`。

## Goal

修改 vLLM 0.8.5 V0 路径，使一个 decoding step 中 Teacher 和 Student 可以针对同一 prefix forward，并用 Student 概率影响 next token 选择。

## Implemented Modes

- Hard mode：在 Teacher 可接受候选集合中，选择 Student 概率最低的 token。
- Soft mode：基于 Teacher/Student sampler logprobs 做重加权采样，用 `soft_student_weight` 控制对抗强度。

## Design Principles

- 保持普通 vLLM 路径默认行为不变。
- 只有显式传入 dual/adversarial 配置时才启用 dual worker。
- 优先在 worker/model runner 边界内收敛复杂度，避免大范围改 scheduler。
- 先完成单机 smoke，再扩大到多卡和长序列。

## Current Interface

- 配置入口：`dual_model_config`，只有显式传入时才切到 `vllm.worker.dual_worker.DualModelWorker`。
- 模式字段：`adversarial_mode`，支持 `special_token`、`hard`、`soft`。
- hard 候选集合：Teacher sampler 概率的 `hard_candidate_top_k`，可再用 `hard_candidate_top_p` 截断。
- soft 公式：在 Teacher 候选集合内计算 `teacher_logprob - soft_student_weight * student_logprob`，再按 `soft_temperature` 采样。
- 当前实现使用 vLLM sampler 处理后的 full-vocab `probs/logprobs`，因此包含 temperature、top-k/top-p/min-p 和 repetition/presence/frequency penalty 等采样侧处理。
- 当前要求 Teacher/Student tokenizer vocab 和 sampler rows 对齐；不对齐时会在 tensor shape mismatch 处报错。
- 输出标记：初始化打印 `ADISTILL_DUAL_ADVERSARIAL enabled mode=...`，step 打印 `ADISTILL_DUAL_ADVERSARIAL step mode=... interventions=.../...`。

## Smoke Status

- `vllm_dual/test_dual_worker.py` 已支持 `--adv-mode hard|soft`，并打印 `SMOKE_DUAL_REQUEST`、`SMOKE_DUAL_CONFIG_SUPPORTED`、`SMOKE_DUAL_CONFIG_FORWARDED`、`SMOKE_DUAL_EFFECTIVE` 四类可见标记。
- `vllm_dual/test.sh` 可通过 `ADV_MODE=hard|soft` 或追加脚本参数切换 smoke 模式，例如 `ADV_MODE=soft bash vllm_dual/test.sh --max-tokens 16`。
- smoke 会运行时探测当前 conda 环境中已同步的 `vllm.config.DualModelConfig` 字段，并在 `SMOKE_DUAL_CONFIG_FORWARDED` 中显示 hard/soft 参数实际转发到的字段名。
- 2026-04-27 已完成 `sync.sh` dry run、正式同步、hard smoke、soft smoke 和普通 vLLM smoke。hard/soft smoke 均显示 `SMOKE_DUAL_EFFECTIVE worker_cls=vllm.worker.dual_worker.DualModelWorker adv_mode=...`，普通 smoke 显示 `parallel_config.worker_cls: vllm.worker.worker.Worker`。
