# Task: vLLM-Dual Token-Level Adversarial Decoding

## Goal

在 vLLM-dual 中实现 token-level hard/soft adversarial decoding，使 Teacher 和 Student 可以针对同一 prefix 共同决定 next token。

## Required Reading

- `AGENTS.md`
- `docs/README.md`
- `docs/spec/vllm_dual.md`
- `docs/plan/vllm_dual_adversarial_decoding.md`
- `docs/anti_distillation.md`

## Context

当前项目基于 vLLM 0.8.5，运行时必须设置 `VLLM_USE_V1=0`。普通预实验路径不应默认启用 dual worker；只有显式 dual/adversarial 配置时才切换。

## Implementation List

- 明确 dual/adversarial 配置入口和默认关闭行为。
- 实现 Teacher/Student 同 prefix forward 的最小闭环。
- 实现 hard mode：Teacher 候选集合中选 Student 概率低的 token。
- 实现 soft mode：根据 Teacher/Student logits 重加权采样。
- 添加 smoke 检查，区分普通 worker 和 dual worker 日志。
- 同步到 conda 环境前先用 `DRY_RUN=1` 检查。

## Current Status

- 已完成 hard/soft token-level adversarial decoding 的首版实现。
- `DualModelConfig` 新增 `adversarial_mode`、`hard_candidate_top_k`、`hard_candidate_top_p`、`soft_student_weight`、`soft_temperature`、`debug_log_interval`。
- `DualModelWorker` 在 hard/soft 模式下打开 teacher/student sampler 的 `include_gpu_probs_tensor`，使用完整 GPU `probs/logprobs` 合成最终 token。
- smoke 脚本已支持 `--adv-mode hard|soft` 参数切换；`vllm_dual/test.sh` 可用 `ADV_MODE=hard|soft` 透传。
- smoke 会打印 `SMOKE_DUAL_REQUEST`、`SMOKE_DUAL_CONFIG_SUPPORTED`、`SMOKE_DUAL_CONFIG_FORWARDED`、`SMOKE_DUAL_EFFECTIVE`，用于确认 dual worker 和 adversarial 配置是否被当前同步环境读取。
- 2026-04-27 已完成 `ASSUME_YES=1 DRY_RUN=1 bash sync.sh`、`ASSUME_YES=1 bash sync.sh`、hard smoke、soft smoke 和普通 vLLM smoke。

## Acceptance Criteria

- [x] 普通 vLLM smoke 仍走 `vllm.worker.worker.Worker`。
- [x] dual smoke 走 `vllm.worker.dual_worker.DualModelWorker`。
- [x] hard/soft mode 可通过参数切换。
- [x] smoke 输出能证明 adversarial decoding 配置被读取并生效。

## Do Not

- 不要把 `vllm_dual/` 加到 `PYTHONPATH`。
- 不要跳过 `sync.sh` dry run。
- 不要破坏普通 vLLM 预实验路径。
