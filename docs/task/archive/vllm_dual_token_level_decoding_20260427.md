# Archive: vLLM-Dual Token-Level Adversarial Decoding

归档日期：2026-04-27

## Goal

在 vLLM-dual 中实现 token-level hard/soft adversarial decoding，使 Teacher 和 Student 可以针对同一 prefix 共同决定 next token。

## Completed Work

- 明确 dual/adversarial 配置入口和默认关闭行为。
- 实现 Teacher/Student 同 prefix forward 的最小闭环。
- 实现 hard mode：Teacher 候选集合中选 Student 概率低的 token。
- 实现 soft mode：根据 Teacher/Student sampler logprobs 重加权采样。
- 添加 smoke 检查，区分普通 worker 和 dual worker 日志。
- 按规范先执行 `DRY_RUN=1`，再同步到 conda 环境。

## Current Status

- `DualModelConfig` 新增 `adversarial_mode`、`hard_candidate_top_k`、`hard_candidate_top_p`、`soft_student_weight`、`soft_temperature`、`debug_log_interval`。
- `DualModelWorker` 在 hard/soft 模式下打开 teacher/student sampler 的 `include_gpu_probs_tensor`，使用完整 GPU `probs/logprobs` 合成最终 token。
- `vllm_dual/test_dual_worker.py` 支持 `--adv-mode hard|soft` 参数切换；`vllm_dual/test.sh` 可用 `ADV_MODE=hard|soft` 透传。
- smoke 会打印 `SMOKE_DUAL_REQUEST`、`SMOKE_DUAL_CONFIG_SUPPORTED`、`SMOKE_DUAL_CONFIG_FORWARDED`、`SMOKE_DUAL_EFFECTIVE`。

## Acceptance

- [x] 普通 vLLM smoke 仍走 `vllm.worker.worker.Worker`。
- [x] dual smoke 走 `vllm.worker.dual_worker.DualModelWorker`。
- [x] hard/soft mode 可通过参数切换。
- [x] smoke 输出能证明 adversarial decoding 配置被读取并生效。

## Follow-Up Task

下一步执行 `docs/task/vllm_dual_full_smoke_pipeline.md`，把 hard/soft decoding 接入现有 pre-exp 框架并跑完整 smoke。
