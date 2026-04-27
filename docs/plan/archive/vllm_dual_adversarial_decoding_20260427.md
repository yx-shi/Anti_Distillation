# Archive: vLLM-Dual Adversarial Decoding Implementation

归档日期：2026-04-27

## Completed Goal

修改 vLLM 0.8.5 V0 路径，使一个 decoding step 中 Teacher 和 Student 可以针对同一 prefix forward，并用 Student 概率影响 next token 选择。

## Implemented Modes

- Hard mode：在 Teacher 可接受候选集合中，选择 Student 概率最低的 token。
- Soft mode：基于 Teacher/Student sampler logprobs 做重加权采样，用 `soft_student_weight` 控制对抗强度。

## Final Interface

- 配置入口：`dual_model_config`，只有显式传入时才切到 `vllm.worker.dual_worker.DualModelWorker`。
- 模式字段：`adversarial_mode`，支持 `special_token`、`hard`、`soft`。
- hard 候选集合：Teacher sampler 概率的 `hard_candidate_top_k`，可再用 `hard_candidate_top_p` 截断。
- soft 公式：在 Teacher 候选集合内计算 `teacher_logprob - soft_student_weight * student_logprob`，再按 `soft_temperature` 采样。
- 当前实现使用 vLLM sampler 处理后的 full-vocab `probs/logprobs`。
- 当前要求 Teacher/Student tokenizer vocab 和 sampler rows 对齐；不对齐时会在 tensor shape mismatch 处报错。
- 输出标记：初始化打印 `ADISTILL_DUAL_ADVERSARIAL enabled mode=...`，step 打印 `ADISTILL_DUAL_ADVERSARIAL step mode=... interventions=.../...`。

## Verification

- `python -m py_compile vllm_dual/test_dual_worker.py vllm_dual/vllm/config.py vllm_dual/vllm/engine/arg_utils.py vllm_dual/vllm/worker/dual_worker.py` 通过。
- `ASSUME_YES=1 DRY_RUN=1 bash sync.sh` 通过。
- `ASSUME_YES=1 bash sync.sh` 通过，备份目录为 `.sync_backups/vllm_20260427_211355`。
- hard smoke 通过，日志包含 `SMOKE_DUAL_EFFECTIVE ... adv_mode=hard`。
- soft smoke 通过，日志包含 `SMOKE_DUAL_EFFECTIVE ... adv_mode=soft`。
- 普通 vLLM smoke 通过，日志显示 `parallel_config.worker_cls: vllm.worker.worker.Worker`。

## Follow-Up

后续不再继续扩展 worker-level smoke 本身，而是把 hard/soft decoding 接入现有 `src/pre_exp/` 框架，跑完整数据、训练、评测 smoke。
