# vLLM-Dual Full Smoke Plan

本文档记录 token-level adversarial decoding 完成首版后的下一阶段路线。已完成的 hard/soft 实现归档见 `archive/vllm_dual_adversarial_decoding_20260427.md`，可执行任务见 `../task/vllm_dual_full_smoke_pipeline.md`。

## Goal

把 vLLM-dual hard/soft decoding 接入现有 `src/pre_exp/` 实验框架，跑通一个完整 smoke：Teacher 生成、候选/输出质量分析、SFT 数据构建、小规模训练和评测。

## Current Facts

- vLLM-dual 已实现 `adversarial_mode=special_token|hard|soft`。
- hard/soft 使用 vLLM sampler 处理后的 full-vocab `probs/logprobs`，实现细节见 `../spec/vllm_dual.md`。
- `vllm_dual/test_dual_worker.py` 级别 smoke 已通过 hard 和 soft，但这只验证 worker 链路，不验证数据产物是否能被现有训练/评测框架消费。
- 现有完整实验框架在 `src/pre_exp/` 和 `scripts/run_pre_exp_smoke_pipeline.sh`，当前主要服务 response-level baseline/adversarial selection。

## Full Smoke Scope

第一版 full smoke 只做最小闭环，不追求最终实验质量：

- 数据集：优先使用较小子集，建议从 GSM8K 16-32 条开始；若工程稳定，再切 DeepScaleR 32-64 条。
- 生成模式：至少包含 `teacher_plain`、`teacher_token_hard`、`teacher_token_soft`。
- 生成长度：先用短上限验证链路，再放大到数学题可用长度。
- 训练：每种模式只跑极短 SFT smoke，确认 dataset schema、checkpoint 输出和 eval 消费路径。
- 评测：固定小子集 eval，记录 accuracy、finish_reason/截断率、输出长度、Student NLL 分布。

## Integration Direction

- 不把正式逻辑写进 `vllm_dual/test_dual_worker.py`；该文件继续保留为 worker-level smoke。
- 在 `src/pre_exp/` 或 `scripts/` 中增加正式 pipeline 入口，使 token-level decoding 输出走和 response-level 预实验相同的数据、训练、评测目录规范。
- 保留普通 vLLM 生成路径作为对照，确保不传 `dual_model_config` 时仍走 `vllm.worker.worker.Worker`。
- hard/soft 配置应能在脚本顶部或 CLI 参数中切换，避免手改源码。
- 大体量产物继续放在 `/home/disk2/shiyixuan`，仓库内只放小型分析 JSON 和日志路径引用。

## Required Outputs

full smoke 应至少产出：

- 每种生成模式的 raw generation/candidate JSONL。
- 每种生成模式的 SFT-ready distill JSONL。
- 数据分析 summary，包含有效率、截断率、输出长度、Student NLL 和 hard/soft intervention 标记统计。
- 每种模式的短训 checkpoint 或 final checkpoint。
- 小样本 eval JSON。
- 一个简短 run summary，说明 hard/soft 是否真的被启用、普通路径是否仍为普通 worker、是否有 tokenizer/vocab 对齐问题。

## Stop Conditions

遇到以下问题时先停止，不要继续跑训练：

- hard/soft 生成日志没有出现 `ADISTILL_DUAL_ADVERSARIAL enabled mode=...`。
- 普通对照生成走到了 `DualModelWorker`。
- Teacher/Student vocab shape mismatch。
- 生成输出大面积为空、无法抽答案或全部截断。
- full smoke 产物将写入仓库大目录或 checkpoint 目录配置错误。
