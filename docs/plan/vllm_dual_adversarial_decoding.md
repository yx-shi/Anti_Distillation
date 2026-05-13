# vLLM-Dual Data-Side And Full Smoke Plan

本文档记录 token-level adversarial decoding 完成首版后的下一阶段路线。已完成的 hard/soft 实现归档见 `archive/vllm_dual_adversarial_decoding_20260427.md`，可执行任务见 `../task/vllm_dual_full_smoke_pipeline.md`。

## Goal

建立与 response-level `src/pre_exp/` 并列的 vLLM-dual token-level 实验链路。当前阶段先完成 worker smoke 复验和 data-side smoke：Teacher 生成、候选/输出质量分析、Student NLL 打分；数据质量达标后再进入 SFT-ready 数据构建、小规模训练和评测。

## Current Facts

- vLLM-dual 已实现 `adversarial_mode=special_token|hard|soft`。
- hard/soft 使用 vLLM sampler 处理后的 full-vocab `probs/logprobs`，实现细节见 `../spec/vllm_dual.md`。
- `vllm_dual/test_dual_worker.py` 级别 smoke 已通过 hard 和 soft，但这只验证 worker 链路，不验证数据产物是否能被现有训练/评测框架消费。
- 现有 response-level 实验框架在 `src/pre_exp/`、`scripts/run_pre_exp_pipeline.sh` 和 `scripts/run_pre_exp_deepscaler_smoke_data.sh`，只服务 response-level baseline/adversarial selection；token-level vLLM-dual 链路放在 `src/vllm_dual_decoding/`。
- 当前 GPU 资源不足时，允许先只完成代码与文档准备，不启动 vLLM 生成、SFT 训练或 checkpoint eval。

## Data-Side Smoke Scope

正式 full smoke 前先跑 data-side smoke：

- 前置检查：确认 vLLM-dual 已同步到 conda 环境，复跑 `vllm_dual/test.sh` 的 hard/soft worker smoke。
- 数据集：GSM8K 24 条固定子集，后续再切 DeepScaleR 32-64 条。
- 生成模式：`teacher_plain`、`teacher_token_hard`、`teacher_token_soft`。
- 生成长度：先用 `max_new_tokens=512`、`max_model_len=2048` 验证链路和基本输出质量。
- 数据分析：记录 accuracy、finish_reason/截断率、输出长度、Student NLL 分布；hard/soft intervention 统计当前以日志 marker 为准。
- 不跑训练，不产出 checkpoint，不做最终 eval。

## Full Smoke Scope

data-side smoke 通过后，再做最小 full smoke，不追求最终实验质量：

- 数据集：沿用 data-side smoke 中质量达标的小子集；若工程稳定，再切 DeepScaleR 32-64 条。
- 生成模式：至少包含 `teacher_plain`、`teacher_token_hard`、`teacher_token_soft`。
- 生成长度：先用短上限验证链路，再放大到数学题可用长度。
- 训练：每种模式只跑极短 SFT smoke，确认 dataset schema、checkpoint 输出和 eval 消费路径。
- 评测：固定小子集 eval，记录 accuracy、finish_reason/截断率、输出长度、Student NLL 分布。

## Integration Direction

- 不把正式逻辑写进 `vllm_dual/test_dual_worker.py`；该文件继续保留为 worker-level smoke。
- 在 `src/vllm_dual_decoding/` 和 `scripts/run_vllm_dual_*.sh` 中维护正式 pipeline 入口；可复用共享 prompt、grading、SFT-ready 字段约定，但不依赖 `src/pre_exp/` 主入口。
- 保留普通 vLLM 生成路径作为对照，确保不传 `dual_model_config` 时仍走 `vllm.worker.worker.Worker`。
- hard/soft 配置应能在脚本顶部或 CLI 参数中切换，避免手改源码。
- data-side smoke 入口使用 `scripts/run_vllm_dual_data_smoke.sh`；后续 full smoke 可在它通过后再扩展训练和 eval 阶段。
- 大体量产物继续放在 `/home/disk2/shiyixuan`，仓库内只放小型分析 JSON 和日志路径引用。

## Required Outputs

data-side smoke 应至少产出：

- 每种生成模式的 raw generation/candidate JSONL。
- 数据分析 summary，包含有效率、截断率、输出长度、Student NLL 和 hard/soft intervention 标记统计。
- 一个简短 data quality summary，说明 hard/soft 是否真的被启用、普通路径是否仍为普通 worker、是否有 tokenizer/vocab 对齐问题。

后续 full smoke 再追加：

- 每种生成模式的 SFT-ready distill JSONL。
- 每种模式的短训 checkpoint 或 final checkpoint。
- 小样本 eval JSON。

## Stop Conditions

遇到以下问题时停止在 data-side，不要进入训练：

- hard/soft 生成日志没有出现 `ADISTILL_DUAL_ADVERSARIAL enabled mode=...`。
- 普通对照生成走到了 `DualModelWorker`。
- Teacher/Student vocab shape mismatch。
- 生成输出大面积为空、无法抽答案或全部截断。
- full smoke checkpoint 或大体量产物准备写入仓库磁盘而不是 `/home/disk2/shiyixuan`。
