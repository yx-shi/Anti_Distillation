# Task: vLLM-Dual Full Smoke Pipeline

## Goal

把已实现的 vLLM-dual hard/soft token-level adversarial decoding 接入现有 `src/pre_exp/` 框架，跑通一个完整 smoke：生成、质量分析、SFT-ready 数据构建、短训和小样本 eval。

## Required Reading

- `AGENTS.md`
- `docs/README.md`
- `docs/spec/vllm_dual.md`
- `docs/spec/pre_exp.md`
- `docs/spec/data_and_eval.md`
- `docs/plan/vllm_dual_adversarial_decoding.md`

## Context

hard/soft decoding 的 worker-level smoke 已完成，但还没有进入正式实验框架。现有 response-level pipeline 在 `src/pre_exp/` 和 `scripts/run_pre_exp_smoke_pipeline.sh`，数据产物约定见 `docs/spec/data_and_eval.md`。下一步需要让 token-level 生成策略复用这套目录、schema、训练和评测闭环，而不是继续把正式逻辑堆在 `vllm_dual/test_dual_worker.py`。

当前必须保留的事实：

- vLLM 运行时设置 `VLLM_USE_V1=0`。
- `dual_model_config.adversarial_mode` 支持 `special_token`、`hard`、`soft`。
- 不传 `dual_model_config` 时普通生成必须继续走 `vllm.worker.worker.Worker`。
- 大体量 checkpoint 和训练产物写到 `/home/disk2/shiyixuan`。

## Implementation List

- 设计 token-level 生成模式在 `src/pre_exp/` 中的入口：
  - 可以扩展 `teacher_generate.py`，也可以新增专用脚本。
  - CLI 至少能切换 `plain`、`hard`、`soft`。
  - hard/soft 参数需要从 CLI 或脚本顶部变量传入，不要手改源码。
- 明确 token-level 输出 schema：
  - 记录生成模式、dual/adversarial 配置、是否启用 `DualModelWorker`。
  - 保留与现有 candidate/dataset 分析兼容的字段。
  - 记录可用于分析的 intervention 标记或 step 统计；如果当前代码只能从日志获得，要在 task 结果中说明限制。
- 增加一个 full smoke 串联脚本，建议新建 `scripts/run_vllm_dual_full_smoke_pipeline.sh`：
  - 生成 `teacher_plain`、`teacher_token_hard`、`teacher_token_soft` 三组小样本数据。
  - 对三组输出做基础质量分析和 Student NLL 分析。
  - 构建 SFT-ready distill JSONL。
  - 每组跑极短 SFT smoke。
  - 每组跑固定小样本 eval。
- 先用 GSM8K 16-32 条样本验证链路；链路稳定后再切 DeepScaleR 32-64 条。
- smoke 结束后更新 `docs/spec/vllm_dual.md`、`docs/spec/project_state.md` 和本 task 的状态。

## Acceptance Criteria

- `plain` 生成日志显示 `parallel_config.worker_cls: vllm.worker.worker.Worker`。
- `hard` 生成日志显示 `SMOKE_DUAL_EFFECTIVE ... adv_mode=hard` 或等价的正式 pipeline marker，并出现 `ADISTILL_DUAL_ADVERSARIAL enabled mode=hard`。
- `soft` 生成日志显示 `SMOKE_DUAL_EFFECTIVE ... adv_mode=soft` 或等价的正式 pipeline marker，并出现 `ADISTILL_DUAL_ADVERSARIAL enabled mode=soft`。
- 三种模式都产出 SFT-ready JSONL，且字段能被 `src/train_sft.py --dataset-format distill_jsonl` 消费。
- 三种模式都完成极短训练并产出 checkpoint 或 final checkpoint。
- 三种模式都完成固定小样本 eval，并产出 JSON summary。
- 产物路径、日志路径和关键指标写入一个 run summary。

## Stop Conditions

- hard/soft 没有实际启用 dual worker。
- 普通 plain 路径误走 dual worker。
- Teacher/Student vocab shape mismatch。
- 输出大面积为空、无法抽答案或全部截断。
- checkpoint 或大 JSONL 准备写入仓库磁盘而不是 `/home/disk2/shiyixuan` 或受控 result 目录。

## Do Not

- 不要把 `vllm_dual/` 加到 `PYTHONPATH`。
- 不要跳过 `sync.sh` dry run；如果修改了 `vllm_dual/vllm/`，必须先 dry run 再同步。
- 不要默认跑大规模训练或长 DeepScaleR 生成；先完成小 full smoke。
- 不要删除已有实验结果、checkpoint 或 `.sync_backups/`。
