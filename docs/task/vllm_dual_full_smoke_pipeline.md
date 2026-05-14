# Task: vLLM-Dual Full Smoke Pipeline

## Goal

把已实现的 vLLM-dual hard/soft token-level adversarial decoding 做成独立实验链路，与 response-level 的 `src/pre_exp/` 并列。当前先完成 full smoke 前置的 data-side smoke：同步/worker 检查、三种生成模式、质量分析和 Student NLL 打分；数据质量达标后再扩展到 SFT-ready 数据构建、短训和小样本 eval。

## Required Reading

- `AGENTS.md`
- `docs/README.md`
- `docs/spec/vllm_dual.md`
- `docs/spec/data_and_eval.md`
- `docs/plan/vllm_dual_adversarial_decoding.md`

## Context

hard/soft decoding 的 worker-level smoke 已完成历史验证，但还没有进入正式 token-level 实验链路。`src/pre_exp/` 只保留 response-level baseline/adversarial selection。下一步需要先通过 `vllm_dual/test.sh` 复验 DualModelWorker，再在 `src/vllm_dual_decoding/` 维护独立的生成、打分、分析入口；可以复用共享 prompt、grading、SFT-ready 字段约定，但不要把 token-level 逻辑继续塞进 `src/pre_exp/`。

当前 GPU 资源紧张时，只做代码和文档准备，不启动 vLLM GPU smoke、不跑训练。

当前必须保留的事实：

- vLLM 运行时设置 `VLLM_USE_V1=0`。
- `dual_model_config.adversarial_mode` 支持 `special_token`、`hard`、`soft`。
- 不传 `dual_model_config` 时普通生成必须继续走 `vllm.worker.worker.Worker`。
- 大体量 checkpoint 和训练产物写到 `/home/disk2/shiyixuan`。

2026-05-13 当前状态：

- conda 中的 vLLM 已同步到仓库 `vllm_dual/vllm`：`config.py`、`engine/arg_utils.py`、`worker/dual_worker.py` 逐字节一致。
- hard/soft worker smoke 已复验通过。
- `scripts/run_vllm_dual_data_smoke.sh` 已从 DeepScaleR 小样本 smoke 扩展为 10000 条 data build 默认设置，并内置 preflight、可选同步、可选 hard/soft worker smoke、三模式顺序生成、8 shard 并发 Student NLL 打分和 summary。
- DeepScaleR 1 条、16 token micro chain 已跑通，证明正式 Teacher/Student 组合不存在立即的 vocab shape 或 JSON schema 阻塞；正式 10000 条 data-side run 尚未运行。

2026-05-14 当前状态：

- `src/vllm_dual_decoding/build_distill_dataset.py` 已新增，用于把 token-level `scored_candidates.jsonl` 转成 `src/train_sft.py --dataset-format distill_jsonl` 可消费的 SFT-ready JSONL。
- `src/vllm_dual_decoding/run_full_pipeline.py` 已新增，支持 `build_distill`、`build_holdout`、`train`、`checkpoint_eval`、`final_eval` 阶段；默认只执行安全的数据准备阶段，训练和 rollout eval 需要显式传 `--stages train,checkpoint_eval,final_eval`。
- 本地 DeepScaleR 8000 条三组数据已完成 SFT-ready 转换：`distill_teacher_plain.jsonl`、`distill_teacher_token_hard.jsonl`、`distill_teacher_token_soft.jsonl` 均为 8000 条；`deepscaler_holdout_eval_1024_seed42.jsonl` 已构建。尚未启动长训练或 GPU rollout eval。
- 三组 SFT 训练已完成：每组 1600 step，每 200 step 记录 trainer validation loss/ppl，均产出 final checkpoint。训练曲线和 val loss/ppl 曲线已写入 `result/vllm_dual_decoding/analysis/vllm_dual_deepscaler8000_k1_t0.7_p0.8_len4096/curves/`。checkpoint/final rollout eval 尚未运行。

## Implementation List

- 前置检查阶段：
  - 先做 `sync.sh` dry run 和 conda import 检查，确认 `vllm_dual` 已同步到 conda 环境。
  - GPU 可用后分别运行 `ADV_MODE=hard bash vllm_dual/test.sh` 和 `ADV_MODE=soft bash vllm_dual/test.sh`。
  - hard/soft worker-level smoke 通过后，再运行正式 vLLM-dual token-level data-side smoke。
- 设计 token-level 生成模式在 `src/vllm_dual_decoding/` 中的入口：
  - 使用 `src/vllm_dual_decoding/teacher_generate.py --generation-mode plain|hard|soft`。
  - hard/soft 参数需要从 CLI 或脚本顶部变量传入，不要手改源码。
- 明确 token-level 输出 schema：
  - 记录生成模式、dual/adversarial 配置、是否启用 `DualModelWorker`。
  - 保留与现有 candidate/dataset 分析兼容的字段。
  - 记录可用于分析的 intervention 标记或 step 统计；如果当前代码只能从日志获得，要在 task 结果中说明限制。
- 增加一个 data-side smoke 串联脚本 `scripts/run_vllm_dual_data_smoke.sh`：
  - 生成 `teacher_plain`、`teacher_token_hard`、`teacher_token_soft` 三组小样本数据。
  - 对三组输出做基础质量分析和 Student NLL 分析。
  - 不跑 SFT，不产出 checkpoint。
- data-side smoke 质量达标后，再增加 full smoke 串联脚本：
  - 构建 SFT-ready distill JSONL。（已完成）
  - 每组跑极短 SFT smoke。（DeepScaleR 8000 三组 1600 step 训练已完成）
  - 每组跑固定小样本 eval。（入口已准备，尚未实际运行）
- 当前按用户要求准备 DeepScaleR 10000 条 token-level data build；三种模式顺序跑，每组内部 8 卡 TP=1 shard 并发。若 hard/soft 出现 OOM，优先降低 `GEN_PROMPT_BATCH_SIZE` 或 `GEN_MAX_NUM_SEQS` 后重跑对应阶段。
- smoke 结束后更新 `docs/spec/vllm_dual.md`、`docs/spec/project_state.md` 和本 task 的状态。

## Data-Side Acceptance Criteria

- `plain` 生成日志显示 `VLLM_DUAL_PIPELINE_EFFECTIVE ... worker_cls=vllm.worker.worker.Worker ... adv_mode=None` 或普通 vLLM smoke 中的 `parallel_config.worker_cls: vllm.worker.worker.Worker`。
- `hard` 生成日志显示 `VLLM_DUAL_PIPELINE_EFFECTIVE ... adv_mode=hard` 或 worker smoke 中的 `SMOKE_DUAL_EFFECTIVE ... adv_mode=hard`，并出现 `ADISTILL_DUAL_ADVERSARIAL enabled mode=hard`。
- `soft` 生成日志显示 `VLLM_DUAL_PIPELINE_EFFECTIVE ... adv_mode=soft` 或 worker smoke 中的 `SMOKE_DUAL_EFFECTIVE ... adv_mode=soft`，并出现 `ADISTILL_DUAL_ADVERSARIAL enabled mode=soft`。
- 三种模式都产出 `candidate_pool.jsonl` 和 `scored_candidates.jsonl`。
- `data_quality_summary.json` 记录三种模式的有效率、截断率、正确率、输出长度和 Student NLL 分布。
- data-side 阶段不产生 checkpoint，也不调用 `src/train_sft.py`。

## Later Full-Smoke Acceptance Criteria

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
- GPU 资源不足以可靠运行 smoke；此时只做代码和文档准备。

## Do Not

- 不要把 `vllm_dual/` 加到 `PYTHONPATH`。
- 不要跳过 `sync.sh` dry run；如果修改了 `vllm_dual/vllm/`，必须先 dry run 再同步。
- 不要默认跑训练或长 DeepScaleR 生成；先完成 worker smoke 和小 data-side smoke。
- 不要删除已有实验结果、checkpoint 或 `.sync_backups/`。
