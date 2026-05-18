# Project State

本文档记录当前项目事实，供 agent 快速建立上下文。它不是计划，也不是任务清单；发生事实变化时再更新。

## Research Focus

项目目标是研究 anti-distillation：在 Teacher 输出质量基本不下降的前提下，让小模型更难通过蒸馏学习 Teacher 的能力。

当前采用两条路线：

- response-level 预实验：Teacher 对同一题生成多个候选；baseline 选第一条候选，adversarial 先匹配 baseline 所选候选的正确性，再在同正确性候选中选择 Student 平均 NLL 最高的可打分候选。
- token-level adversarial decoding：修改 vLLM，使 Teacher 和 Student 在同一 prefix 上共同 forward，并在 token 级别选择对 Student 更难的 next token；hard/soft 首版已在 vLLM-dual 路径实现并通过 smoke。

## Current Code Layout

- `src/pre_exp/`：预实验数据侧、候选打分、候选选择、数据分析和最终评测脚本。
- `src/vllm_dual_decoding/`：vLLM-dual token-level 独立链路，包含生成、打分、分析、SFT-ready 数据构建和 full pipeline 串联入口。
- `src/experiment/`：token-level 实验配置、dataset registry、run_id、路径和 launcher 逻辑。
- `src/run_experiment.py`：token-level 新主入口，读取 `configs/*.yaml` 后按 stage 调用现有生成、打分、训练和 eval 脚本。
- `configs/`：轻量实验 YAML；当前包含 `deepscaler.yaml` 与 `gsm8k.yaml`。
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
- 已归档 vLLM-dual token-level decoding 实现任务；下一步活跃任务是建设与 `src/pre_exp/` 并列的独立 token-level 链路。当前先准备并运行 data-side smoke，数据质量达标后再扩展到完整 SFT/eval smoke。

## Current Experimental State

- GSM8K 预实验显示任务偏简单，Student base 正确率较高，蒸馏提升和 adversarial 差异不明显。
- DeepScaleR response-level main8000 预实验已完成：8000 samples、`k=8`、`temperature=0.9`、`top_p=0.85`、`max_new_tokens=4096`、`max_model_len/score_max_length=8192`。
- main8000 训练使用 `TRAIN_MAX_LENGTH=5120`、8 卡 FSDP、1000 step；rollout eval 使用 DeepScaleR holdout，排除 seed-42 main8000 训练子集。final 4096-sample holdout acc：`teacher_baseline` 37.43%，`teacher_adversarial` 36.47%。结果摘要见 `result/pre_exp/analysis/deepscaler_main8000_k8_t0.9_p0.85_len4096/run_summary.md`。
- 2026-05-13 已复用 main8000 scored candidates 跑通 correctness-matched 数据侧，不启动训练。新产物目录：`result/pre_exp/datasets/deepscaler_main8000_k8_t0.9_p0.85_len4096_correctness_matched` 和 `result/pre_exp/analysis/deepscaler_main8000_k8_t0.9_p0.85_len4096_correctness_matched`。selected correctness 已对齐：baseline/adversarial 均为 54.85%，逐样本 correctness mismatch 为 0，平均 Student NLL gap 为 +0.06155。
- 2026-05-13 correctness-matched 两组 SFT 训练、checkpoint eval、final eval 和曲线绘制已完成。训练输出目录：`/home/disk2/shiyixuan/pre_exp_runs/deepscaler_main8000_k8_t0.9_p0.85_len4096_correctness_matched`。结果摘要见 `result/pre_exp/analysis/deepscaler_main8000_k8_t0.9_p0.85_len4096_correctness_matched/run_summary.md`。训练内 val_loss/ppl：baseline 0.2591 / 1.2958，adversarial 0.2495 / 1.2834。final 4096-sample holdout acc：baseline 35.99%，adversarial 37.55%，adversarial 高 1.56 个百分点。
- 2026-05-13 token-level vLLM-dual preflight 已完成：`sync.sh` dry run 后正式同步到 conda 环境，备份目录 `.sync_backups/vllm_20260513_165733`；`config.py`、`engine/arg_utils.py`、`worker/dual_worker.py` 与 conda site-packages 逐字节一致；hard/soft worker smoke 均出现 `DualModelWorker` 和 `ADISTILL_DUAL_ADVERSARIAL enabled` marker。
- 2026-05-13 `scripts/run_vllm_dual_data_smoke.sh` 已从小样本 smoke 扩展为 DeepScaleR 10000 条 token-level data build 入口：三种模式顺序运行，每种模式内部按 8 个 TP=1 shard 并发生成和打分，默认产物写到 `/home/disk2/shiyixuan/Anti_Distillation/result/vllm_dual_decoding`。已用 DeepScaleR 1 条、16 token micro chain 验证 `teacher_plain`、`teacher_token_hard`、`teacher_token_soft` 均能生成、Student NLL 打分并汇总。
- 2026-05-14 token-level DeepScaleR 8000 条三组数据已转成 SFT-ready JSONL：`result/vllm_dual_decoding/datasets/vllm_dual_deepscaler8000_k1_t0.7_p0.8_len4096/distill_teacher_plain.jsonl`、`distill_teacher_token_hard.jsonl`、`distill_teacher_token_soft.jsonl`，每组 8000 条；固定 holdout eval JSONL `deepscaler_holdout_eval_1024_seed42.jsonl` 已构建，排除同 seed-42 的 8000 训练子集。当前尚未启动长训练或 GPU rollout eval。
- 2026-05-14 token-level DeepScaleR 8000 三组 SFT 训练已完成，不含 rollout eval：`teacher_plain`、`teacher_token_hard`、`teacher_token_soft` 均训练 1600 step，每 200 step 记录 validation loss/ppl，并产出 final checkpoint。训练输出目录：`/home/disk2/shiyixuan/Anti_Distillation/result/vllm_dual_decoding/runs/vllm_dual_deepscaler8000_k1_t0.7_p0.8_len4096`。最终 val loss/ppl：plain 0.2909 / 1.3377，hard 0.2593 / 1.2960，soft 0.2570 / 1.2931。曲线产物已写入 `result/vllm_dual_decoding/analysis/vllm_dual_deepscaler8000_k1_t0.7_p0.8_len4096/curves/`，包含 `train_loss_curve.svg`、`val_loss_ppl_curve.svg` 和 `curve_data.json`。checkpoint/final rollout eval 尚未运行。
- 2026-05-16 已新增配置驱动 token-level 主入口：`src/run_experiment.py --config configs/deepscaler.yaml|configs/gsm8k.yaml --stage ...`。新链路通过 `src/experiment/dataset_registry.py` 管理 DeepScaleR/GSM8K 字段和答案抽取策略，统一使用 run_id 输出到 `result/vllm_dual_decoding/{candidates,datasets,analysis}/{run_id}`，训练 checkpoint 仍写入 `/home/disk2/shiyixuan/Anti_Distillation/result/vllm_dual_decoding/runs/{run_id}`。旧 `scripts/dual_decoding/*.sh` 已迁入 `scripts/archive/dual_decoding/`。
- 2026-05-16 `src/vllm_dual_decoding/` 下旧专项脚本 `analyze_generation_modes.py`、`case_study.py`、`run_full_pipeline.py`、`run_parallel_rollout_eval.py` 已删除；新主链路只保留生成、打分、distill 构建和共享 helper。
- 2026-05-17 token-level 主入口已接入 `plot`/`curves` stage，复用 `src/pre_exp/plot_curves.py` 汇总 run_id 对应的 train/eval/final rollout 曲线。`smoke_gpu` 下 GSM8K 与 DeepScaleR 的 `teacher_plain` 8-sample eval 已用新入口复验，并在 `result/vllm_dual_decoding/analysis/{run_id}/curves/` 产出 `curve_data.json`、`train_loss_curve.svg`、`val_loss_ppl_curve.svg` 和 `rollout_accuracy_curve.svg`。
- 2026-05-17 token-level 主入口新增 `rollout_eval`/`checkpoint_eval` stage：按 `rollout_eval.gpu_ids` 并行评测 base Student step 0、训练中间 checkpoint 和 final checkpoint，输出 `checkpoint_eval_*.json` 与 `checkpoint_eval.json`。通用离线 rollout 入口迁到 `src/evaluation/rollout_eval.py`，`src/pre_exp/final_eval.py` 保留兼容 wrapper。已用 GSM8K teacher_plain 2-sample smoke 跑通 base/final 两个 checkpoint，并确认 curves 阶段优先使用 checkpoint eval 文件而不是大样本 `final_eval.json`。
- 2026-05-18 token-level 主入口新增 group-level `final_summary`/`result_summary` stage：读取同一 group 下多个 mode 的 data quality、train log、checkpoint rollout eval 与 full final eval，统一写入 `result/summary/{group_run_id}/final_result_summary.{json,md}`，并在同目录生成三张跨 mode SVG：`train_loss_curve.svg`、`val_loss_ppl_curve.svg`、`rollout_accuracy_curve.svg`。GSM8K main 三组完整结果已汇总到 `result/summary/Gsm8k-main__ds-gsm8k__t-0.7__p-0.8__n-7000__seed-42/`；full final eval acc：plain 83.74%，hard 83.02%，soft 83.38%。

## Maintenance Rule

如果代码或实验事实变化，优先更新本文件或对应 `docs/spec/*`。不要把临时日志、长命令输出或一次性分析堆进 `AGENTS.md`。
