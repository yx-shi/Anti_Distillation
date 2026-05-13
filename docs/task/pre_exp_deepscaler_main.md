# Task: DeepScaleR Main Pre-Experiment

## Goal

基于 DeepScaleR smoke 结果，执行当前 response-level DeepScaleR 主实验流水线，并产出可比较的训练、评测和曲线结果。

## Required Reading

- `AGENTS.md`
- `docs/README.md`
- `docs/spec/pre_exp.md`
- `docs/spec/data_and_eval.md`
- `docs/plan/pre_exp_next_run.md`

## Context

GSM8K 难度偏低，Student base 正确率较高，adversarial 差异不明显。DeepScaleR smoke 显示数据更难；`deepscaler_smoke128_k8_t0.9_p0.85_len4096` 基本解决了早期 `max_new_tokens=1024` 的严重截断问题。

当前 main 设置固定为 8000 samples、`k=8`、`temperature=0.9`、`top_p=0.85`、`max_new_tokens=4096`、`max_model_len=8192`、`score_max_length=8192`。

2026-05-10 已完成的 main run 使用旧完整 Teacher 分布 selection policy：baseline 选 Teacher 第一条候选；adversarial 在可计算 NLL 的候选中选 `student_mean_nll` 最大者；不按正确性、截断状态或 extractable/valid 字段过滤。后续当前代码已改为 correctness-matched adversarial selection：adversarial 先匹配 baseline 所选候选的 `is_correct`，再在同正确性候选中选 `student_mean_nll` 最大者。质量字段必须保留到 selection、distill dataset 和 summary 中，用于解释训练曲线差异。

## Implementation List

- [completed] 2026-05-10 main run 当时检查并修正 selection policy，使 baseline/adversarial 学旧完整 Teacher 分布，而不是只学正确候选子分布；该口径已被后续 correctness-matched policy 替代。
- [completed] 使用 TP=1 多 replica Teacher generation 跑 DeepScaleR main 候选生成，避免单个 vLLM tensor-parallel run 绑死所有 GPU；每个 replica 写入互不覆盖的 shard。
- [completed] 用 main 设置完成候选打分、baseline/adversarial selection 和 distill dataset 构建。
- [completed] 增强 dataset summary，报告 candidate-level 与 sample-level 的 empty/extractable/valid/correct/truncated、selected-candidate quality、长度分布、NLL gap、fallback rate 和 baseline/adversarial 选中差异率。
- [completed] 训练前确认 baseline/adversarial 数据长度、正确率和截断率差异可解释；若数据质量明显异常，停在数据侧并记录原因。
- [completed] 数据质量通过后，分别训练 `teacher_baseline` 和 `teacher_adversarial`，保持相同 Student 初始化和训练超参数。
- [completed] 对训练 checkpoint 执行 checkpoint eval，并对最终 checkpoint 执行 final eval。
- [completed] 运行 plot curves，产出 baseline/adversarial 的训练曲线、checkpoint eval 曲线、final eval 对比图及底层数据。

## Result Note

2026-05-10 已跑完 DeepScaleR main8000 response-level 预实验剩余链路。产物目录：

- data/analysis：`result/pre_exp/analysis/deepscaler_main8000_k8_t0.9_p0.85_len4096`
- training：`/home/disk2/shiyixuan/pre_exp_runs/deepscaler_main8000_k8_t0.9_p0.85_len4096`
- summary：`result/pre_exp/analysis/deepscaler_main8000_k8_t0.9_p0.85_len4096/run_summary.md`

关键结果：

- 数据侧 sanity 通过：candidate 截断率 7.24%，score 截断率 0%，baseline/adversarial 选中不同候选 87.71%，平均 Student NLL gap +0.0714。
- 训练口径调整为 `TRAIN_MAX_LENGTH=5120`；checkpoint eval 使用 DeepScaleR holdout 1024 条，final eval 使用 DeepScaleR holdout 4096 条，均排除 seed-42 main8000 训练子集。
- final rollout acc：`teacher_baseline` 37.43%，`teacher_adversarial` 36.47%，adversarial 低 0.95 个百分点。

2026-05-13 复用同一份 main8000 `scored_candidates.jsonl` 跑通 correctness-matched 数据侧，未启动训练。新产物目录：

- datasets：`result/pre_exp/datasets/deepscaler_main8000_k8_t0.9_p0.85_len4096_correctness_matched`
- analysis：`result/pre_exp/analysis/deepscaler_main8000_k8_t0.9_p0.85_len4096_correctness_matched`

数据侧 sanity：baseline/adversarial selected correct rate 均为 54.85%，逐样本 correctness mismatch 为 0，adversarial fallback rate 为 0，baseline/adversarial 选中不同候选 81.69%，平均 Student NLL gap +0.06155。

同日已用上述两份 correctness-matched distill 数据完成 SFT 训练、checkpoint eval、final eval 和曲线绘制。训练输出目录：

- `/home/disk2/shiyixuan/pre_exp_runs/deepscaler_main8000_k8_t0.9_p0.85_len4096_correctness_matched/teacher_baseline`
- `/home/disk2/shiyixuan/pre_exp_runs/deepscaler_main8000_k8_t0.9_p0.85_len4096_correctness_matched/teacher_adversarial`

训练参数沿用旧 main8000：8 卡 FSDP、1000 step、`TRAIN_MAX_LENGTH=5120`、per-device batch 2、lr `5e-5`、checkpoint every 200。训练内 final val_loss/ppl：baseline 0.2591 / 1.2958；adversarial 0.2495 / 1.2834。

rollout eval 结果：checkpoint eval 使用 1024 条 holdout，final eval 使用 4096 条 holdout。final rollout acc：baseline 35.99%（1474/4096），adversarial 37.55%（1538/4096），adversarial 高 1.56 个百分点。paired final eval 中 baseline-only correct 363，adversarial-only correct 427，adversarial 净多 64 题。结果摘要见 `result/pre_exp/analysis/deepscaler_main8000_k8_t0.9_p0.85_len4096_correctness_matched/run_summary.md`。

## Acceptance Criteria

- Teacher generation 使用 TP=1 multi-replica/sharded 方式完成，shard 合并后样本数、候选数和配置与 main 设置一致。
- 数据侧产物包含 candidate pool、scored candidates、selection、distill jsonl 和 expanded dataset summary。
- summary 明确显示截断率、valid/correct sample rate、selected-candidate quality、长度分布、fallback rate、选中差异率和 NLL gap。
- distill 数据保留正确性、valid、truncation、NLL、token count 和 fallback 等质量字段，且 selection 不使用这些字段过滤。
- 训练产物包含 baseline/adversarial checkpoint、checkpoint eval 结果、final eval 结果和 plot curves artifacts。
- 如果数据质量不达标，停止在数据侧分析，不启动训练，并在 summary 或 run note 中说明原因。

## Do Not

- 不要在数据质量不明时直接跑 SFT。
- 不要运行未分片、会独占全部 GPU 的 Teacher main generation。
- 不要覆盖历史 smoke/main 结果目录。
- 不要删除 `/home/disk2/shiyixuan` 下已有 checkpoint。
