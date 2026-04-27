# Project Documentation Index

本目录采用渐进式披露：先读本索引，再按任务读取对应文档。不要把 `docs/` 下所有文件一次性塞入上下文。

## Core

- `anti_distillation.md`：项目核心研究 idea、理论动机和长期方法设想。
- `spec/project_state.md`：当前代码、环境、实验状态的事实说明。

## Specs: Current Facts

- `spec/pre_exp.md`：response-level 预实验规格、数据格式、模块边界和 sanity checks。
- `spec/vllm_dual.md`：vLLM-dual 开发方式、同步机制、smoke 检查和回退流程。
- `spec/data_and_eval.md`：数据集、评分器、输出产物和评测口径。

## Plans: Intended Direction

- `plan/roadmap.md`：项目阶段进度和下一阶段方向。
- `plan/pre_exp_next_run.md`：DeepScaleR 预实验后续正式运行建议。
- `plan/vllm_dual_adversarial_decoding.md`：token-level adversarial decoding 的实现路线。

## Tasks: Implementation Lists

- `task/README.md`：任务卡模板和 agent 使用规则。
- `task/pre_exp_deepscaler_main.md`：DeepScaleR 主实验任务卡。
- `task/vllm_dual_token_level_decoding.md`：vLLM-dual token-level decoding 任务卡。
- `task/documentation_maintenance.md`：长任务结束后的文档维护任务卡。

## Archive

- `archive/Anti_Distillation.pdf`：历史 PDF 材料，默认不需要读取。

## Agent Reading Rules

1. 新对话先读 `AGENTS.md` 和本文件。
2. 做实验相关工作时读 `spec/pre_exp.md`、`spec/data_and_eval.md`，再读对应 task。
3. 做 vLLM-dual 工作时读 `spec/vllm_dual.md`、相关 plan 和 task。
4. 做研究方向讨论时读 `anti_distillation.md` 和 `plan/roadmap.md`。
5. 若任务卡与 spec 冲突，以 spec 为当前事实；若 spec 与代码冲突，以代码为准，并在最终回复中提醒需要更新文档。

## Codex Skills

- `$anti-distillation-workflow`：项目导航、渐进式披露和主/子 agent 协作约束。
- `$anti-distillation-task-card`：根据 plan 或用户目标创建、更新 `docs/task/*.md` 任务卡。
