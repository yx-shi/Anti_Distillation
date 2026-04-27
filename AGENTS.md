# Anti Distillation Agent Guide

本文件是 coding agent 进入项目时的最小入口。只放长期稳定的开发规范和文档索引；具体实验规格、计划和任务卡请按需读取 `docs/README.md` 中列出的文件。

## Project Goal

Anti Distillation 的目标是在尽量不降低 Teacher 输出质量的前提下，提高小模型蒸馏 Teacher 的难度。当前核心方向是让 Teacher 在高质量候选中偏向 Student 低概率区域，从而增大蒸馏学习难度。

## Read First

- 项目文档入口：`docs/README.md`
- 核心研究 idea：`docs/anti_distillation.md`
- 当前项目状态：`docs/spec/project_state.md`
- 预实验规格：`docs/spec/pre_exp.md`
- vLLM-dual 开发规范：`docs/spec/vllm_dual.md`
- 可执行任务卡：`docs/task/README.md`

Agent 工作时不要默认全量读取所有文档。先读 `docs/README.md`，再根据任务类型读取相关 spec、plan、task，实现渐进式披露。

## Agent Workflow

- 主 agent 负责理解目标、选择必要文档、拆分任务、调度和最终整合。
- 子 agent 只读取任务卡中列出的必要上下文，完成具体 implementation task。
- 长任务开始前先确认 task 卡中的目标、输入、涉及文件、验收标准和禁止事项。
- 长任务结束后，如项目事实发生变化，更新对应 `docs/spec/*`；如路线变化，更新 `docs/plan/*`；如任务完成，更新 `docs/task/*`，确保后续开发通过读取相关文档能够精准理解项目进度

## Engineering Rules

- 这是课题组服务器，避免危险操作；不要运行破坏性命令，不要删除实验结果或 checkpoint。
- 代码修改要遵循已有工程结构，优先复用本仓库已有 helper、脚本和配置。
- 复杂实现要保留必要中文注释，说明库函数用途、关键参数含义和常用工程范式。
- 给用户解释复杂概念时，按“直观解释 → 原理分析 → 技术细节”的顺序组织。
- 遇到 Hugging Face 细节问题时，使用 Hugging Face 官方文档、插件或 skills 查证。
- 回复中给出 bash 命令时，如果包含不常见参数，要解释参数含义。

## Environment Constraints

- Conda 环境：`adistill-unified`
- vLLM 版本：`0.8.5`
- vLLM 运行口径：设置 `VLLM_USE_V1=0`，强制使用 V0 engine。
- SFT Student 默认模型：`/home/disk1/public_checkpoint/Qwen3-1.7B`
- Teacher 默认模型：`/home/disk1/public_checkpoint/Qwen3-8B`
- 训练 checkpoint 和大体量运行产物优先放在 `/home/disk2/shiyixuan`，避免写满仓库磁盘。

## Repository Map

- `src/pre_exp/`：response-level 预实验数据构建、候选选择、评测相关代码。
- `src/sft/`：基于 Transformers + PyTorch/FSDP 的 SFT 训练框架。
- `grading/`：数学题答案抽取与判分逻辑。
- `examples/`：学习和 demo 用例，不作为正式实验主路径。
- `docs/`：项目规格、计划、任务卡和归档文档。
