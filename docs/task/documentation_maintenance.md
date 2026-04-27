# Task: Documentation Maintenance

## Goal

在长任务结束后，让项目文档反映新的事实、计划和任务状态，保证新对话可以无缝接续。

## Required Reading

- `AGENTS.md`
- `docs/README.md`
- 与本次任务相关的 spec、plan、task。

## Context

本项目采用渐进式披露文档结构。入口文档应保持短小，具体事实和任务状态放在对应目录。

## Implementation List

- 如果代码行为或实验事实变化，更新对应 `docs/spec/*`。
- 如果下一步路线或参数建议变化，更新对应 `docs/plan/*`。
- 如果某个 implementation task 完成或失效，更新对应 `docs/task/*`。
- 检查 `docs/README.md` 的索引是否需要新增或调整。

## Acceptance Criteria

- 新 agent 只读 `AGENTS.md` 和 `docs/README.md` 能找到下一步该读的文档。
- 没有旧路径引用。
- 文档更新不包含大段临时日志。

## Do Not

- 不要把实验日志全文粘进文档。
- 不要把 task 细节塞进 `AGENTS.md`。
