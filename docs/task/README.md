# Task Cards

本目录放可交给 agent 执行的 implementation list。task 卡应足够具体，让子 agent 不需要重新猜目标和边界。

## Template

```markdown
# Task: <name>

## Goal

本任务要完成什么。

## Required Reading

- `AGENTS.md`
- `docs/README.md`
- 其他必要 spec/plan，最多 3-5 个。

## Context

当前状态、已知约束、已有结论。

## Implementation List

- 明确步骤。
- 涉及文件或模块。
- 需要保留的行为。

## Acceptance Criteria

- 如何判断完成。
- 需要跑哪些检查。

## Do Not

- 禁止事项，例如不要跑训练、不要同步 vLLM、不要删除结果。
```

## Agent Rule

主 agent 可以读多个 task 做调度；子 agent 只读被分配 task 的 Required Reading，不默认全量读取 `docs/`。
