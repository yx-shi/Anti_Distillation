# Response-Level Pre-Experiment Spec

本文档记录 response-level 预实验的主要规格和实现口径。早期版本以 GSM8K 为默认数据集；后续实验已转向 DeepScaleR，数据集与下一轮参数建议见 `data_and_eval.md` 和 `../plan/pre_exp_next_run.md`。

## 1. 文档目标

这份文档定义项目第一轮预实验的**最终实现规格**。目标不是直接实现最终的 token-level anti-distillation，而是先用一个更容易落地的 response-level 版本验证下面这个核心问题：

> 对同一个数学题 prompt，让 Teacher 生成 `k` 个候选回答，再选择其中 **Student 条件概率最低**（等价地，completion token 平均 NLL 最高）的回答作为蒸馏样本，是否会让 Student 的性能提升速度慢于普通蒸馏？

这里的“慢”不是模糊描述，而是指：

1. 在相同训练预算下，`teacher_adversarial` 组的 `rollout_acc` 提升慢于 `teacher_baseline` 组。
2. 这种变慢不能主要由“答案错误变多”或“response 长度异常膨胀”解释。
3. Teacher 候选本身仍保留基本任务质量，因此实验结论可以被解释为“更难蒸馏”，而不是“Teacher 输出坏掉了”。

## 2. 早期 GSM8K 预实验冻结设置

### 2.1 模型与数据

- Teacher：`Qwen3-8B`
- Student：`Qwen3-1.7B`
- 训练数据：`openai/gsm8k` train split
- 评测数据：`openai/gsm8k` test split

### 2.2 推理与环境

- Teacher 候选生成框架：`vLLM==0.8.5`
- 运行时固定设置：`VLLM_USE_V1=0`
- Student 打分与训练框架：`transformers + torch`
- 项目运行环境：`conda` 环境 `adistill-unified`

### 2.3 当前 DeepScaleR 调整

GSM8K 预实验显示任务偏简单，后续实验转向 `agentica-org/DeepScaleR-Preview-Dataset`。当前建议：

- `question_field=problem`
- `answer_field=answer`
- `num_candidates=10`
- `temperature=0.9`
- `top_p=0.9`
- `max_new_tokens=2048`
- `max_model_len=6144`
- selection policy 已切换为完整 Teacher 分布蒸馏：baseline 选 Teacher 第一条候选，adversarial 选 Student NLL 最大候选，不按 Teacher 正误或候选有效性过滤

256-sample smoke 的详细结论见 `../plan/pre_exp_next_run.md`。

### 2.4 Prompt 规范

预实验从这轮开始统一向 **Qwen3 官方 chat template** 靠拢，而不是继续扩展当前仓库里手写的 `### Question / ### Answer` 模板。

原因有三点：

1. Qwen3 官方能力和推荐用法本身就是围绕 chat template 定义的。
2. Teacher 的 `disable_thinking` 在官方用法中对应 `tokenizer.apply_chat_template(..., enable_thinking=False)`，如果不统一 prompt 体系，后续很难解释 Teacher 与 Student 的分布差异。
3. 预实验虽然只做 response-level，但后续无论继续做 token-level 还是继续扩展 SFT / rollout eval，统一 prompt helper 都会减少维护成本。

Teacher 候选生成阶段固定使用：

- `enable_thinking=False`
- 单轮 user message 形式构造 GSM8K 问题输入

## 3. 预实验定位与非目标

### 3.1 定位

本预实验是一个**可行性验证（feasibility study）**，目的不是直接产出最终 anti-distillation 方法，而是先回答：

> 在不修改 vLLM 内部 decoding 路径的前提下，仅通过 response-level 的候选重排，是否已经能观测到“让 Student 更难学”的信号？

### 3.2 本轮明确不做的事情

- 不修改 vLLM 内部 scheduler / executor / worker / model_runner
- 不实现 token-level anti-distillation decoding
- 不在本轮方案里引入额外 judge model
- 不把正式实验逻辑继续堆进 `examples/`
- 不重写新的 SFT trainer

## 4. 当前仓库内容的处理原则

### 4.1 直接复用的正式模块

- `src/sft/*`
  - 作为 Student 训练主干
  - 后续只扩展数据接口、prompt helper、checkpoint 输出，不重写 trainer 主循环
- `grading/*`
  - 继续承担 GSM8K 自动判分
  - 同时承担 Teacher 候选质量过滤
- `examples/vllm_offline/run_qwen_offline.py`
  - 作为 Teacher 批量生成脚本的原型
  - 可复用其 vLLM 初始化、SamplingParams 组织和 JSONL 落盘思路

### 4.2 保留但不纳入预实验主路径的学习脚本

- `phaseA_infer.py`
- `phaseB_debug_small.py`
- `src/phaseB_infer.py`

这些文件继续保留，原因是它们对理解 logits、decoding、SFT label masking 很有帮助；但它们不作为预实验正式工作流的入口，也不继续往上叠加新功能。

### 4.3 目录边界要求

正式预实验逻辑后续统一落在 `src/pre_exp/` 下，不在 `examples/` 下继续长正式脚本。

## 5. 预实验工作流

预实验工作流固定为四阶段，严格按“先造数据，再训练，再评测”的顺序组织。

### 阶段 A：Teacher 候选生成

输入：GSM8K train 子集中的每个问题。

过程：

1. 用 Qwen3 chat template 构造 prompt。
2. 用 vLLM 生成 `k=8` 个候选回答。
3. 对每条候选记录完整元数据并落盘。

第一版冻结的生成参数：

- `k=8`
- `temperature=0.7`
- `top_p=0.8`
- `max_new_tokens=512`
- 固定随机种子
- `enable_thinking=False`
- prompt 中追加 `Please reason step by step, and put your final answer within \boxed{}.`，以提高数学题最终答案格式的一致性

### 阶段 B：候选质量标注

对每个候选回答执行如下质量标注：

1. 标记空输出。
2. 标记是否因为长度上限截断。
3. 使用现有 `grading` 管线尝试提取最终答案。
4. 使用 gold answer 标记 `is_correct`。

这些字段只用于数据分析和实验解释，不再作为 selection filter。当前 response-level 实验的目标是让 Student 学 Teacher 的完整采样分布，而不是只学 Teacher 正确且格式良好的子分布。

只有当同一题所有候选都没有可计算 NLL 的非空 completion 时，`teacher_adversarial` 才回退到 baseline 并写入 `fallback_reason=no_scoreable_candidate`。

### 阶段 C：Student 打分

对所有非空 Teacher completion，使用 Student 基座模型计算：

- `student_mean_nll`
- `student_token_count`

这里的打分定义固定为：

> 在给定 prompt 的条件下，只对 completion 部分 token 计算平均负对数似然（mean completion-token NLL）。

不使用整段总 logprob 的原因：

1. 总 logprob 强烈受 response 长度影响。
2. 我们想衡量的是“Student 对这个完成结果有多不自然”，而不是“它有多长”。

Student 打分使用 `transformers + torch`，不使用 vLLM。原因是：

- 条件 NLL 计算本质上是训练式前向，更适合 HF/PyTorch
- 当前训练主干已经在 `src/sft/*`
- 让 vLLM 专注于高吞吐生成，工程分工更清晰

### 阶段 D：蒸馏数据构建

所有训练组共享同一个候选池和同一套质量标注，只允许**候选选择规则**不同。

本轮正式必做的训练组固定为两组：

1. `teacher_baseline`
2. `teacher_adversarial`

额外允许一个可选 sanity-check 组：

3. `gold_sft`

其中，两组蒸馏选择规则固定如下。

#### `teacher_baseline`

- 始终选择 Teacher 采样返回的第一条候选，即最小 `candidate_id`
- 不考虑该候选是否正确、是否截断、是否可抽取答案

#### `teacher_adversarial`

- 在所有可计算 NLL 的 Teacher 候选中选择 `student_mean_nll` **最高**的那个
- 不考虑该候选是否正确、是否截断、是否可抽取答案
- 如果所有候选都无法计算 NLL，则回退到 `teacher_baseline`

这样定义后，`teacher_adversarial` 和 `teacher_baseline` 的核心差异就只剩：

> 给定同一批 Teacher 原始采样候选，是否故意选 Student 最难拟合的那一个。

## 6. 数据规模与运行分层

### 6.1 Smoke Run

用途：打通全链路。

- 数据规模：GSM8K train 中固定 128 条样本
- 目标：验证候选生成、过滤、打分、数据构建、SFT 训练、评测脚本全都能无人工补丁跑通

### 6.2 Main Run

用途：形成第一轮可分析结论。

- 数据规模：GSM8K train 中固定 2000 条样本
- 抽样种子：`42`
- 评测集：GSM8K test split

本轮主实验默认不直接上 full-train，原因是：

1. response-level 预实验的核心是先确认是否存在信号。
2. 全量数据会显著拉长生成、打分和训练周期。
3. 固定中等规模子集更利于快速迭代 selection policy 和 sanity check。

full-train 可作为后续扩展，不是本轮必做项。

## 7. 代码模块边界

后续正式模块规划固定为：

```text
src/pre_exp/
  __init__.py
  teacher_generate.py
  student_score.py
  select_candidates.py
  build_distill_dataset.py
  analyze_dataset.py
  final_eval.py
```

各模块职责如下。

### `teacher_generate.py`

- 从 GSM8K 子集读取问题
- 构造 Qwen3 chat messages
- 调用 vLLM 生成 `k` 个候选
- 输出 `candidate_pool.jsonl`

### `student_score.py`

- 读取 `candidate_pool.jsonl`
- 对所有非空候选计算 completion token 平均 NLL
- 输出 `scored_candidates.jsonl`

### `select_candidates.py`

- 实现三种 selection rule
- 从 `scored_candidates.jsonl` 中选择蒸馏样本

### `build_distill_dataset.py`

- 产出 `distill_*.jsonl`
- 生成可直接喂给 `src/sft` 的训练数据

### `analyze_dataset.py`

- 做数据统计和 sanity check
- 输出候选正确率、可用率、长度分布、NLL gap 等分析结果

### `final_eval.py`

- 对训练完成的 Student checkpoint 做 GSM8K test 全量 rollout grading
- 汇总最终比较结果

## 8. 产物格式

### 8.1 `candidate_pool.jsonl`

每条记录至少包含：

- `sample_id`
- `question`
- `gold_answer`
- `messages`
- `prompt_text`
- `candidate_id`
- `candidate_text`
- `generation_config`
- `is_extractable`
- `is_correct`
- `fallback_reason`

说明：

- `messages` 保存 chat template 的结构化输入，便于后续复现 prompt 构造。
- `prompt_text` 保存最终渲染后的文本，便于人工抽查和跨框架调试。
- selection 阶段通常不写 fallback；只有 adversarial 遇到无可打分候选时才写 `no_scoreable_candidate`。

### 8.2 `scored_candidates.jsonl`

在 `candidate_pool.jsonl` 基础上新增：

- `student_mean_nll`
- `student_token_count`

### 8.3 `distill_*.jsonl`

每条记录至少包含：

- `sample_id`
- `question`
- `prompt`
- `completion`
- `selection_mode`
- `selected_candidate_id`
- `teacher_candidate_count`
- `teacher_answer_correct`
- `teacher_candidate_valid`
- `teacher_generation_truncated`
- `student_mean_nll`
- `fallback_reason`

其中：

- `prompt` 与 `completion` 是最终送入 SFT 的标准字段
- `selection_mode` 取值固定为 `teacher_baseline` / `teacher_adversarial`

## 9. 对 `src/sft/*` 的接口要求

预实验不重写 SFT trainer，但文档必须明确后续需要的接口改动。

### 9.1 数据加载

`src/sft/data.py` 需要同时支持两类输入：

1. 原始 GSM8K 数据
2. 本地 `distill_jsonl`

也就是说，SFT 层应从“只会加载一个固定 HF 数据集”，升级为“可以消费标准 prompt/completion 数据文件”。

### 9.2 配置参数

`src/sft/config.py` 需要新增本地数据接口参数，例如：

- `--dataset-format`
- `--train-file`
- `--eval-file`

其中：

- `--dataset-format` 用于区分 `gsm8k_raw` 与 `distill_jsonl`
- `--train-file` / `--eval-file` 用于指定本地 JSONL 路径

### 9.3 Prompt Helper

训练和 rollout eval 的 prompt 渲染必须统一到同一个 Qwen3 chat-template helper 上，不能继续让：

- `src/sft/data.py`
- `src/sft/rollout_eval.py`
- `src/run_grading_eval.py`

各自维护一份 `### Question / ### Answer` 模板。

统一 helper 的好处是：

1. Teacher 生成、Student 训练、rollout eval 的 prompt 分布更一致。
2. Qwen3 的 `enable_thinking=False` 语义可以被干净地接入。
3. 后续从 response-level 升级到 token-level 时，prompt 不是额外干扰变量。

### 9.4 Checkpoint 输出

训练流程需要增加 `output_dir` 概念，并在训练结束时保存最终 checkpoint。

原因：

1. 当前 `src/sft/trainer.py` 会打印训练和评测日志，但没有显式 checkpoint 输出接口。
2. 没有稳定的输出目录，后续 `final_eval.py` 无法可靠消费不同实验组的最终模型。

## 10. 输出目录布局

后续预实验产物统一放在：

```text
result/pre_exp/
  candidates/
  datasets/
  runs/
  analysis/
```

推荐约定如下：

- `result/pre_exp/candidates/`
  - 候选池与打分文件
- `result/pre_exp/datasets/`
  - 两组蒸馏 JSONL
- `result/pre_exp/runs/`
  - 训练日志与 checkpoint
- `result/pre_exp/analysis/`
  - 数据统计、曲线汇总、最终比较结果

## 11. 训练与评测协议

### 11.1 训练预算

两组正式实验必须使用完全一致的 Student 初始化和超参数。

冻结的训练预算：

- 相同 Student 初始权重
- 相同 optimizer 与学习率设置
- `max_steps=1000`
- `eval_every=100`
- 训练中 `rollout_eval_max_samples=64`

这里最重要的原则是：

> 除了蒸馏数据的 selection rule，不再引入新的自变量。

### 11.2 训练中评测

继续复用现有 `src/sft/rollout_eval.py` 的思路，但要在后续实现里迁移到统一的 Qwen3 prompt helper。

训练中至少记录：

- `train_loss`
- `val_loss`
- `rollout_acc`

### 11.3 训练后评测

每个实验组训练结束后，必须在 GSM8K test split 上执行一次**全量** rollout grading，并比较：

- `rollout_acc`
- `val_loss`
- 达到某个准确率阈值所需 step 数的趋势

这一步由后续 `src/pre_exp/final_eval.py` 负责。

## 12. 必做的数据 sanity checks

在开始训练前，至少要跑完以下统计：

1. 候选正确率
2. 候选有效率、截断率、正确率
3. 两组选中 response 的长度分布
4. `teacher_baseline` 与 `teacher_adversarial` 之间的平均 `student_mean_nll` 差值

这些统计是必须的，因为如果最终只看到训练曲线差异，却不知道数据层面发生了什么，就很难判断实验是否真的支持 anti-distillation 假设。

## 13. 可接受结果与失败信号

### 13.1 验收标准

本轮预实验达到以下条件，可认为方案成立并值得继续扩展：

1. smoke run 可以端到端跑通，不需要人工插入临时补丁。
2. main run 中，adversarial 数据没有导致候选正确率明显塌陷。
3. `teacher_adversarial` 组在相同训练预算下，`rollout_acc` 提升明显慢于 `teacher_baseline` 组。
4. 这种差异不能主要由 response 崩坏或长度极端膨胀解释。

### 13.2 失败信号

如果出现以下情况，应优先回到数据构建层定位问题，而不是直接否定研究方向：

#### 失败信号 1：Teacher 候选正确率过低

说明：

- prompt 或采样设置不稳定
- 候选池本身没有形成“高质量 + 有差异”的候选集合

优先动作：

- 先调整 Teacher 生成参数，不进入下一层 selection 讨论

#### 失败信号 2：adversarial 样本主要由噪声 response 构成

说明：

- 完整 Teacher 分布蒸馏口径下，selection 不按正误过滤
- Student NLL 排序压过了任务质量约束

优先动作：

- 同时报告 selected-candidate correctness / truncation，必要时调整 Teacher 生成参数或单独设计质量约束实验

#### 失败信号 3：baseline 与 adversarial 几乎没有差异

可能原因：

- `k` 太小
- Teacher 候选多样性不足
- Student NLL 没有有效区分“难学程度”
- GSM8K 对 response-level 干预不够敏感

这时才考虑：

- 增大 `k`
- 调整采样参数
- 或进入更细粒度的 token-level 方法

## 14. 本轮方案与后续正式 anti-distillation 的关系

这轮做的是 response-level 版本，但它不是一次性脚手架。后续升级到 token-level 时，下列资产都可以继续保留：

- `grading` 作为质量过滤与任务评测器
- `src/sft/*` 作为 Student 训练主干
- `candidate -> score -> select -> train -> eval` 这条数据闭环
- 统一的 Qwen3 prompt helper
- 按实验组组织的结果目录和比较流程

真正需要变化的主要是：

- 候选从“整段 response 的重排”升级为“每一步 token 的候选决策”

## 15. 实施顺序

后续实现必须按下面顺序推进：

1. 实现统一的 Qwen3 prompt helper，并设计 `candidate_pool.jsonl` / `distill_*.jsonl` schema
2. 用 vLLM 在 smoke 子集上生成候选并人工抽查
3. 接入 `grading` 完成 boxed-answer extractable + correctness 标注
4. 实现 Student completion-token mean NLL 打分
5. 产出两组蒸馏 JSONL
6. 扩展 `src/sft` 数据接口，使其能读取 `distill_jsonl`
7. 在 smoke run 上打通训练与评测
8. 运行 2000 样本 main run，汇总曲线和最终结果

这个顺序的核心原则是：

> 先把数据造对，再让训练消费数据；不要把 teacher 生成、candidate 选择、SFT 训练和最终评测混成一个大脚本。

## 16. 一句话结论

本预实验的正式目标是：

> 在统一的 Qwen3 prompt 体系下，基于同一个 Teacher 候选池，比较 baseline / adversarial 两种 response 选择规则对 Student 学习曲线的影响，并判断 response-level anti-distillation 是否已经能提供可观测的“蒸馏变难”信号。
