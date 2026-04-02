(Credit by GPT5.2)

## 一、背景介绍

### 1.1 大模型蒸馏问题

知识蒸馏（Knowledge Distillation）是当前模型压缩与能力迁移的重要方法。通常流程为：

* 大模型（Teacher）生成高质量输出或logits

* 小模型（Student）拟合Teacher的输出分布

* 通过KL散度最小化：

$$\mathcal{L}_{KD} = KL(p_T || p_S)$$

其中：

* $$p_T$$ 为Teacher输出分布

* $$p_S$$ 为Student输出分布

然而，在商业和安全场景下存在问题：

* API输出被收集后可用于训练小模型

* 黑盒蒸馏（Black-box Distillation）可以通过采样数据逼近Teacher

* 模型能力被“复制”

因此产生需求：

> 是否可以在不明显降低输出质量的前提下，使得生成数据对小模型更难学习？

## 二、核心思想

### 2.1 Anti-Distillation 目标

我们希望构造一种解码策略，使得：

* 输出仍然语义合理

* 但在概率分布层面**对小模型具有“误导性”**

* 提高蒸馏的KL距离或训练难度

### 2.2 方法概述：Cross-Model Adversarial Decoding (CMAD)

在解码时：

1. Teacher生成Top-k候选token集合：

   $$\mathcal{T}_k = \{t_1, t_2, ..., t_k\}$$

2. 对每个候选token，查询Student的概率：

   $$p_S(t_i)$$

3) 在Teacher高概率集合中，选择：

$$t^* = \arg\min_{t_i \in \mathcal{T}_k} p_S(t_i)$$

即：

> 在Teacher认为“差不多好”的词中，优先选Student最不自信的词。

### 2.3 改进版本

#### (1) 加权版本

$$t^* = \arg\max_{t_i} \left( \log p_T(t_i) - \lambda \log p_S(t_i) \right)$$

其中：

* $$\lambda$$ 控制对抗强度

#### (2) 分布重加权版本

构造新的采样分布：

$$\tilde{p}(t_i) \propto p_T(t_i) \cdot (1 - p_S(t_i))^\alpha$$

优点：

* 可平滑控制扰动

* 不必总选极端token

## 三、理论动机

### 3.1 提高蒸馏难度

蒸馏本质是最小化：

$$KL(p_T || p_S)$$

我们的策略使得：

* 采样分布偏向Student低概率区域

* 增大初始KL距离

* 增强学习难度

### 3.2 直觉解释

小模型难以学习：

* 低频但合法的表达

* 边缘语义替代表达

* 长尾token模式

我们主动偏向这些区域，使其：

* 语义保持

* 统计模式变得“非自然”

## 四、实验设计

## 4.1 实验目标

验证以下假设：

1. ✅ 不显著降低大模型输出质量

2. ✅ 提高小模型蒸馏误差

3) ✅ 增加Student收敛时间

4) ✅ 提高蒸馏后模型与Teacher差距

## 4.2 实验设置

### 模型选择

| 角色      | 模型       |
| ------- | -------- |
| Teacher | 7B/13B模型 |
| Student | 1B/3B模型  |

### 数据集

* Instruction tuning 数据

* QA数据 ()

* 推理任务数据 (DeepscaleR)

### 对比方法

1. 普通Top-k采样

2. Temperature sampling

3) 本文 Anti-Distill 解码

## 4.3 评估指标

### 1. 输出质量评估

* Perplexity

* GPT-based evaluation

* 人工评分

### 2. 蒸馏效果评估

* Student验证集loss

* Student BLEU/ROUGE

* Student与Teacher输出KL距离

* 收敛epoch数

### 3. 分布分析

* Token entropy变化

* 长尾token比例

* 词频分布偏移

## 五、预期结果

### 5.1 输出质量

预期：

* 轻微下降（<5%）

* 语义基本保持

### 5.2 蒸馏难度

预期：

* Student loss上升 10–25%

* 收敛时间增加 20–40%

* Student生成质量显著下降

### 5.3 分布现象

预期：

* token多样性增加

* 更长尾

* Student难以学习分布形状

## 六、可能风险

| 风险             | 说明    |
| -------------- | ----- |
| 输出质量显著下降       | λ过大   |
| 语义偏移           | 过度对抗  |
| Student适应后仍能收敛 | 需动态策略 |

## 七、可扩展方向

* 动态 λ 调节

* 多Student联合对抗

* 在logits层加入对抗扰动

* 与Watermarking结合

# 八、进度安排

## 第一阶段：理论与实现

* ✅ 实现Cross-model decoding框架

* ✅ 实现logits reweight版本

* ✅ 完成小规模实验验证

## 第二阶段：系统实验

* ✅ 大规模蒸馏实验

* ✅ 不同λ对比

* ✅ 收敛速度分析

* ✅ 分布统计分析

## 第三阶段：稳定性与泛化

* ✅ 不同任务验证

* ✅ 多Student测试

* ✅ 对比不同规模Teacher

## 第四阶段：论文整理

* ✅ 理论动机强化

* ✅ 可视化

* ✅ 撰写论文

# 九、项目贡献点

1. 提出一种推理阶段可部署的anti-distillation机制

2. 不需要修改模型参数

3) 不影响训练

4) 可与watermark技术兼容

5. 可形成商业API保护方案

# 进度

## 20260302-20260309

* 完成服务器环境配置

* 初步熟悉HF的Transformer框架

* 完成一个HF的推理demo，并手写`generate`循环观察生成

## 20260309-20260316

* 用HF的tokenizer和torch手写完整SFT训练

  * 模型用Qwen3-1.7B

  * dataset为Dolly-15k

* 熟悉torch训练pipeline

## 20260316-20260323

* SFT跑通，loss稳定下降

  * 解决了之前由于部分数据prompt过长，label全被mask导致loss nan问题

* 熟悉HF的tokenizer细节

* 修改SFT

  * 数据集换为gsm8k

  * FSDP部署到多卡跑

## 20260323-20260330

* TODO

  * [x] FSDP跑通

    * [x] 2-4卡确保都能运行（2卡正常运行）

  * [x] 接GSM 8k和MATH数据集跑通（MATH下架了）

    * [ ] 自行处理数据混合和读取

  - Vllm vanilla inference跑通

    * vllm用0.8.5版本，跑的时候指定环境变量VLLM\_USE\_V1=0强制使用v0版本跑。因为是旧版本，参阅文档的时候注意版本号https://docs.vllm.ai/en/v0.8.5/

    * 用offline Inference版本（`LLM`对象）

    * 一般需要装flash-attention，这个包一般是直接通过whl文件来装的，可以根据系统选择包的版本https://github.com/Dao-AILab/flash-attention/releases

  - 尝试魔改（代码`@李旭浚`提供）

## 20260330-20260406

* TODO

  * [ ] 跑通vanilla vllm

    * [ ] 看到显著的速度提升，监控GPU利用率和功率（watch nvidia-smi）

    * [ ] 了解基本部件，不需要知道完整实现

      * [ ] 和vanilla generate的较大区别

        * [ ] Batch inference

        * [ ] Prefill & Decoding

      * [ ] 需要看：llm\_engine, executor, worker, model\_runner的关键函数，大致了解model\_input的构建，executor\_req的格式

      * [ ] 不需要看：scheduler, page\_attention, swap

  * [ ] 读懂+跑通魔改vllm

    * [ ] 了解魔改链路

  * [ ] 进一步修改，添加Adversarial Decoding的逻辑，跑通

    * [ ] 分别实现hard和soft的机制，通过参数控制

  * [ ] 在GSM prompt上用teacher（协同student）打一批adversarial data，蒸馏student

    * [ ] Teacher Qwen3 8B

      * [ ] https://modelscope.cn/models/Qwen/Qwen3-8B，初期先disable\_thinking

    * [ ] Student Qwen3 1.7B
