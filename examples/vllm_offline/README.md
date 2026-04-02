# vLLM Offline Demo

这个目录专门放和 `vLLM` 原生推理相关的最小 demo，不放进 `src/sft/` 的原因很简单：

- `src/sft/` 现在主要承担训练入口和训练模块
- vLLM offline inference 属于一条独立 workflow
- 后面如果你继续补 online serving、benchmark、批量评测，也可以自然沿着这个目录扩展

## 1. 先搞清楚安装边界

根据 vLLM `v0.8.5` 官方 GPU 安装文档：

- 运行环境要求是 Linux
- Python 版本要求是 3.9 到 3.12
- NVIDIA GPU 需要 compute capability 7.0 或更高
- 预编译 wheel 默认按 CUDA 12.4 和公开 PyTorch release 版本构建
- 官方明确建议：`fresh new environment`

为什么这里我要特别强调“新环境”：

- 你当前的 `adistill` 环境里，`Python 3.10.19` 是满足要求的
- 但我实际检查到里面的 `torch` 是 `2.10.0+cu128`
- 官方文档明确说：如果你想复用一个已经装了不同 CUDA / PyTorch 组合的环境，通常需要从源码构建 vLLM

所以对于“先把 offline demo 稳定跑通”这个目标，最靠谱的方案是：

- 保留现在的 `adistill` 给训练
- 新建一个专门跑 vLLM 0.8.5 的环境，比如 `adistill-vllm085`

## 2. 推荐安装方案

### 2.1 预检查

先检查 Python 版本：

```bash
conda run -n adistill python --version
```

这里的 `-n` 表示“指定 conda 环境名”。

再检查 GPU/驱动是否可见：

```bash
nvidia-smi
```

如果这里都看不到 GPU，vLLM 基本不可能正常跑起来。

### 2.2 创建一个干净环境

```bash
conda create -n adistill-vllm085 python=3.10 -y
```

参数解释：

- `-n`：指定新环境名字
- `-y`：默认回答 yes，避免安装过程中反复确认

然后激活环境：

```bash
conda activate adistill-vllm085
```

### 2.3 安装 vLLM 0.8.5

```bash
pip install "vllm==0.8.5"
```

这里我没有让你额外手动装 `torch`，因为官方推荐路径就是直接 `pip install vllm`，让 pip 按 vLLM 的 wheel 和依赖关系解决。

如果安装后发现 `import vllm` 在启动阶段报错，而环境里被解析成了 `transformers 5.x`，建议把
`transformers` 明确 pin 回 4.x 的已知可用版本。当前项目里一个经过实际参考的组合是：

- `torch==2.6.0`
- `xformers==0.0.29.post2`
- `transformers==4.52.4`
- `vllm==0.8.5`

也就是说，如果你发现 pip 给你装成了 `transformers 5.x`，可以直接执行：

```bash
python -m pip install --force-reinstall "transformers==4.52.4"
```

参数解释：

- `python -m pip`：确保调用的是当前环境自己的 `pip`
- `--force-reinstall`：即使已经装了其他版本，也强制重装成你指定的版本
- `"transformers==4.52.4"`：用 `==` 精确锁定版本，避免再次被升级到 5.x

### 2.4 验证安装

```bash
python -c "import sys, vllm; print(vllm.__version__); print(sys.version)"
```

如果看到 `0.8.5`，说明最基本的 Python 包安装成功了。

如果你想把关键信息一起打印出来，更推荐这条：

```bash
python -c "import sys, torch, transformers, vllm; print(torch.__version__); print(transformers.__version__); print(vllm.__version__); print(sys.version)"
```

## 3. 为什么不推荐直接装进 adistill

如果你执意装进现有 `adistill`，风险点是：

- 现有 `torch 2.10.0+cu128` 和 vLLM 0.8.5 预编译 wheel 的默认二进制环境不一致
- 官方文档明确说，这种“复用已有 PyTorch/CUDA 组合”的情况更适合源码构建
- 一旦你为了 vLLM 改动现有环境，很可能反过来影响训练侧依赖

所以我的建议非常明确：

- 学习 vLLM demo 阶段，不要污染现有训练环境

## 4. 运行这个离线 demo

单卡 first run：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 conda run -n adistill-vllm085 python examples/vllm_offline/run_qwen_offline.py --model /data1/public_checkpoints/Qwen3-1.7B
```

这条命令里每一部分的作用：

- `CUDA_VISIBLE_DEVICES=0`：只让程序看到第 0 张 GPU
- `VLLM_USE_V1=0`：按你的项目要求，强制使用 vLLM 的 v0 路径
- `conda run -n adistill-vllm085`：不手动 `activate` 也能直接在指定环境里运行
- `--model`：指定模型路径；这里用的是你项目里现成的本地 Qwen3-1.7B

虽然脚本内部也会默认把 `VLLM_USE_V1` 设成 `0`，但我仍然建议你在命令行显式写出来，因为这样日志和命令记录更清楚。

如果你想用文件里的多条 prompt 做一个小批量测试：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 conda run -n adistill-vllm085 python examples/vllm_offline/run_qwen_offline.py --model /data1/public_checkpoints/Qwen3-1.7B --prompt-file examples/vllm_offline/sample_prompts.txt --output-file result/vllm_offline_demo_outputs.jsonl
```

这里：

- `--prompt-file`：从文本文件按行读取 prompt
- `--output-file`：把结果写成 JSONL，方便后面做自动评测或误差分析

双卡示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 VLLM_USE_V1=0 conda run -n adistill-vllm085 python examples/vllm_offline/run_qwen_offline.py --model /data1/public_checkpoints/Qwen3-1.7B --tensor-parallel-size 2
```

这里的 `--tensor-parallel-size 2` 表示把模型按张量并行切到 2 张可见 GPU 上。一个最重要的对应关系是：

- 可见 GPU 数量必须不少于 `tensor_parallel_size`

## 5. 这个 demo 在教你什么

脚本的核心 workflow 只有五步：

1. 先准备一个 prompt 列表
2. 用 `SamplingParams` 定义“如何生成”
3. 用 `LLM(...)` 定义“加载哪个模型，以及显存/并行怎么配”
4. 调用 `llm.generate(prompts, sampling_params)`
5. 把每条 prompt 的输出整理出来并保存

你可以把它和 Transformers 的 `model.generate()` 对照理解：

- Transformers 更像“你自己直接操作模型对象”
- vLLM 更像“你把请求交给一个专门为高吞吐推理优化的 engine”

直观上，vLLM 的优势主要来自：

- 更高效的 KV cache 管理
- 更适合批量请求的调度方式
- 更适合 serving / batch inference 的执行模型

## 6. 关键参数怎么理解

脚本里你最值得先掌握的是这些参数：

- `tensor_parallel_size`
  直观解释：把模型切到几张卡上一起算。
  一般 first run 先用 `1`，多卡跑通后再升。

- `max_model_len`
  直观解释：vLLM 预留多大的上下文窗口。
  值越大，KV cache 越大，显存占用通常越高。

- `max_num_seqs`
  直观解释：一次最多同时服务多少条序列。
  值越大，吞吐可能更好，但显存也更吃紧。

- `gpu_memory_utilization`
  直观解释：允许 vLLM 用掉多少比例的 GPU 显存。
  first run 先用 `0.8` 到 `0.85` 更稳。

- `enforce_eager`
  直观解释：不用 CUDA graph，回到更朴素的 eager 模式。
  常见用途是排障和省一部分显存。

- `temperature`
  直观解释：控制随机性。
  做教学和可复现实验时，经常先设成 `0.0`。

## 7. OOM 或首跑失败时怎么减配

vLLM 官方离线推理文档里给了几个很实用的减配方向。你 first run 如果炸显存，优先按这个顺序试：

1. 调小 `--max-model-len`
2. 调小 `--max-num-seqs`
3. 把 `--gpu-memory-utilization` 从 `0.85` 再降一点
4. 加上 `--enforce-eager`
5. 如果是多卡，确认 `CUDA_VISIBLE_DEVICES` 和 `--tensor-parallel-size` 是否匹配

## 8. 下一步你可以怎么扩展

这个 demo 跑通后，下一步最自然的扩展方向有三个：

1. 把输入从 `sample_prompts.txt` 扩展到你自己的 JSONL 数据集
2. 把输出接到现有 `grading/` 评测脚本，形成一个完整 offline inference -> eval 闭环
3. 再单独补一个 online serving demo，对比“本地 batch 推理”和“OpenAI-compatible server”的差别

## 参考资料

- vLLM 0.8.5 GPU 安装文档: https://docs.vllm.ai/en/v0.8.5/getting_started/installation/gpu.html
- vLLM 0.8.5 Offline Inference: https://docs.vllm.ai/en/v0.8.5/serving/offline_inference.html
- vLLM 0.8.5 Generative Models: https://docs.vllm.ai/en/v0.8.5/models/generative_models.html
- vLLM 0.8.5 Environment Variables: https://docs.vllm.ai/en/v0.8.5/serving/env_vars.html
- vLLM 0.8.5 PyPI 发布页: https://pypi.org/project/vllm/0.8.5/
