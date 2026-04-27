# vLLM-Dual Development Workflow

本文档记录本项目后续修改 `vllm-dual` 的推荐流程。

## 0. 当前实现事实

vLLM-dual 已在 vLLM 0.8.5 V0 路径中实现 hard/soft token-level adversarial decoding 首版。

核心入口：

- 源码副本：`vllm_dual/vllm/`
- 同步目标：`$HOME/miniconda3/envs/adistill-unified/lib/python3.10/site-packages/vllm/`
- 配置入口：`dual_model_config`
- worker：`vllm.worker.dual_worker.DualModelWorker`

新增配置字段：

- `adversarial_mode`：`special_token`、`hard`、`soft`。默认 `special_token` 保留旧 special-token 仲裁行为。
- `hard_candidate_top_k`：hard/soft 共用的 Teacher top-k 候选集合大小。
- `hard_candidate_top_p`：在 Teacher top-k 内按累计概率进一步截断候选集合。
- `soft_student_weight`：soft mode 中 Student logprob 的惩罚权重。
- `soft_temperature`：soft mode 对重加权 score 采样前的温度。
- `debug_log_interval`：adversarial step 标记打印间隔。

实现口径：

- 只有显式传入 `dual_model_config` 时，`arg_utils.py` 才把 `parallel_config.worker_cls` 切到 `DualModelWorker`。
- hard/soft 模式会在 Teacher 和 Student worker 的 sampler 上打开 `include_gpu_probs_tensor`，从 `SamplerOutput.sampled_token_probs` 和 `SamplerOutput.logprobs` 读取完整 GPU 分布。
- 使用的是 sampler 处理后的分布，因此已包含 temperature、top-k/top-p/min-p、min token、repetition/presence/frequency penalty 等采样侧处理。
- hard mode 在 Teacher 可接受候选集合中选 Student logprob 最低的 token。
- soft mode 在 Teacher 候选集合中计算 `teacher_logprob - soft_student_weight * student_logprob`，再按 `soft_temperature` 采样。

当前限制：

- Teacher/Student tokenizer 和 vocab shape 必须对齐；不对齐会在 `DualModelWorker` 中报 tensor shape mismatch。
- 当前 smoke 用 Qwen3-1.7B 作为 reasoner、Qwen2.5-0.5B-Instruct 作为 controller，已验证工程链路；正式实验仍应改回项目默认 Teacher/Student 组合并重新 smoke。
- 目前只完成 `vllm_dual/test_dual_worker.py` 级别 smoke，还没有接入 `src/pre_exp/` 的完整 candidate -> dataset -> SFT -> eval 框架。

2026-04-27 验证状态：

- `python -m py_compile vllm_dual/test_dual_worker.py vllm_dual/vllm/config.py vllm_dual/vllm/engine/arg_utils.py vllm_dual/vllm/worker/dual_worker.py` 通过。
- `ASSUME_YES=1 DRY_RUN=1 bash sync.sh` 通过，同步范围为 `config.py`、`engine/arg_utils.py`、`worker/dual_worker.py`。
- `ASSUME_YES=1 bash sync.sh` 通过，备份目录为 `.sync_backups/vllm_20260427_211355`。
- hard smoke 通过，日志包含 `SMOKE_DUAL_EFFECTIVE ... adv_mode=hard` 和 `ADISTILL_DUAL_ADVERSARIAL enabled mode=hard`。
- soft smoke 通过，日志包含 `SMOKE_DUAL_EFFECTIVE ... adv_mode=soft` 和 `ADISTILL_DUAL_ADVERSARIAL enabled mode=soft`。
- 普通 vLLM smoke 通过，日志显示 `parallel_config.worker_cls: vllm.worker.worker.Worker`。

## 1. 目录职责

- `vllm_dual/vllm/`：本项目维护的 vLLM Python 源码副本，后续 adversarial decoding 的代码在这里改。
- `sync.sh`：把 `vllm_dual/vllm/` 同步到当前 conda 环境中已安装的 `site-packages/vllm/`。
- conda 环境 `adistill-unified`：实际运行实验时 import 的 vLLM 包位置。

不要把 `vllm_dual/` 加到 `PYTHONPATH`。否则 Python 会优先 import 工作区源码目录，但这个目录没有编译好的 `vllm._C` CUDA 扩展，容易报 `ModuleNotFoundError: No module named 'vllm._C'`。

## 2. import 机制

普通预实验脚本从项目根目录运行，例如：

```bash
conda run -n adistill-unified python src/pre_exp/teacher_generate.py
```

Python 的 import 搜索路径会包含项目根目录和 `src/`，但不会自动包含 `vllm_dual/`。因此：

```python
import vllm
```

会导入 conda 环境里的 `site-packages/vllm`，而不是 `vllm_dual/vllm`。

`vllm_dual/test_dual_worker.py` 比较特殊：测试文件就在 `vllm_dual/` 目录下。如果直接运行它，Python 默认会把 `vllm_dual/` 放到搜索路径最前面。测试脚本里已经显式移除了这个路径，确保测试的是同步后的 conda 版 vLLM。

## 3. 常规开发循环

1. 修改源码：

```bash
$EDITOR vllm_dual/vllm/worker/dual_worker.py
```

2. 做一次 dry run：

```bash
ASSUME_YES=1 DRY_RUN=1 bash sync.sh
```

- `ASSUME_YES=1`：跳过交互确认，方便脚本化执行。
- `DRY_RUN=1`：只显示将同步哪些文件，不真正写入 conda 环境。

3. 正式同步：

```bash
ASSUME_YES=1 bash sync.sh
```

同步前脚本会把 conda 环境里现有的 vLLM Python 源码备份到 `.sync_backups/`。

4. 跑 import 检查：

```bash
VLLM_USE_V1=0 conda run -n adistill-unified python -c "import vllm; from vllm.worker.dual_worker import DualModelWorker; print(vllm.__version__, DualModelWorker)"
```

- `VLLM_USE_V1=0`：强制使用 vLLM 0.8.5 的 V0 engine 路径。
- `-n adistill-unified`：指定 conda 环境。
- `-c`：让 Python 直接执行后面的字符串代码。

5. 跑 dual smoke：

```bash
bash vllm_dual/test.sh
```

6. 跑普通 vLLM smoke，确认预实验路径仍然不走 dual worker：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}" \
conda run -n adistill-unified python src/pre_exp/teacher_generate.py \
  --model-name-or-path /home/disk1/public_checkpoint/Qwen3-1.7B \
  --max-samples 1 \
  --num-candidates 1 \
  --max-new-tokens 16 \
  --prompt-batch-size 1 \
  --save-every-prompts 1 \
  --tensor-parallel-size 1 \
  --max-model-len 1024 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.5 \
  --enforce-eager \
  --output-file /tmp/adistill_plain_vllm_smoke.jsonl
```

这里几个常用参数含义：

- `CUDA_VISIBLE_DEVICES=0`：让当前进程只看到物理 GPU 0。
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`：使用更稳的多进程启动方式，避免 fork 后重复初始化 CUDA。
- `LIBRARY_PATH=.../stubs`：让 Triton 编译 launcher 时能找到 `libcuda.so`。
- `--max-samples 1`：只抽 1 条样本做 smoke。
- `--max-new-tokens 16`：只生成 16 个 token，快速验证链路。
- `--enforce-eager`：关闭 CUDA graph/编译路径，降低 smoke 测试的不确定性。

## 4. 判断是否启用 dual worker

普通路径日志应看到：

```text
parallel_config.worker_cls: vllm.worker.worker.Worker
```

dual 路径日志应看到：

```text
parallel_config.worker_cls: vllm.worker.dual_worker.DualModelWorker
```

只有传入 `dual_model_config` 时，才会切到 `DualModelWorker`。

## 5. 回退

如果同步后环境坏了，可以从最近一次备份回退：

```bash
rsync -av .sync_backups/<backup_name>/ "$HOME/miniconda3/envs/adistill-unified/lib/python3.10/site-packages/vllm/"
```

- `rsync -av`：递归同步并打印详细文件列表；`-a` 是 archive 模式，保留目录结构和文件属性；`-v` 是 verbose，显示同步过程。

回退后再跑 import 检查和普通 vLLM smoke。
