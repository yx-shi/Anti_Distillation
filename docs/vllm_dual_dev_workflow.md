# vLLM-Dual Development Workflow

本文档记录本项目后续修改 `vllm-dual` 的推荐流程。

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
