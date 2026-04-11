from __future__ import annotations

import os
from pathlib import Path


def ensure_writable_hf_datasets_cache() -> str:
    """为 Hugging Face datasets 指定一个可写 cache 目录。

    当前环境里，`~/.cache/huggingface/datasets` 可能已经有缓存，但所在挂载点是只读的。
    `datasets.load_dataset(...)` 即使只是复用缓存，也会尝试创建 lock 文件，
    于是会在“明明数据已经在本地”的情况下，因为 lock 无法写入而失败。

    这里采用一个非常实用的工程范式：
    - Hub cache 仍然可以沿用系统现有位置
    - 但 datasets 自己的工作缓存与 lock 文件统一放到 `/tmp`

    这样可以最小化对现有环境的干扰，同时避免只读文件系统导致的假失败。
    """

    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return cache_dir

    cache_dir = "/tmp/anti_distillation_hf_datasets_cache"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    return cache_dir
