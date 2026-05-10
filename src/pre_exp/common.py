from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件并返回对象列表。"""

    input_path = Path(path)
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_idx} in {input_path}.") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL line {line_idx} in {input_path} is not a JSON object.")
            records.append(item)
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """把对象列表写成 JSONL。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """把结果以易读 JSON 形式落盘。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def choose_subset_indices(dataset_size: int, max_samples: int, seed: int) -> list[int]:
    """从数据集中抽固定大小的随机子集，并按原始顺序返回索引。

    这里先随机抽样、再排序，是科研里很常见的一个折中：
    - 抽样本身由随机种子控制，方便复现
    - 最终顺序按原始索引排序，方便你回头人工排查某一道题
    """

    if max_samples <= 0 or max_samples >= dataset_size:
        return list(range(dataset_size))

    rng = random.Random(seed)
    selected = rng.sample(range(dataset_size), k=max_samples)
    selected.sort()
    return selected


def choose_subset_indices_from_pool(pool_indices: list[int], max_samples: int, seed: int) -> list[int]:
    """从给定候选 index 池中抽固定大小子集，并按原始 index 排序返回。"""

    pool = list(pool_indices)
    if max_samples <= 0 or max_samples >= len(pool):
        return sorted(pool)

    rng = random.Random(seed)
    selected = rng.sample(pool, k=max_samples)
    selected.sort()
    return selected


def choose_holdout_indices(
    dataset_size: int,
    *,
    exclude_max_samples: int,
    exclude_seed: int,
    max_samples: int,
    subset_seed: int,
) -> list[int]:
    """先排除一个固定训练子集，再从剩余 holdout 中抽评测子集。

    DeepScaleR 只有 train split，因此 main8000 后续评测需要显式排除当时
    `max_samples=8000, subset_seed=42` 抽到的训练题目。这里把这个范式集中到
    一个 helper 中，避免训练验证集、checkpoint eval、final eval 各自手写一套。
    """

    excluded: set[int] = set()
    if exclude_max_samples > 0:
        excluded = set(choose_subset_indices(dataset_size, exclude_max_samples, exclude_seed))

    holdout_pool = [idx for idx in range(dataset_size) if idx not in excluded]
    return choose_subset_indices_from_pool(holdout_pool, max_samples, subset_seed)
