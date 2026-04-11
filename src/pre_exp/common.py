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
