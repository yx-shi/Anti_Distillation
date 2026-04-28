from __future__ import annotations

import json
import math
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

# `datasets` 会间接导入 `pyarrow`。在一些 Conda + CUDA 环境里，如果先导入 torch，
# 动态链接器可能先锁定系统自带的旧版 libstdc++，随后 pyarrow 再加载时就会报
# `GLIBCXX_x.x.x not found`。因此这里显式把 datasets 放到 torch 前面导入。
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from sft.config import TrainConfig
from sft.data import (
    SupervisedFineTuningCollator,
    SupervisedFineTuningDataset,
    load_supervised_dataset_splits,
)
from sft.distributed import (
    DistributedContext,
    cleanup_distributed,
    is_main_process,
    reduce_mean,
    setup_distributed,
)


def log_main(runtime: DistributedContext, message: str) -> None:
    """Print one log line from rank 0 and flush immediately for redirected stdout."""

    if is_main_process(runtime):
        print(message, flush=True)


def log_rank(runtime: DistributedContext, message: str) -> None:
    """Print one log line from every rank. Useful for locating distributed stalls."""

    print(f"[rank={runtime.rank}] {message}", flush=True)


def set_random_seed(seed: int) -> None:
    """Seed Python and PyTorch so repeated runs are easier to reproduce."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move every tensor in the batch to the target device with one helper call."""

    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def compute_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """Compute the standard next-token loss used by causal language models."""

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    valid_mask = flat_labels.ne(ignore_index)
    valid_count = valid_mask.sum()
    if valid_count.item() == 0:
        return flat_logits.sum() * 0.0

    loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=ignore_index,
        reduction="sum",
    )
    return loss / valid_count


def build_tokenizer(config: TrainConfig):
    """Load the tokenizer once and make sure it has a padding token for dynamic batching."""

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        local_files_only=config.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_model(config: TrainConfig, runtime: DistributedContext, pad_token_id: int | None):
    """Load the causal LM and wrap it with FSDP when running under torchrun."""

    model_config = AutoConfig.from_pretrained(
        config.model_name_or_path,
        local_files_only=config.local_files_only,
    )
    if getattr(model_config, "tie_word_embeddings", False):
        model_config.tie_word_embeddings = False

    model_kwargs = {
        "config": model_config,
        "torch_dtype": "auto",
        "local_files_only": config.local_files_only,
    }
    if config.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = config.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)
    if pad_token_id is not None:
        model.config.pad_token_id = pad_token_id
    elif model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    if runtime.use_fsdp:
        return FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=runtime.device,
            use_orig_params=True,
        )

    return model.to(runtime.device)


def build_dataloaders(config: TrainConfig, tokenizer, runtime: DistributedContext):
    """Create train/eval dataloaders and keep the raw eval dataset for offline checkpoint evaluation.

    一个常见范式是同时保留两种视图：
    - tokenized dataset / dataloader：给 loss 计算用
    - raw dataset：给训练外的 checkpoint rollout 评测用

    因为离线 rollout grading 需要原始 question 和原始 gold answer，不能只看 token ids。
    """

    train_split, eval_split = load_supervised_dataset_splits(config)
    train_dataset = SupervisedFineTuningDataset(train_split, tokenizer, config.max_length)
    eval_dataset = SupervisedFineTuningDataset(eval_split, tokenizer, config.max_length)

    collator = SupervisedFineTuningCollator(
        tokenizer=tokenizer,
        max_length=config.max_length,
        ignore_index=config.ignore_index,
        train_on_prompt=config.train_on_prompt,
    )

    train_sampler = None
    eval_sampler = None
    if runtime.use_fsdp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=True,
        )
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=False,
        )

    loader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": runtime.device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
        "collate_fn": collator,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        sampler=eval_sampler,
        **loader_kwargs,
    )
    return train_loader, eval_loader, train_sampler, eval_dataset.dataset


def build_dataset_label(config: TrainConfig) -> str:
    """生成日志里使用的数据源标签。"""

    if config.dataset_format == "distill_jsonl":
        train_label = config.train_file or "<missing-train-file>"
        if config.eval_file:
            eval_label = config.eval_file
        else:
            eval_label = f"{config.dataset_name}:{config.eval_split}"
        return f"distill_jsonl(train={train_label}, eval={eval_label})"

    if config.dataset_namespace:
        return f"{config.dataset_namespace}/{config.dataset_name}"
    return config.dataset_name


def save_model_checkpoint(
    model,
    tokenizer,
    runtime: DistributedContext,
    checkpoint_dir: Path,
) -> None:
    """把当前模型权重保存到指定目录。

    这里把“保存模型本体”的逻辑独立出来，是工程里非常常见的重构方式：
    - final checkpoint 和 periodic checkpoint 共用同一套保存范式
    - 后续如果要接离线 vLLM 评测，只需要保证中间目录和最终目录都可加载
    """

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if runtime.use_fsdp:
        # FSDP 下如果直接在每个 rank 上各自 `save_pretrained`，很容易写出不完整权重。
        # 这里使用 FULL_STATE_DICT + rank0_only 的常见范式：
        # - 先把完整参数聚合到 rank 0
        # - 再只由 rank 0 负责真正落盘
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            full_state_dict = model.state_dict()

        if is_main_process(runtime):
            model_to_save = model.module
            model_to_save.save_pretrained(
                checkpoint_dir,
                state_dict=full_state_dict,
                safe_serialization=False,
            )
            tokenizer.save_pretrained(checkpoint_dir)
        dist.barrier()
    else:
        if not is_main_process(runtime):
            return
        model.save_pretrained(checkpoint_dir, safe_serialization=False)
        tokenizer.save_pretrained(checkpoint_dir)


def save_training_outputs(
    model,
    tokenizer,
    config: TrainConfig,
    runtime: DistributedContext,
    metrics: dict[str, float | str],
) -> None:
    """在训练结束后保存最终 checkpoint、配置和最终指标。"""

    output_root = Path(config.output_dir)
    checkpoint_dir = output_root / "final_checkpoint"

    output_root.mkdir(parents=True, exist_ok=True)
    save_model_checkpoint(
        model=model,
        tokenizer=tokenizer,
        runtime=runtime,
        checkpoint_dir=checkpoint_dir,
    )

    if not is_main_process(runtime):
        return

    config_path = output_root / "train_config.json"
    metrics_path = output_root / "final_metrics.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def checkpoint_dir_for_step(output_dir: str, step: int) -> Path:
    """把 step 号映射到统一的 checkpoint 目录名。"""

    return Path(output_dir) / f"checkpoint-step-{step:06d}"


def save_periodic_checkpoint(
    model,
    tokenizer,
    config: TrainConfig,
    runtime: DistributedContext,
    step: int,
    metrics: dict[str, float | str],
) -> None:
    """保存一个周期性 checkpoint，并附带当前验证指标。"""

    checkpoint_dir = checkpoint_dir_for_step(config.output_dir, step)
    save_model_checkpoint(
        model=model,
        tokenizer=tokenizer,
        runtime=runtime,
        checkpoint_dir=checkpoint_dir,
    )

    if not is_main_process(runtime):
        return

    metrics_path = checkpoint_dir / "metrics.json"
    metadata_path = checkpoint_dir / "metadata.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump({"step": step}, f, ensure_ascii=False, indent=2)


def prune_old_checkpoints(output_dir: str, keep_limit: int, runtime: DistributedContext) -> None:
    """按 step 序删除较旧的周期性 checkpoint。"""

    if keep_limit <= 0 or not is_main_process(runtime):
        return

    output_root = Path(output_dir)
    checkpoint_dirs = sorted(
        path for path in output_root.glob("checkpoint-step-*") if path.is_dir()
    )
    stale_dirs = checkpoint_dirs[:-keep_limit]
    for checkpoint_dir in stale_dirs:
        shutil.rmtree(checkpoint_dir)


def compute_training_budget(config: TrainConfig, steps_per_epoch: int) -> tuple[int, int, float]:
    """把用户配置归一成“实际训练步数 + 实际循环 epoch 数”。"""

    if steps_per_epoch <= 0:
        raise ValueError("Training dataloader is empty. Check dataset splits and batch size.")

    if config.max_steps > 0:
        target_steps = config.max_steps
        loop_epochs = math.ceil(target_steps / steps_per_epoch)
    else:
        loop_epochs = config.num_epochs
        target_steps = loop_epochs * steps_per_epoch

    effective_epochs = target_steps / steps_per_epoch
    return target_steps, loop_epochs, effective_epochs


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    runtime: DistributedContext,
    config: TrainConfig,
) -> dict[str, float | str]:
    """Run validation metrics for LM loss only.

    当前训练主循环只负责快速给出 token-level 拟合指标。
    rollout 任务级评测已经从 trainer 主循环中解耦出去，改由独立脚本消费 checkpoint。
    这样训练速度和评测速度就不会再互相绑死。
    """

    model.eval()
    total_loss = 0.0
    total_steps = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, runtime.device)
        # 整段 NLL 前向不需要 generation cache；显式关闭可兼容右 padding + FlashAttention2。
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        loss = compute_causal_lm_loss(outputs.logits, batch["labels"], config.ignore_index)
        total_loss += loss.item()
        total_steps += 1

    if runtime.use_fsdp:
        total_loss_tensor = torch.tensor(total_loss, device=runtime.device)
        total_steps_tensor = torch.tensor(total_steps, device=runtime.device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_steps_tensor, op=dist.ReduceOp.SUM)
        total_loss = total_loss_tensor.item()
        total_steps = int(total_steps_tensor.item())

    avg_loss = total_loss / max(total_steps, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return {"loss": avg_loss, "ppl": ppl}


def train(config: TrainConfig) -> None:
    """Top-level SFT workflow: setup, load, train, evaluate, cleanup."""

    runtime = setup_distributed()
    dataset_label = build_dataset_label(config)

    try:
        set_random_seed(config.seed + runtime.rank)

        tokenizer = build_tokenizer(config)
        model = build_model(config, runtime, tokenizer.pad_token_id)
        train_loader, eval_loader, train_sampler, _eval_source_dataset = build_dataloaders(config, tokenizer, runtime)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        target_training_steps, loop_epochs, effective_epochs = compute_training_budget(
            config=config,
            steps_per_epoch=len(train_loader),
        )

        warmup_steps = int(target_training_steps * config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=target_training_steps,
        )

        log_main(
            runtime,
            "training_start "
            f"device={runtime.device} "
            f"use_fsdp={runtime.use_fsdp} "
            f"world_size={runtime.world_size} "
            f"dataset_format={config.dataset_format} "
            f"dataset_backend={config.dataset_backend} "
            f"dataset_name={dataset_label} "
            f"rollout_eval={config.rollout_eval} "
            f"rollout_eval_max_samples={config.rollout_eval_max_samples} "
            f"requested_attn_implementation={config.attn_implementation} "
            f"actual_attn_implementation={getattr(model.config, '_attn_implementation', 'unknown')} "
            f"output_dir={config.output_dir} "
            f"train_steps_per_epoch={len(train_loader)} "
            f"target_max_steps={target_training_steps} "
            f"effective_epochs_for_budget={effective_epochs:.2f} "
            f"checkpoint_every={config.checkpoint_every}",
        )

        global_step = 0
        running_loss = 0.0
        steps_since_log = 0
        stop_training = False
        latest_eval_metrics: dict[str, float | str] = {}

        for epoch in range(loop_epochs):
            model.train()
            log_main(runtime, f"epoch_start epoch={epoch}")

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            for batch in train_loader:
                step_start_time = time.perf_counter()
                is_first_step = global_step == 0

                if is_first_step and config.debug_fsdp:
                    log_rank(runtime, "first_step_stage=batch_fetched")

                batch = move_batch_to_device(batch, runtime.device)
                if is_first_step and config.debug_fsdp:
                    log_rank(
                        runtime,
                        "first_step_stage=batch_to_device "
                        f"input_shape={tuple(batch['input_ids'].shape)}",
                    )

                if is_first_step and config.debug_fsdp:
                    log_rank(runtime, "first_step_stage=before_forward")
                # 整段训练前向不需要 generation cache；显式关闭可兼容右 padding + FlashAttention2。
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                if is_first_step and config.debug_fsdp:
                    log_rank(runtime, "first_step_stage=after_forward")
                loss = compute_causal_lm_loss(outputs.logits, batch["labels"], config.ignore_index)

                optimizer.zero_grad(set_to_none=True)
                if is_first_step and config.debug_fsdp:
                    log_rank(runtime, "first_step_stage=before_backward")
                loss.backward()
                if is_first_step and config.debug_fsdp:
                    log_rank(runtime, "first_step_stage=after_backward")
                optimizer.step()
                scheduler.step()
                if is_first_step and config.debug_fsdp:
                    log_rank(runtime, "first_step_stage=after_optimizer_step")

                mean_loss = reduce_mean(loss, runtime)
                running_loss += mean_loss.item()
                steps_since_log += 1
                global_step += 1
                step_time = time.perf_counter() - step_start_time

                if global_step == 1:
                    log_main(
                        runtime,
                        f"first_step_done step={global_step} "
                        f"train_loss={mean_loss.item():.4f} "
                        f"step_time={step_time:.2f}s",
                    )

                should_log_step = global_step <= 3 or (
                    config.log_every > 0 and global_step % config.log_every == 0
                )
                if should_log_step and is_main_process(runtime):
                    avg_train_loss = running_loss / max(steps_since_log, 1)
                    print(
                        f"epoch={epoch} step={global_step} "
                        f"train_loss={avg_train_loss:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.6e} "
                        f"step_time={step_time:.2f}s",
                        flush=True,
                    )
                    running_loss = 0.0
                    steps_since_log = 0

                if config.eval_every > 0 and global_step % config.eval_every == 0:
                    metrics = evaluate(model, eval_loader, runtime, config)
                    latest_eval_metrics = metrics
                    eval_message = (
                        f"[eval] step={global_step} "
                        f"val_loss={metrics['loss']:.4f} "
                        f"val_ppl={metrics['ppl']:.4f}"
                    )
                    log_main(runtime, eval_message)
                    model.train()

                should_save_checkpoint = (
                    config.checkpoint_every > 0
                    and global_step % config.checkpoint_every == 0
                    and global_step < target_training_steps
                )
                if should_save_checkpoint:
                    save_periodic_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        config=config,
                        runtime=runtime,
                        step=global_step,
                        metrics=latest_eval_metrics,
                    )
                    prune_old_checkpoints(
                        output_dir=config.output_dir,
                        keep_limit=config.max_checkpoints_to_keep,
                        runtime=runtime,
                    )
                    model.train()

                if global_step >= target_training_steps:
                    stop_training = True
                    break

            if is_main_process(runtime) and steps_since_log > 0:
                avg_train_loss = running_loss / steps_since_log
                print(
                    f"epoch={epoch} step={global_step} "
                    f"train_loss={avg_train_loss:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.6e}",
                    flush=True,
                )
                running_loss = 0.0
                steps_since_log = 0

            if stop_training:
                break

        metrics = evaluate(model, eval_loader, runtime, config)
        final_message = (
            "training_done "
            f"steps={global_step} "
            f"val_loss={metrics['loss']:.4f} "
            f"val_ppl={metrics['ppl']:.4f}"
        )
        log_main(runtime, final_message)
        save_training_outputs(
            model=model,
            tokenizer=tokenizer,
            config=config,
            runtime=runtime,
            metrics=metrics,
        )
    finally:
        cleanup_distributed(runtime)
