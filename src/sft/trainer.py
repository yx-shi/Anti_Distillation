from __future__ import annotations

import math
import random
import time

# `datasets` 会间接导入 `pyarrow`。在一些 Conda + CUDA 环境里，如果先导入 torch，
# 动态链接器可能先锁定系统自带的旧版 libstdc++，随后 pyarrow 再加载时就会报
# `GLIBCXX_x.x.x not found`。因此这里显式把 datasets 放到 torch 前面导入。
from datasets import load_dataset
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        config=model_config,
        torch_dtype="auto",
        local_files_only=config.local_files_only,
    )
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
    """Create train/eval dataloaders and the matching distributed samplers."""

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
    return train_loader, eval_loader, train_sampler


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    runtime: DistributedContext,
    ignore_index: int,
) -> dict[str, float]:
    """Run a validation pass and report loss/perplexity."""

    model.eval()
    total_loss = 0.0
    total_steps = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, runtime.device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss = compute_causal_lm_loss(outputs.logits, batch["labels"], ignore_index)
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
    dataset_label = (
        f"{config.dataset_namespace}/{config.dataset_name}"
        if config.dataset_namespace
        else config.dataset_name
    )

    try:
        set_random_seed(config.seed + runtime.rank)

        tokenizer = build_tokenizer(config)
        model = build_model(config, runtime, tokenizer.pad_token_id)
        train_loader, eval_loader, train_sampler = build_dataloaders(config, tokenizer, runtime)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_training_steps = config.num_epochs * len(train_loader)
        if total_training_steps == 0:
            raise ValueError("Training dataloader is empty. Check dataset splits and batch size.")

        warmup_steps = int(total_training_steps * config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

        log_main(
            runtime,
            "training_start "
            f"device={runtime.device} "
            f"use_fsdp={runtime.use_fsdp} "
            f"world_size={runtime.world_size} "
            f"dataset_backend={config.dataset_backend} "
            f"dataset_name={dataset_label} "
            f"train_steps_per_epoch={len(train_loader)}",
        )

        global_step = 0
        running_loss = 0.0
        steps_since_log = 0
        stop_training = False

        for epoch in range(config.num_epochs):
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
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
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
                    metrics = evaluate(model, eval_loader, runtime, config.ignore_index)
                    log_main(
                        runtime,
                        f"[eval] step={global_step} "
                        f"val_loss={metrics['loss']:.4f} "
                        f"val_ppl={metrics['ppl']:.4f}",
                    )
                    model.train()

                if config.max_steps > 0 and global_step >= config.max_steps:
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

        metrics = evaluate(model, eval_loader, runtime, config.ignore_index)
        log_main(
            runtime,
            "training_done "
            f"steps={global_step} "
            f"val_loss={metrics['loss']:.4f} "
            f"val_ppl={metrics['ppl']:.4f}",
        )
    finally:
        cleanup_distributed(runtime)
