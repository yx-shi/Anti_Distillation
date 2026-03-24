import math
import os
# dataclass：Python 内置装饰器，自动生成 __init__/__repr__/__eq__ 等样板代码
# 详见 SFTCollator 类的注释
from dataclasses import dataclass
# typing 模块提供类型注解支持，让函数签名更可读；Python 3.9+ 可直接用 dict/list 替代
# Any = 任意类型；Dict[K,V] = 字典；List[T] = 列表
from typing import Any, Dict, List

from datasets import load_dataset     # 先导入 datasets/pyarrow，避免与 torch 的 libstdc++ 冲突
import torch                          # PyTorch 核心库，提供 Tensor、自动求导、神经网络模块
import torch.distributed as dist
import torch.nn.functional as F       # 无状态的函数式接口：激活函数、损失函数等（无可学习参数）
from torch.utils.data import DataLoader, Dataset, DistributedSampler  # Dataset：定义单样本；DataLoader：自动批处理 + 多进程加载
from data import SFTDataset, SFTCollator, format_dolly_sample  # 导入我们定义的数据处理类和函数
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from transformers import (
    AutoModelForCausalLM,              # 自动识别模型架构并加载因果语言模型（Causal LM，即 GPT 风格）
    AutoTokenizer,                     # 自动识别并加载对应的分词器
    get_linear_schedule_with_warmup,   # HF 提供的 LR 调度器：先线性 warmup，再线性衰减到 0
)
from config import *                         # 导入配置文件，包含模型/数据集名称、训练超参等全局设置
from utils import *


def build_tokenized_feature(example: Dict[str, Any], tokenizer, max_length: int) -> Dict[str, Any]:
    """
    用和 SFTDataset.__getitem__ 相同的逻辑，手工构造单条训练样本。

    之所以单独写成函数，是为了做“教学可视化”时不必依赖真实数据集，
    直接传入手写的小样本即可复用同一套处理流程。
    """
    item = format_dolly_sample(example)

    prompt_ids = tokenizer(
        item["prompt"],
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    full_ids = tokenizer(
        item["full_text"],
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    return {
        "prompt_text": item["prompt"],
        "full_text": item["full_text"],
        "prompt_ids": prompt_ids,
        "input_ids": full_ids,
    }


def setup_distributed() -> Dict[str, Any]:
    """
    初始化分布式/FSDP 运行时。

    约定：
    - 单卡直接 `python src/phaseB_infer.py`
    - 多卡通过 `torchrun --nproc_per_node=... src/phaseB_infer.py`

    当使用 torchrun 时，环境变量里会自动带上：
    - WORLD_SIZE：总进程数（通常=总 GPU 数）
    - RANK：全局 rank
    - LOCAL_RANK：本机第几张卡
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_fsdp = world_size > 1

    if use_fsdp:
        if not torch.cuda.is_available():
            raise RuntimeError("FSDP requires CUDA GPUs, but torch.cuda.is_available() is False.")

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
    else:
        rank = 0
        local_rank = 0
        device = DEVICE

    return {
        "use_fsdp": use_fsdp,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "device": device,
    }


def cleanup_distributed(runtime: Dict[str, Any]) -> None:
    if runtime["use_fsdp"] and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(runtime: Dict[str, Any]) -> bool:
    return runtime["rank"] == 0


def distributed_mean(value: torch.Tensor, runtime: Dict[str, Any]) -> torch.Tensor:
    """
    在多卡下把每个 rank 上的标量做 all-reduce 平均。
    这样日志里的 loss 才是“全局平均”而不是单张卡的局部值。
    """
    if not runtime["use_fsdp"]:
        return value.detach()

    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= runtime["world_size"]
    return reduced


# ------------------------------
# Loss computation
# ------------------------------
def compute_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """
    logits: [B, T, V]  — B=batch_size, T=seq_len, V=vocab_size
    labels: [B, T]

    For causal LM, token at position t predicts token at position t+1.
    So we shift:
      shift_logits = logits[:, :-1, :]   — 去掉最后一个位置的预测（因为没有 t+1 目标）
      shift_labels = labels[:, 1:]       — 去掉第一个 token（它没有前驱来预测它）

    对齐后：shift_logits[i] 的预测目标就是 shift_labels[i]
    """
    # TODO-2:
    # Complete the shifting and cross-entropy computation.
    #
    # ★ Causal LM 的 loss 计算范式（必须掌握）：
    # 位置 0 的 logit → 预测位置 1 的 token
    # 位置 1 的 logit → 预测位置 2 的 token
    # ...
    # 位置 T-2 的 logit → 预测位置 T-1 的 token
    # 位置 T-1 的 logit → 无目标（丢弃）
    #
    # logits[:, :-1, :]：取前 T-1 个位置的 logit，shape [B, T-1, V]
    # labels[:, 1:]：取第 2 到第 T 个 token 作为目标，shape [B, T-1]
    shift_logits = logits[:, :-1, :].contiguous()  # .contiguous()：确保内存连续，.view() 前必须调用
    shift_labels = labels[:, 1:].contiguous()

    # F.cross_entropy 要求：
    #   input: [N, C]，N=样本数，C=类别数（此处=vocab_size）
    #   target: [N]，每个值是正确类别的索引
    # 所以用 .view() 把 [B, T-1, V] 展平为 [B*(T-1), V]，[B, T-1] 展平为 [B*(T-1)]
    # .view(-1, C) 中 -1 表示"自动推断该维度大小"
    # ignore_index=-100：标签为 -100 的位置（prompt + padding）不参与 loss 计算
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    # 关键边界情况：
    # 如果一个 batch 里所有位置都被 mask 成 ignore_index（常见原因是 prompt 太长，
    # full_text 截断后 response 完全消失），那么 F.cross_entropy(..., reduction="mean")
    # 会对“空集合”求平均，结果变成 nan。
    #
    # 训练/验证日志里偶发的 train_loss=nan、稳定的 val_loss=nan，通常就是这个原因。
    valid_mask = flat_labels.ne(ignore_index)
    valid_count = valid_mask.sum()
    if valid_count.item() == 0:
        # 返回一个 0 loss，表示这个 batch 没有任何可监督 token。
        # 用 logits.sum() * 0 保持返回值仍在当前 device/dtype 上。
        return flat_logits.sum() * 0.0

    loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=ignore_index,
        reduction="sum",
    ) / valid_count
    return loss


def run_mini_batch_walkthrough(tokenizer):
    """
    手工构造一个很小的 batch，专门展示：
    1. prompt / full_text 如何格式化
    2. tokenize 后得到什么
    3. truncation 在哪里发生
    4. dynamic padding 如何补齐
    5. attention_mask / labels / shift_labels 长什么样

    这里故意把 demo_max_length 设得比较小，让第一条样本发生截断，
    第二条样本更短，从而在同一个 batch 里同时看到“截断”和“padding”。
    """
    demo_max_length = 28

    demo_examples = [
        {
            "instruction": "Write a friendly reply to the user.",
            "context": "",
            "response": (
                "Hello there! It is nice to meet you today. "
                "I hope your research is going smoothly."
            ),
        },
        {
            "instruction": "Answer with one short word.",
            "context": "",
            "response": "Yes.",
        },
    ]

    demo_features = [
        build_tokenized_feature(example, tokenizer, demo_max_length)
        for example in demo_examples
    ]

    demo_collator = SFTCollator(
        tokenizer=tokenizer,
        max_length=demo_max_length,
        ignore_index=IGNORE_INDEX,
        train_on_prompt=False,
    )
    demo_batch = demo_collator(demo_features)

    print("\n=== Mini SFT Batch Walkthrough ===")
    print(f"demo_max_length={demo_max_length}")
    print("目标：在一个 batch 中同时观察 truncation、padding、attention_mask、labels、shift_labels")

    for sample_idx, feature in enumerate(demo_features):
        original_full_ids = tokenizer(
            feature["full_text"],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        print(f"\n----- Sample {sample_idx} -----")
        print("[Prompt text]")
        print(feature["prompt_text"])
        print("[Full text]")
        print(feature["full_text"])
        print(
            f"prompt_token_len={len(feature['prompt_ids'])} "
            f"full_token_len_after_trunc={len(feature['input_ids'])} "
            f"full_token_len_before_trunc={len(original_full_ids)}"
        )
        if len(original_full_ids) > len(feature["input_ids"]):
            print("该样本发生了 truncation：full_text 超过 demo_max_length，尾部 token 被截掉。")
        else:
            print("该样本没有发生 truncation。")

        print_sequence_table(
            tokenizer,
            title="[prompt_ids]",
            values=feature["prompt_ids"],
        )
        print_sequence_table(
            tokenizer,
            title="[input_ids before padding]",
            values=feature["input_ids"],
        )

    batch_input_ids = demo_batch["input_ids"].tolist()
    batch_attention_mask = demo_batch["attention_mask"].tolist()
    batch_labels = demo_batch["labels"].tolist()
    batch_shift_labels = demo_batch["labels"][:, 1:].tolist()

    print("\n=== After Collator (dynamic padding + labels) ===")
    print(f"batch input_ids shape      = {tuple(demo_batch['input_ids'].shape)}")
    print(f"batch attention_mask shape = {tuple(demo_batch['attention_mask'].shape)}")
    print(f"batch labels shape         = {tuple(demo_batch['labels'].shape)}")

    for sample_idx in range(len(demo_features)):
        print(f"\n===== Batch sample {sample_idx} =====")
        print_sequence_table(
            tokenizer,
            title="[input_ids after padding]",
            values=batch_input_ids[sample_idx],
        )
        print_sequence_table(
            tokenizer,
            title="[attention_mask]",
            values=batch_attention_mask[sample_idx],
            decode_tokens=False,
        )
        print_sequence_table(
            tokenizer,
            title="[labels]",
            values=batch_labels[sample_idx],
        )
        print_sequence_table(
            tokenizer,
            title="[shift_labels = labels[:, 1:]]",
            values=batch_shift_labels[sample_idx],
        )

        active_loss_positions = [
            pos for pos, value in enumerate(batch_shift_labels[sample_idx]) if value != IGNORE_INDEX
        ]
        print(f"参与 loss 计算的位置（在 shift_labels 中）: {active_loss_positions}")
        print("解释：prompt 和 padding 的 label 会被置为 -100，cross_entropy 会忽略这些位置。")


# ------------------------------
# Eval
# ------------------------------
# ★ @torch.no_grad()：函数装饰器，禁用该函数内的梯度计算
# 推理/评估时不需要反向传播，关闭梯度：
#   1. 节省显存（不存储计算图中的中间激活值）
#   2. 加速前向传播
# 等价写法：在函数体内用 with torch.no_grad(): 包裹所有代码
@torch.no_grad()
def evaluate(model, dataloader, device: str, runtime: Dict[str, Any]) -> Dict[str, float]:
    # model.eval()：切换到评估模式
    # 影响两类层的行为：
    #   - Dropout：训练时随机丢弃神经元（正则化），评估时关闭（全部保留）
    #   - BatchNorm：训练时用当前 batch 统计量，评估时用训练期间积累的运行统计量
    # 与 model.train() 对应，两者必须成对使用
    model.eval()
    total_loss = 0.0
    total_steps = 0

    for batch in dataloader:
        # ★ 将 batch 中所有 Tensor 移到目标设备（CPU/GPU）
        # 字典推导式：{新键: 新值 for 键, 值 in 字典.items()}
        # .to(DEVICE) 返回新 Tensor（如已在目标设备则不复制）
        batch = {k: v.to(device) for k, v in batch.items()}

        # 调用模型做前向传播，返回 CausalLMOutputWithCrossAttentions 对象
        # 注意：这里没有传 labels，所以模型不自动计算 loss，我们手动计算
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        # outputs.logits：shape [B, T, V]，是模型输出的原始得分（未经 softmax）
        logits = outputs.logits
        loss = compute_causal_lm_loss(logits, batch["labels"])

        # .item()：将单元素 Tensor 转为 Python float，避免不断累积计算图
        total_loss += loss.item()
        total_steps += 1

    avg_loss = total_loss / max(total_steps, 1)  # max(..., 1) 防止除以零

    # 多卡验证时，每个 rank 只看到自己那部分数据，所以要把分子/分母都做 all-reduce。
    if runtime["use_fsdp"]:
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_steps_tensor = torch.tensor(total_steps, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_steps_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (total_loss_tensor / total_steps_tensor.clamp(min=1)).item()

    # Perplexity（困惑度）= exp(loss)，是语言模型的标准评估指标
    # 直觉：困惑度越低，模型对文本的预测越"不困惑"，越接近 1 越完美
    # avg_loss < 20 是防止 exp 溢出（e^20 ≈ 5 亿，超出一般范围则直接返回 inf）
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return {"loss": avg_loss, "ppl": ppl}


# ------------------------------
# Main training function
# ------------------------------
def main():
    runtime = setup_distributed()
    device = runtime["device"]

    try:
        if is_main_process(runtime):
            print(
                f"Using device: {device} | use_fsdp={runtime['use_fsdp']} "
                f"| world_size={runtime['world_size']}"
            )

        # ★ AutoTokenizer.from_pretrained(name)：
        # 从 HF Hub（或本地缓存）下载分词器配置和词汇表文件
        # 会自动识别模型类型，如 distilgpt2 → GPT2Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            # GPT 系列模型词汇表里没有专用的 [PAD] token，这是标准处理方式：
            # 将 eos_token（序列结束符，如 <|endoftext|>）复用为 pad_token
            # padding 时 token ID 和 eos 相同，但 attention_mask=0 确保模型忽略它
            tokenizer.pad_token = tokenizer.eos_token

        # 先用一个手工构造的小 batch 把训练数据流走一遍。
        # 这类可视化打印只需要 rank0 做一次，避免多卡下重复刷屏。
        # if is_main_process(runtime):
        #     run_mini_batch_walkthrough(tokenizer)

        # ★ AutoModelForCausalLM.from_pretrained(name)：
        # 下载预训练权重并加载到内存。这里先在 CPU 上构造模型对象，
        # 再根据是否启用 FSDP 决定后续放置方式。
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        if runtime["use_fsdp"]:
            # FSDP 的核心思想：
            # - 不是每张卡都完整保存一份模型参数
            # - 而是把参数/梯度/优化器状态分片到不同 rank 上
            # 这样单卡显存压力会明显下降，更适合大模型训练。
            #
            # 这里先采用最容易理解的“整模型包一层 FSDP”写法。
            # 真正大规模训练时，通常还会按 Transformer block 做更细粒度 auto-wrap。
            model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=torch.device(device),
                use_orig_params=True,
            )
        else:
            # 单卡/CPU 模式下，仍然按普通 PyTorch 方式把模型移到目标设备
            model.to(device)

        # ★ load_dataset(name) 返回 DatasetDict，类似 {"train": Dataset, "test": Dataset}
        # 数据首次下载后会缓存到 ~/.cache/huggingface/datasets/，后续秒加载
        raw_dataset = load_dataset(DATASET_NAME, "main")
        train_ds = raw_dataset["train"]
        test_ds = raw_dataset["test"]

        # 用自定义 Dataset 类包装 HF Dataset，使其适配 DataLoader
        train_dataset = SFTDataset(train_ds, tokenizer, MAX_LENGTH)
        val_dataset = SFTDataset(test_ds, tokenizer, MAX_LENGTH)

        # 实例化 collator（因为用了 @dataclass，直接传关键字参数即可）
        collator = SFTCollator(
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
            ignore_index=IGNORE_INDEX,
            train_on_prompt=False,  # response-only loss：只对 response 部分优化
        )

        # 多卡训练时，DataLoader 需要用 DistributedSampler。
        # 否则每个 rank 都会看到同一份训练数据，等于“重复训练”，FSDP 就失去意义了。
        train_sampler = None
        val_sampler = None
        if runtime["use_fsdp"]:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=runtime["world_size"],
                rank=runtime["rank"],
                shuffle=True,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=runtime["world_size"],
                rank=runtime["rank"],
                shuffle=False,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=collator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False if val_sampler is None else False,
            sampler=val_sampler,
            collate_fn=collator,
        )

        # ★ AdamW：Adam + Weight Decay 解耦（L2 正则化）
        # 注意：如果用了 FSDP，optimizer 要基于“包裹后的 model.parameters()”创建。
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        # 计算总步数：len(train_loader) = 当前 rank 每个 epoch 的步数
        total_training_steps = NUM_EPOCHS * len(train_loader)
        warmup_steps = int(total_training_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

        global_step = 0
        model.train()

        demo_prompt = (
            "### Instruction:\n"
            "Explain what supervised fine-tuning is in simple words.\n\n"
            "### Response:\n"
        )

        if is_main_process(runtime):
            print("\n=== Before training ===")
            print(generate_preview(model, tokenizer, demo_prompt, device=device))

        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0

            # DistributedSampler 需要在每个 epoch 手动 set_epoch，
            # 这样不同 epoch 的 shuffle 才会不同，且各 rank 之间保持一致。
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits = outputs.logits
                loss = compute_causal_lm_loss(logits, batch["labels"])

                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪在大模型训练里很常见，这里先保留为注释位。
                # 如果后面训练不稳定，可以打开：
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                # 多卡日志取全局平均，更符合你直觉里的“整个训练任务 loss”。
                mean_loss = distributed_mean(loss, runtime)
                running_loss += mean_loss.item()
                global_step += 1

                if is_main_process(runtime) and global_step % LOG_EVERY == 0:
                    avg_train_loss = running_loss / LOG_EVERY
                    print(
                        f"epoch={epoch} step={global_step} "
                        f"train_loss={avg_train_loss:.4f} lr={scheduler.get_last_lr()[0]:.6e}"
                    )
                    running_loss = 0.0

                if global_step % EVAL_EVERY == 0:
                    metrics = evaluate(model, val_loader, device=device, runtime=runtime)
                    if is_main_process(runtime):
                        print(
                            f"[eval] step={global_step} val_loss={metrics['loss']:.4f} "
                            f"val_ppl={metrics['ppl']:.4f}"
                        )
                        print("\n=== Generation preview ===")
                        print(generate_preview(model, tokenizer, demo_prompt, device=device))
                        print("==========================\n")
                    model.train()

        if is_main_process(runtime):
            print("\n=== After training ===")
            print(generate_preview(model, tokenizer, demo_prompt, device=device))

        # TODO-5:
        # Save model and tokenizer.
        # ★ FSDP 下保存权重比单卡稍复杂，因为参数是分片的。
        # 这里先不在教学版里展开 state_dict 聚合保存，先把“如何训练起来”这条链路打通。
        # 等你把 FSDP 训练流程理解后，再单独做 checkpoint 保存会更清晰。
    finally:
        cleanup_distributed(runtime)


def show_mini_training():
    print(f"Using device: {DEVICE}")

    # ★ AutoTokenizer.from_pretrained(name)：
    # 从 HF Hub（或本地缓存）下载分词器配置和词汇表文件
    # 会自动识别模型类型，如 distilgpt2 → GPT2Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        # GPT 系列模型词汇表里没有专用的 [PAD] token，这是标准处理方式：
        # 将 eos_token（序列结束符，如 <|endoftext|>）复用为 pad_token
        # padding 时 token ID 和 eos 相同，但 attention_mask=0 确保模型忽略它
        tokenizer.pad_token = tokenizer.eos_token

    # 先用一个手工构造的小 batch 把训练数据流走一遍。
    # 这样即使你还没开始真正训练，也能先看清楚：
    # prompt/full_text -> tokenize -> truncation -> padding -> attention_mask -> labels -> shift_labels
    run_mini_batch_walkthrough(tokenizer)

# ★ Python 标准范式：只有直接运行此脚本时才执行 main()
# 如果被其他模块 import，则不执行（__name__ 此时是模块名而非 "__main__"）
if __name__ == "__main__":
    main()
