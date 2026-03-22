import math
# dataclass：Python 内置装饰器，自动生成 __init__/__repr__/__eq__ 等样板代码
# 详见 SFTCollator 类的注释
from dataclasses import dataclass
# typing 模块提供类型注解支持，让函数签名更可读；Python 3.9+ 可直接用 dict/list 替代
# Any = 任意类型；Dict[K,V] = 字典；List[T] = 列表
from typing import Any, Dict, List

from datasets import load_dataset     # 先导入 datasets/pyarrow，避免与 torch 的 libstdc++ 冲突
import torch                          # PyTorch 核心库，提供 Tensor、自动求导、神经网络模块
import torch.nn.functional as F       # 无状态的函数式接口：激活函数、损失函数等（无可学习参数）
from torch.utils.data import Dataset, DataLoader  # Dataset：定义单样本；DataLoader：自动批处理 + 多进程加载
from transformers import (
    AutoModelForCausalLM,              # 自动识别模型架构并加载因果语言模型（Causal LM，即 GPT 风格）
    AutoTokenizer,                     # 自动识别并加载对应的分词器
    get_linear_schedule_with_warmup,   # HF 提供的 LR 调度器：先线性 warmup，再线性衰减到 0
)
from config import *                         # 导入配置文件，包含模型/数据集名称、训练超参等全局设置


# ============================================================
# PhaseB: Minimal SFT demo without HF Trainer
# ------------------------------------------------------------
# Goals:
# 1. Keep tokenizer/model loading from Hugging Face.
# 2. Write Dataset / collate_fn / train loop / eval loop by hand.
# 3. Leave several TODOs for you to complete.
# ============================================================

# ------------------------------
# Text formatting helpers
# ------------------------------
def format_dolly_sample(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert Dolly-style record into prompt/completion.

    Dolly examples typically contain:
      - instruction
      - context
      - response

    We format them into a single instruction-following sample.
    """
    # .strip() 去掉首尾空白字符（空格、换行等）
    instruction = example["instruction"].strip()
    # .get("context") 比 example["context"] 更安全：键不存在时返回 None 而非报 KeyError
    # "or """ 处理 None 和空字符串两种情况（None or "" == ""）
    context = (example.get("context") or "").strip()
    response = example["response"].strip()

    # 根据是否有 context 选择不同的 prompt 模板
    # 这种 Alpaca 风格的模板是 SFT 数据格式化的常见范式
    if context:
        # f-string（格式化字符串字面量）：f"...{变量}..." 在运行时插值
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{context}\n\n"
            f"{RESPONSE_PREFIX}"       # 此处 prompt 以 "### Response:\n" 结尾
        )
    else:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            f"{RESPONSE_PREFIX}"
        )

    completion = response
    # full_text = prompt + response，是完整的训练序列
    # 模型看到 full_text，但只对 response 部分计算 loss（见 SFTCollator）
    full_text = prompt + completion

    return {
        "prompt": prompt,
        "completion": completion,
        "full_text": full_text,
    }


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


# ------------------------------
# Dataset
# ------------------------------
# ★ PyTorch Dataset 的标准写法范式（必须掌握）：
#   继承 torch.utils.data.Dataset，实现三个方法：
#   1. __init__：存储数据和配置
#   2. __len__：返回数据集大小（DataLoader 用此决定 epoch 长度）
#   3. __getitem__：按索引返回单个样本（DataLoader 内部调用此方法取数据）
class SFTDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length: int):
        self.dataset = hf_dataset   # HF Dataset 对象，支持 dataset[i] 取第 i 条
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # DataLoader 通过此方法知道一共有多少样本，从而决定 epoch 内迭代多少次
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # DataLoader 会传入整数索引 idx，返回单个样本（Python 原生 dict/list，不用是 Tensor）
        raw = self.dataset[idx]           # 得到原始 Dolly 字典
        item = format_dolly_sample(raw)   # 格式化为 prompt/completion/full_text

        # 为什么要分别 tokenize prompt 和 full_text？
        # 因为 response-only loss 需要知道 prompt 在 token 空间的长度，
        # 才能在 Collator 里把 prompt 对应的 labels 位置遮掩掉（设为 -100）
        # 注意：不能简单地用字符长度，因为中文/特殊字符的 token 长度不等于字符长度
        prompt_ids = self.tokenizer(
            item["prompt"],
            add_special_tokens=False,  # 不加 BOS/EOS，避免影响 prompt 长度计算
            truncation=True,           # 超过 max_length 时截断（从末尾）
            max_length=self.max_length,
        )["input_ids"]                 # tokenizer 返回 BatchEncoding（类字典），取其中的 token ID 列表

        full_ids = self.tokenizer(
            item["full_text"],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        # 返回的是 Python dict，值可以是 list（非 Tensor），Collator 负责后续转换
        return {
            "prompt_text": item["prompt"],    # 原始字符串，供调试查看
            "full_text": item["full_text"],   # 原始字符串
            "prompt_ids": prompt_ids,          # Python list[int]，prompt 的 token IDs
            "input_ids": full_ids,             # Python list[int]，完整序列的 token IDs
        }


# ------------------------------
# Collator
# ------------------------------
# ★ @dataclass 装饰器：Python 内置，自动生成 __init__、__repr__、__eq__
# 等价于手动写：
#   def __init__(self, tokenizer, max_length, ignore_index=IGNORE_INDEX, train_on_prompt=False):
#       self.tokenizer = tokenizer; self.max_length = max_length; ...
# 字段声明格式：字段名: 类型 = 默认值（有默认值的字段必须放在无默认值字段后面）
@dataclass
class SFTCollator:
    tokenizer: Any          # 分词器，用于获取 pad_token_id
    max_length: int         # 再次截断保险（Dataset 里已截过一次）
    ignore_index: int = IGNORE_INDEX   # 默认 -100，与 F.cross_entropy 保持一致
    train_on_prompt: bool = False      # False = 只对 response 计算 loss（更常用）

    # ★ __call__ 使对象可以像函数一样被调用：collator(list_of_samples)
    # DataLoader 的 collate_fn 参数正是调用此方法来将 sample 列表合并为 batch
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Build a batch with padding.

        Output:
          input_ids:      [B, T]
          attention_mask: [B, T]
          labels:         [B, T]

        Two modes:
          1. train_on_prompt=True  -> full-sequence LM loss
          2. train_on_prompt=False -> response-only loss
        """
        # 列表推导式：[表达式 for 变量 in 可迭代对象]，相当于 for 循环建列表
        # [:self.max_length] 切片截断，防止单条样本超长（Dataset 已截过，这里是双重保险）
        batch_input_ids = [f["input_ids"][: self.max_length] for f in features]
        batch_prompt_ids = [f["prompt_ids"][: self.max_length] for f in features]

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must have a pad_token_id.")

        # 找本 batch 内最长的序列，所有序列都补齐到这个长度（动态 padding）
        # max(...) + 生成器表达式：惰性求值，比列表推导式更省内存
        max_len_in_batch = max(len(x) for x in batch_input_ids)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        # zip(a, b)：同时迭代两个列表，每次取对应位置的元素
        for input_ids, prompt_ids in zip(batch_input_ids, batch_prompt_ids):
            seq_len = len(input_ids)                    # 真实序列长度
            pad_len = max_len_in_batch - seq_len        # 需要补多少个 pad token

            # Python list 拼接：[a] + [b]*n 生成右侧补 pad 的新列表
            # ★ 范式：right-padding（在右侧补 pad）对 Causal LM 更自然，配合 attention_mask 使用
            padded_input_ids = input_ids + [pad_id] * pad_len
            # attention_mask：1 表示真实 token，0 表示 padding（模型 attention 时忽略 0 的位置）
            attention_mask = [1] * seq_len + [0] * pad_len

            # labels 初始化为 input_ids 的副本（注意要用 .copy()，否则是引用同一列表！）
            labels = padded_input_ids.copy()

            # TODO-1:
            # If train_on_prompt is False, mask the prompt part with ignore_index.
            # Hint:
            #   prompt_token_len = min(len(prompt_ids), seq_len)
            #   labels[:prompt_token_len] = [self.ignore_index] * prompt_token_len
            if not self.train_on_prompt:
                # 为什么取 min？因为 full_text 被截断后，prompt 部分可能也被截掉一部分
                # 例：full_text 截到 256 tokens，但 prompt 本身就有 300 tokens
                # 此时 input_ids 里全是 prompt，没有 response，response 部分也不可见
                # 所以 prompt_token_len = min(len(prompt_ids), seq_len)
                prompt_token_len = min(len(prompt_ids), seq_len)
                # 切片赋值：将 labels 列表的前 prompt_token_len 个元素替换为 ignore_index
                labels[:prompt_token_len] = [self.ignore_index] * prompt_token_len

            # padding 位置也设为 ignore_index（避免模型被迫预测无意义的 pad token）
            for i in range(seq_len, max_len_in_batch):
                labels[i] = self.ignore_index

            # torch.tensor(list, dtype=...)：将 Python list 转为 Tensor
            # dtype=torch.long 即 int64，是 embedding 层和 cross_entropy 要求的索引类型
            input_ids_list.append(torch.tensor(padded_input_ids, dtype=torch.long))
            attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        # torch.stack(list_of_tensors, dim=0)：将多个形状相同的 1D Tensor 堆叠为 2D Tensor
        # 例：stack([[1,2,3], [4,5,6]], dim=0) -> [[1,2,3],[4,5,6]]，shape=[2,3]
        # 结果 shape: [batch_size, max_len_in_batch]
        batch = {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(attention_mask_list, dim=0),
            "labels": torch.stack(labels_list, dim=0),
        }
        return batch


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


def print_sequence_table(tokenizer, title: str, values: List[int], decode_tokens: bool = True):
    """把一维序列按 position / id / token 的形式打印出来，方便对齐观察。"""
    print(f"\n{title}")
    print("pos | value  | token")
    print("-" * 50)
    for pos, value in enumerate(values):
        if decode_tokens and value != IGNORE_INDEX:
            token_text = repr(tokenizer.decode([value], skip_special_tokens=False))
        else:
            token_text = "-"
        print(f"{pos:>3} | {value:>6} | {token_text}")


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
def evaluate(model, dataloader) -> Dict[str, float]:
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
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

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
    # Perplexity（困惑度）= exp(loss)，是语言模型的标准评估指标
    # 直觉：困惑度越低，模型对文本的预测越"不困惑"，越接近 1 越完美
    # avg_loss < 20 是防止 exp 溢出（e^20 ≈ 5 亿，超出一般范围则直接返回 inf）
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return {"loss": avg_loss, "ppl": ppl}


# ------------------------------
# Generation sanity check
# ------------------------------
@torch.no_grad()
def generate_preview(model, tokenizer, prompt: str, max_new_tokens: int = 80) -> str:
    model.eval()
    # return_tensors="pt"：让 tokenizer 直接返回 PyTorch Tensor 而非 Python list
    # 此时 inputs 是一个字典：{"input_ids": Tensor[1,T], "attention_mask": Tensor[1,T]}
    # .to(DEVICE)：BatchEncoding 对象支持 .to()，将其中所有 Tensor 移到目标设备
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # model.generate()：HF 封装的自回归生成接口，内部循环调用前向传播
    # **inputs：Python 字典解包，等价于 input_ids=..., attention_mask=...
    # max_new_tokens：最多生成多少个新 token（不含 prompt 本身）
    # do_sample=False：贪心解码（每步选 logit 最大的 token），输出确定性最强
    #   如果 do_sample=True 则按概率采样，输出更多样，可配合 temperature、top_p 等参数
    # pad_token_id：生成结束后需要知道 pad token，避免警告
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    # output_ids 形状：[1, prompt_len + new_tokens]（batch_size=1）
    # [0]：取第一个（也是唯一的）样本，得到 1D Tensor
    # tokenizer.decode()：将 token ID 列表转回字符串
    # skip_special_tokens=True：去掉 <|endoftext|> 等特殊 token，输出更干净
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ------------------------------
# Main training function
# ------------------------------
def main():
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
    # run_mini_batch_walkthrough(tokenizer)

    # ★ AutoModelForCausalLM.from_pretrained(name)：
    # 下载预训练权重并加载到内存。distilgpt2 约 82MB，首次运行自动下载并缓存
    # 此时模型权重是预训练好的，SFT 将在此基础上继续训练（而非从随机初始化开始）
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    # .to(DEVICE)：将模型所有参数和 buffer 移到指定设备（GPU/CPU）
    # 必须在创建 optimizer 之前完成，否则 optimizer 追踪的是 CPU 参数
    model.to(DEVICE)

    # ★ load_dataset(name) 返回 DatasetDict，类似 {"train": Dataset, "test": Dataset}
    # 数据首次下载后会缓存到 ~/.cache/huggingface/datasets/，后续秒加载
    raw_dataset = load_dataset(DATASET_NAME,"main")
    train_ds = raw_dataset["train"]  
    test_ds = raw_dataset["test"]    
    # TODO-3:
    # Split train_ds into train and validation sets.
    # For example:
    #   split = train_ds.train_test_split(test_size=0.02, seed=42)
    #   train_hf = split["train"]
    #   val_hf = split["test"]
    # .train_test_split()：HF Dataset 内置的划分方法
    # test_size=0.02 表示 2% 作为验证集；seed 固定随机性，保证可复现

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

    # ★ DataLoader 是 PyTorch 数据加载的核心组件：
    # - batch_size：每次取多少个样本
    # - shuffle=True：训练时打乱顺序，防止模型记住样本顺序，提升泛化
    # - collate_fn：将 Dataset.__getitem__ 返回的样本列表合并为 batch 的函数
    #   这里传入 collator 实例，DataLoader 会调用 collator(list_of_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,  # 验证集不需要打乱
        collate_fn=collator,
    )

    # ★ AdamW：Adam + Weight Decay 解耦（L2 正则化）
    # model.parameters()：返回模型所有可学习参数的迭代器
    # lr：初始学习率；weight_decay：L2 惩罚系数（防止过拟合）
    # AdamW 是 LLM 训练的标准优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 计算总步数：len(train_loader) = ceil(训练样本数 / batch_size)
    total_training_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(total_training_steps * WARMUP_RATIO)  # 前 3% 的步做 warmup
    # ★ LR 调度器：控制学习率随训练步数的变化
    # get_linear_schedule_with_warmup：
    #   [0, warmup_steps]：LR 从 0 线性增加到初始 LR
    #   [warmup_steps, total_steps]：LR 从初始 LR 线性减少到 0
    # 这是 BERT/GPT 系列 fine-tuning 的经典 LR 调度方式
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    global_step = 0
    model.train()  # 切换到训练模式（启用 Dropout 等）

    # 用于训练前后对比生成效果的固定 prompt
    demo_prompt = (
        "### Instruction:\n"
        "Explain what supervised fine-tuning is in simple words.\n\n"
        "### Response:\n"
    )

    print("\n=== Before training ===")
    print(generate_preview(model, tokenizer, demo_prompt))

    # ★ 标准 PyTorch 训练循环结构（必须掌握）：
    # for epoch in range(NUM_EPOCHS):
    #     for batch in train_loader:
    #         1. 数据移到设备
    #         2. 前向传播（Forward）
    #         3. 计算 loss
    #         4. optimizer.zero_grad()  ← 清空上一步的梯度
    #         5. loss.backward()        ← 反向传播，计算梯度
    #         6. （可选）梯度裁剪
    #         7. optimizer.step()       ← 根据梯度更新参数
    #         8. scheduler.step()       ← 更新学习率
    for epoch in range(NUM_EPOCHS):
        model.train()   # 每个 epoch 开始时确保处于训练模式（eval 后需要切回来）
        running_loss = 0.0

        # enumerate(iterable)：同时返回索引和元素，batch_idx 从 0 开始
        for batch_idx, batch in enumerate(train_loader):
            # 将 batch 字典中的所有 Tensor 移到 GPU/CPU
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # ---- Forward Pass ----
            # 注意：此处没有传 labels 参数给模型
            # HF 模型如果传了 labels 会自动计算 loss，但我们用自定义的 compute_causal_lm_loss
            # 以便更好地理解 loss 计算过程
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits  # shape: [B, T, V]
            loss = compute_causal_lm_loss(logits, batch["labels"])

            # ---- Backward Pass ----
            # ★ 顺序非常重要，不能搞错：
            # zero_grad() 必须在 backward() 之前调用，否则梯度会累加（默认行为）
            # 固定顺序：zero_grad → backward → (clip_grad) → step
            
            optimizer.zero_grad()  # 清空所有参数的 .grad 属性
            loss.backward()        # 反向传播：计算 loss 对所有参数的梯度，存入 param.grad

            # TODO-4:
            # Add gradient clipping if you want.
            # 梯度裁剪：防止梯度爆炸（特别是 Transformer 在长序列时容易出现）
            # max_norm=1.0 表示将所有参数梯度的全局 L2 范数裁剪到不超过 1.0
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()   # 根据 param.grad 和优化算法更新参数值
            scheduler.step()   # 更新学习率（必须在 optimizer.step() 之后调用）

            running_loss += loss.item()
            global_step += 1

            if global_step % LOG_EVERY == 0:
                avg_train_loss = running_loss / LOG_EVERY
                # :.4f 保留 4 位小数；:.6e 科学计数法保留 6 位有效数字
                print(
                    f"epoch={epoch} step={global_step} "
                    f"train_loss={avg_train_loss:.4f} lr={scheduler.get_last_lr()[0]:.6e}"
                )
                running_loss = 0.0  # 重置累积 loss，计算下一个窗口的平均值

            if global_step % EVAL_EVERY == 0:
                # evaluate() 内部已经调用了 model.eval()，结束后要切回训练模式
                metrics = evaluate(model, val_loader)
                print(
                    f"[eval] step={global_step} val_loss={metrics['loss']:.4f} "
                    f"val_ppl={metrics['ppl']:.4f}"
                )
                print("\n=== Generation preview ===")
                print(generate_preview(model, tokenizer, demo_prompt))
                print("==========================\n")
                model.train()  # 评估完毕，切回训练模式！

    print("\n=== After training ===")
    print(generate_preview(model, tokenizer, demo_prompt))

    # TODO-5:
    # Save model and tokenizer.
    # ★ HF 保存模型的标准方式：
    # model.save_pretrained(dir)：保存权重文件（pytorch_model.bin 或 model.safetensors）和配置
    # tokenizer.save_pretrained(dir)：保存词汇表和分词器配置
    # 之后可用 from_pretrained(dir) 从本地加载，无需联网
    # save_dir = "./phaseb_ckpt"
    # model.save_pretrained(save_dir)
    # tokenizer.save_pretrained(save_dir)


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
