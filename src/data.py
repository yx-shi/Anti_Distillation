from typing import Any, Dict, List
from dataclasses import dataclass
from torch.utils.data import DataLoader,Dataset
from config import *

def format_dolly_sample(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert Dolly-style record into prompt/completion.

    Dolly examples typically contain:
      - instruction
      - context
      - response

    We format them into a single instruction-following sample.
    """
    # 这份 demo 现在同时兼容两类常见 SFT 数据：
    # 1. Dolly 风格：instruction / context / response
    # 2. GSM8K 风格：question / answer
    #
    # 这样你后面换数据集时，不必立刻重写整条训练流水线。
    if "instruction" in example and "response" in example:
        instruction = example["instruction"].strip()
        context = (example.get("context") or "").strip()
        response = example["response"].strip()
    elif "question" in example and "answer" in example:
        instruction = example["question"].strip()
        context = ""
        response = example["answer"].strip()
    else:
        raise KeyError(
            "Unsupported dataset format. Expected Dolly fields "
            "('instruction', 'response') or GSM8K fields ('question', 'answer')."
        )

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