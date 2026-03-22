from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel


# ------------------------------------------------------------
# 一个“适合断点调试”的超小 SFT demo
# ------------------------------------------------------------
# 设计目标：
# 1. 仍然使用 HF tokenizer 和 HF model 类
# 2. 但模型本身缩得很小，便于单步调试
# 3. 数据集只用手写的几条样本，避免长序列和大数据集干扰理解
# 4. 默认放在 CPU 上，适合在 IDE 里打断点观察张量内容
# ------------------------------------------------------------


TOKENIZER_NAME_OR_PATH = "/data1/public_checkpoints/Qwen3-1.7B"
MAX_LENGTH = 48
BATCH_SIZE = 2
NUM_STEPS = 3
LR = 1e-3
IGNORE_INDEX = -100
DEVICE = "cpu"
RESPONSE_PREFIX = "### Response:\n"


def format_sample(example: Dict[str, str]) -> Dict[str, str]:
    instruction = example["instruction"].strip()
    context = (example.get("context") or "").strip()
    response = example["response"].strip()

    if context:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{context}\n\n"
            f"{RESPONSE_PREFIX}"
        )
    else:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            f"{RESPONSE_PREFIX}"
        )

    full_text = prompt + response
    return {"prompt": prompt, "full_text": full_text}


class TinySFTDataset(Dataset):
    def __init__(self, examples: List[Dict[str, str]], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = format_sample(self.examples[idx])
        prompt_ids = self.tokenizer(
            item["prompt"],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        full_ids = self.tokenizer(
            item["full_text"],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        return {
            "prompt_text": item["prompt"],
            "full_text": item["full_text"],
            "prompt_ids": prompt_ids,
            "input_ids": full_ids,
        }


@dataclass
class TinySFTCollator:
    tokenizer: Any
    max_length: int
    ignore_index: int = IGNORE_INDEX
    train_on_prompt: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [f["input_ids"][: self.max_length] for f in features]
        batch_prompt_ids = [f["prompt_ids"][: self.max_length] for f in features]

        pad_id = self.tokenizer.pad_token_id
        max_len_in_batch = max(len(x) for x in batch_input_ids)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for input_ids, prompt_ids in zip(batch_input_ids, batch_prompt_ids):
            seq_len = len(input_ids)
            pad_len = max_len_in_batch - seq_len

            padded_input_ids = input_ids + [pad_id] * pad_len
            attention_mask = [1] * seq_len + [0] * pad_len

            labels = padded_input_ids.copy()
            if not self.train_on_prompt:
                prompt_token_len = min(len(prompt_ids), seq_len)
                labels[:prompt_token_len] = [self.ignore_index] * prompt_token_len
            labels[seq_len:] = [self.ignore_index] * pad_len

            input_ids_list.append(torch.tensor(padded_input_ids, dtype=torch.long))
            attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        return {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(attention_mask_list, dim=0),
            "labels": torch.stack(labels_list, dim=0),
        }


def compute_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    valid_mask = flat_labels.ne(IGNORE_INDEX)
    valid_count = valid_mask.sum()
    if valid_count.item() == 0:
        return flat_logits.sum() * 0.0

    return F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    ) / valid_count


def print_first_sample(tokenizer, batch: Dict[str, torch.Tensor]):
    input_ids = batch["input_ids"][0].tolist()
    attention_mask = batch["attention_mask"][0].tolist()
    labels = batch["labels"][0].tolist()
    shift_labels = batch["labels"][0, 1:].tolist()

    print("\n=== First Batch Sample ===")
    print("pos | input_id | attention | label   | shift_label | token")
    print("-" * 80)
    max_rows = len(input_ids)
    for pos in range(max_rows):
        token_text = repr(tokenizer.decode([input_ids[pos]], skip_special_tokens=False))
        label_text = labels[pos]
        shift_label_text = shift_labels[pos] if pos < len(shift_labels) else "-"
        print(
            f"{pos:>3} | {input_ids[pos]:>8} | {attention_mask[pos]:>9} | "
            f"{str(label_text):>7} | {str(shift_label_text):>11} | {token_text}"
        )


def main():
    torch.manual_seed(42)
    print(f"Using device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME_OR_PATH,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 这里故意不用大模型权重，而是直接构造一个很小的 HF GPT2 模型。
    # 这样你在 debug 时看到的参数、hidden states、logits 都更容易理解。
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=MAX_LENGTH,
        n_ctx=MAX_LENGTH,
        n_embd=64,
        n_layer=2,
        n_head=2,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config).to(DEVICE)

    examples = [
        {
            "instruction": "Answer with one word: the opposite of cold.",
            "context": "",
            "response": "Hot.",
        },
        {
            "instruction": "Complete the sentence.",
            "context": "The capital of France is",
            "response": "Paris.",
        },
        {
            "instruction": "Answer yes or no.",
            "context": "Is water wet?",
            "response": "Yes.",
        },
        {
            "instruction": "Finish the phrase.",
            "context": "2 + 2 =",
            "response": "4.",
        },
    ]

    dataset = TinySFTDataset(examples, tokenizer, MAX_LENGTH)
    collator = TinySFTCollator(tokenizer=tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 先看一个 batch 的具体长相，方便打断点逐行对应。
    first_batch = next(iter(dataloader))
    print(f"input_ids shape      = {tuple(first_batch['input_ids'].shape)}")
    print(f"attention_mask shape = {tuple(first_batch['attention_mask'].shape)}")
    print(f"labels shape         = {tuple(first_batch['labels'].shape)}")
    print_first_sample(tokenizer, first_batch)

    print("\n=== Tiny Training Loop ===")
    step = 0
    while step < NUM_STEPS:
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            loss = compute_causal_lm_loss(logits, batch["labels"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"step={step} loss={loss.item():.4f} "
                f"logits_shape={tuple(logits.shape)}"
            )

            step += 1
            if step >= NUM_STEPS:
                break


if __name__ == "__main__":
    main()
