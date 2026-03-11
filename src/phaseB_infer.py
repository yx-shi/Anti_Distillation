import math
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


# ============================================================
# PhaseB: Minimal SFT demo without HF Trainer
# ------------------------------------------------------------
# Goals:
# 1. Keep tokenizer/model loading from Hugging Face.
# 2. Write Dataset / collate_fn / train loop / eval loop by hand.
# 3. Leave several TODOs for you to complete.
# ============================================================


# ------------------------------
# Configuration
# ------------------------------
MODEL_NAME = "distilgpt2"  # small and easy to run for a demo
DATASET_NAME = "databricks/databricks-dolly-15k"
MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
NUM_EPOCHS = 1
LR = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
LOG_EVERY = 20
EVAL_EVERY = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE_INDEX = -100
RESPONSE_PREFIX = "### Response:\n"


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

    completion = response
    full_text = prompt + completion

    return {
        "prompt": prompt,
        "completion": completion,
        "full_text": full_text,
    }


# ------------------------------
# Dataset
# ------------------------------
class SFTDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length: int):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw = self.dataset[idx]
        item = format_dolly_sample(raw)

        # Tokenize prompt and full_text separately.
        # Why?
        # Because later we want to support response-only loss,
        # which requires knowing the prompt length in token space.
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


# ------------------------------
# Collator
# ------------------------------
@dataclass
class SFTCollator:
    tokenizer: Any
    max_length: int
    ignore_index: int = IGNORE_INDEX
    train_on_prompt: bool = False

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
        batch_input_ids = [f["input_ids"][: self.max_length] for f in features]
        batch_prompt_ids = [f["prompt_ids"][: self.max_length] for f in features]

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must have a pad_token_id.")

        max_len_in_batch = max(len(x) for x in batch_input_ids)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for input_ids, prompt_ids in zip(batch_input_ids, batch_prompt_ids):
            seq_len = len(input_ids)
            pad_len = max_len_in_batch - seq_len

            padded_input_ids = input_ids + [pad_id] * pad_len
            attention_mask = [1] * seq_len + [0] * pad_len

            # Start from copying input_ids as labels.
            labels = padded_input_ids.copy()

            # TODO-1:
            # If train_on_prompt is False, mask the prompt part with ignore_index.
            # Hint:
            #   prompt_token_len = min(len(prompt_ids), seq_len)
            #   labels[:prompt_token_len] = [self.ignore_index] * prompt_token_len
            if not self.train_on_prompt:
                prompt_token_len = min(len(prompt_ids), seq_len)
                labels[:prompt_token_len] = [self.ignore_index] * prompt_token_len

            # Always mask padding positions.
            for i in range(seq_len, max_len_in_batch):
                labels[i] = self.ignore_index

            input_ids_list.append(torch.tensor(padded_input_ids, dtype=torch.long))
            attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

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
    logits: [B, T, V]
    labels: [B, T]

    For causal LM, token at position t predicts token at position t+1.
    So we shift:
      shift_logits = logits[:, :-1, :]
      shift_labels = labels[:, 1:]
    """
    # TODO-2:
    # Complete the shifting and cross-entropy computation.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
    return loss


# ------------------------------
# Eval
# ------------------------------
@torch.no_grad()
def evaluate(model, dataloader) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_steps = 0

    for batch in dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits
        loss = compute_causal_lm_loss(logits, batch["labels"])

        total_loss += loss.item()
        total_steps += 1

    avg_loss = total_loss / max(total_steps, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return {"loss": avg_loss, "ppl": ppl}


# ------------------------------
# Generation sanity check
# ------------------------------
@torch.no_grad()
def generate_preview(model, tokenizer, prompt: str, max_new_tokens: int = 80) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ------------------------------
# Main training function
# ------------------------------
def main():
    print(f"Using device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        # Common trick for GPT-style tokenizers.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    # Load dataset
    raw_dataset = load_dataset(DATASET_NAME)
    train_ds = raw_dataset["train"]

    # TODO-3:
    # Split train_ds into train and validation sets.
    # For example:
    #   split = train_ds.train_test_split(test_size=0.02, seed=42)
    #   train_hf = split["train"]
    #   val_hf = split["test"]
    split = train_ds.train_test_split(test_size=0.02, seed=42)
    train_hf = split["train"]
    val_hf = split["test"]

    train_dataset = SFTDataset(train_hf, tokenizer, MAX_LENGTH)
    val_dataset = SFTDataset(val_hf, tokenizer, MAX_LENGTH)

    collator = SFTCollator(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        ignore_index=IGNORE_INDEX,
        train_on_prompt=False,  # response-only loss
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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

    print("\n=== Before training ===")
    print(generate_preview(model, tokenizer, demo_prompt))

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Forward
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            loss = compute_causal_lm_loss(logits, batch["labels"])

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # TODO-4:
            # Add gradient clipping if you want.
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % LOG_EVERY == 0:
                avg_train_loss = running_loss / LOG_EVERY
                print(
                    f"epoch={epoch} step={global_step} "
                    f"train_loss={avg_train_loss:.4f} lr={scheduler.get_last_lr()[0]:.6e}"
                )
                running_loss = 0.0

            if global_step % EVAL_EVERY == 0:
                metrics = evaluate(model, val_loader)
                print(
                    f"[eval] step={global_step} val_loss={metrics['loss']:.4f} "
                    f"val_ppl={metrics['ppl']:.4f}"
                )
                print("\n=== Generation preview ===")
                print(generate_preview(model, tokenizer, demo_prompt))
                print("==========================\n")
                model.train()

    print("\n=== After training ===")
    print(generate_preview(model, tokenizer, demo_prompt))

    # TODO-5:
    # Save model and tokenizer.
    # save_dir = "./phaseb_ckpt"
    # model.save_pretrained(save_dir)
    # tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
