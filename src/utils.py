from typing import List,Any
import torch

# 这个工具函数在 phaseB_debug_small.py 中被用来格式化训练样本，生成 prompt 和 full_text。
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


# ------------------------------
# Generation sanity check
# ------------------------------
@torch.no_grad()
def generate_preview(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 80) -> str:
    model.eval()
    # return_tensors="pt"：让 tokenizer 直接返回 PyTorch Tensor 而非 Python list
    # 此时 inputs 是一个字典：{"input_ids": Tensor[1,T], "attention_mask": Tensor[1,T]}
    # .to(DEVICE)：BatchEncoding 对象支持 .to()，将其中所有 Tensor 移到目标设备
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # model.generate()：HF 封装的自回归生成接口，内部循环调用前向传播
    # **inputs：Python 字典解包，等价于 input_ids=..., attention_mask=...
    # max_new_tokens：最多生成多少个新 token（不含 prompt 本身）
    # do_sample=False：贪心解码（每步选 logit 最大的 token），输出确定性最强
    #   如果 do_sample=True 则按概率采样，输出更多样，可配合 temperature、top_p 等参数
    # pad_token_id：生成结束后需要知道 pad token，避免警告
    # FSDP 包装后，真正的 HF 模型在 model.module 里。
    # generate() 这类 HF 高层接口通常直接在底层模型对象上调用更稳妥。
    generate_model = model.module if hasattr(model, "module") else model

    output_ids = generate_model.generate(
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