import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_token(tokenizer, token_id):
    """把单个 token id 转成更容易观察的字符串。"""
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    return repr(text)


def print_top_tokens(tokenizer, logits, title, topn=10):
    """
    给定一维 logits，打印 top-n token 及其 softmax 概率。

    注意：
    - 这里打印的概率，是“对当前这组 logits 做 softmax 之后”的结果。
    - 如果 logits 里有 -inf（例如 top-k 过滤后），softmax 会把对应 token 的概率压成 0。
    """
    probs = torch.softmax(logits, dim=-1)
    k = min(topn, logits.shape[-1])
    top_probs, top_ids = torch.topk(probs, k=k)

    print(f"\n[{title}]")
    for rank, (token_id, prob) in enumerate(zip(top_ids.tolist(), top_probs.tolist()), start=1):
        token_text = format_token(tokenizer, token_id)
        print(
            f"{rank:>2}. token_id={token_id:<6} "
            f"token={token_text:<20} prob={prob:.6f}"
        )


def apply_top_k(logits, top_k):
    """
    只保留 logits 最高的 top_k 个 token，其余位置设为 -inf。

    这样再做 softmax 时，概率质量只会分配给 top_k 集合中的 token。
    """
    if top_k is None or top_k <= 0 or top_k >= logits.shape[-1]:
        return logits

    top_values, _ = torch.topk(logits, k=top_k)
    threshold = top_values[..., -1, None]
    filtered_logits = logits.masked_fill(logits < threshold, float("-inf"))
    return filtered_logits


def main():
    # 1) 指定模型来源：
    #    - Hugging Face Hub: "gpt2"
    #    - 服务器本地目录: "/data1/public_checkpoints/Qwen3-1.7B"
    model_name_or_path = "/data1/public_checkpoints/Qwen3-1.7B"

    # 2) 指定运行设备。
    #    如果服务器能看到 GPU，就走 CUDA；否则自动退回 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3) 加载 tokenizer 和模型。
    #    AutoModelForCausalLM 返回的是一个 PyTorch nn.Module。
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
    )
    model.to(device)
    model.eval()

    # 4) 准备 prompt。
    prompt = "Explain what knowledge distillation is in one sentence."

    # 5) 这是我们手写的“逐步生成”参数。
    #    这里特意不用 model.generate()，而是自己写循环，方便观察每一步发生了什么。
    max_new_tokens = 10
    temperature = 0.8
    top_k = 10
    do_sample = True

    # 6) tokenize 后得到一个字典，通常至少包含：
    #    - input_ids: token id 序列
    #    - attention_mask: 哪些位置是真实 token，哪些位置是 padding
    #
    #    这里的 **inputs 在函数调用里表示“把字典拆成关键字参数”。
    #    例如 model(**inputs) 等价于：
    #    model(input_ids=..., attention_mask=...)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 7) current_input_ids / current_attention_mask 表示“当前生成到哪一步的完整上下文”。
    #    一开始它们只包含原始 prompt；每生成一个新 token，就把它拼到末尾。
    current_input_ids = inputs["input_ids"]
    current_attention_mask = inputs["attention_mask"]

    # 8) 用来记录新生成的 token，最后统一 decode。
    generated_token_ids = []

    print(f"=== Device: {device} ===")
    print("=== Prompt ===")
    print(prompt)
    print("\n=== Manual Decoding Trace ===")

    # 9) 手写生成循环。
    #    每次循环做的事情等价于 generate() 内部的一步：
    #    (a) forward -> 得到所有位置的 logits
    #    (b) 取最后一个位置的 logits -> 这就是“下一个 token”的分布
    #    (c) temperature / top-k 处理
    #    (d) 采样或贪心选出下一个 token
    #    (e) 把新 token 拼回输入，进入下一轮
    #
    #    为了教学清晰，这里每一步都重新喂完整序列做 forward。
    #    真正高效的 generate() 通常会配合 KV cache，避免每步都全量重算。
    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
            )

            # outputs.logits 形状通常是 [batch_size, seq_len, vocab_size]
            # 这里 batch_size=1，所以取 [0, -1, :]：
            # - 0 表示第一个样本
            # - -1 表示当前序列最后一个位置
            # - : 表示这个位置上对整个词表的打分
            next_token_logits = outputs.logits[0, -1, :]

            print(f"\n========== Step {step + 1} ==========")

            # A. 原始 logits 的 top-10
            #    注意这里的概率是“原始 logits 直接 softmax”后的结果。
            print_top_tokens(
                tokenizer,
                next_token_logits,
                title="Raw logits -> softmax top-10",
                topn=10,
            )

            # B. temperature 的作用：
            #    new_logits = logits / temperature
            #    - temperature < 1：分布更尖锐，头部 token 更占优
            #    - temperature > 1：分布更平坦，更随机
            tempered_logits = next_token_logits / temperature
            print_top_tokens(
                tokenizer,
                tempered_logits,
                title=f"After temperature={temperature} -> softmax top-10",
                topn=10,
            )

            # C. top-k 的作用：
            #    只保留分数最高的 k 个 token，其余 token 概率变成 0。
            topk_logits = apply_top_k(tempered_logits, top_k=top_k)
            print_top_tokens(
                tokenizer,
                topk_logits,
                title=f"After top_k={top_k} filtering -> softmax top-10",
                topn=10,
            )

            # D. 根据处理后的分布选下一个 token。
            #    - do_sample=True: 按概率随机采样
            #    - do_sample=False: 直接取概率最大的 token（greedy）
            filtered_probs = torch.softmax(topk_logits, dim=-1)
            if do_sample:
                next_token_id = torch.multinomial(filtered_probs, num_samples=1)
            else:
                next_token_id = torch.argmax(filtered_probs, dim=-1, keepdim=True)

            selected_token_id = next_token_id.item()
            selected_token_text = format_token(tokenizer, selected_token_id)
            selected_token_prob = filtered_probs[selected_token_id].item()

            print(
                f"\n[Selected token] token_id={selected_token_id} "
                f"token={selected_token_text} prob={selected_token_prob:.6f}"
            )

            generated_token_ids.append(selected_token_id)

            # E. 把新 token 拼到输入末尾，作为下一步的上下文。
            next_token_id = next_token_id.view(1, 1)
            current_input_ids = torch.cat([current_input_ids, next_token_id], dim=1)

            # 新的attention_mask末端补1，表示新 token 也是有效输入。
            # dtype的意思是“整数类型”，device的意思是“和 current_attention_mask 在同一设备上”，这样拼接时才不会出问题。
            next_attention = torch.ones((1, 1), dtype=current_attention_mask.dtype, device=device)
            current_attention_mask = torch.cat([current_attention_mask, next_attention], dim=1)

            # F. 如果模型定义了 EOS token，且当前生成到了 EOS，就提前结束。
            if tokenizer.eos_token_id is not None and selected_token_id == tokenizer.eos_token_id:
                print("\n[Stop] Encountered EOS token.")
                break

    # 10) 最终文本：
    #     - full_text: prompt + 新生成内容
    #     - new_text: 只看新生成的部分
    full_text = tokenizer.decode(current_input_ids[0], skip_special_tokens=True)
    new_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    print("\n=== Final Output ===")
    print(full_text)

    print("\n=== Newly Generated Part ===")
    print(new_text)


if __name__ == "__main__":
    main()
