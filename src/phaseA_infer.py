import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # 1) 指定模型来源：
    #    - Hugging Face Hub: "gpt2"
    #    - 服务器本地目录: "/path/to/local/gpt2"
    #
    # 只要目录里有 from_pretrained 所需文件（如 config.json、模型权重、
    # tokenizer.json / vocab.json / merges.txt 等），后续代码完全不用改。
    model_name_or_path = "/data1/public_checkpoints/Qwen3-1.7B"
    # 指定模型加载到哪个设备（GPU/CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) 加载 tokenizer（负责：文本 <-> token id 序列）
    # from_pretrained 会：
    #   - 传 Hub 名称时：优先读本地缓存，没有则从 Hugging Face Hub 下载
    #   - 传本地目录时：直接从该目录读取
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # 3) 加载模型（PyTorch nn.Module，即pytorch模型）
    # AutoModelForCausalLM：适配“自回归语言模型”的头（Causal LM head）
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
    )
    model.to(device)

    # 4) 切到 eval 模式：关闭 dropout 等训练时行为
    model.eval()

    # 5) 准备输入 prompt（你给模型的上下文）
    prompt = "Explain what knowledge distillation is in one sentence."

    # 6) tokenize：把字符串编码成模型可接受的张量 input_ids/attention_mask
    # return_tensors="pt" 表示返回 PyTorch tensor
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # -------- 核心点 A：generate() = 生成循环的封装 --------
    # generate 内部做了：
    #   (a) 用 inputs 做 forward，得到下一 token 的 logits
    #   (b) 按解码策略（greedy/sampling/beam）选 token
    #   (c) 把 token 拼回输入，重复 (a)(b)
    # 直到满足停止条件（max_new_tokens 或 EOS）
    #
    # do_sample=True：启用采样
    # top_k=50：只在概率最高的 50 个 token 里采样
    # temperature=0.8：缩放 logits（<1 更保守、更确定；>1 更发散）
    #
    # 这部分未来就是你 Anti-distillation decoding strategy 的“插刀口”
    with torch.no_grad():  # 推理阶段不需要梯度，节省显存/加速
        out_ids = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,  # 如果是false就是greedy
            top_k=10,
            temperature=0.8,
        )

    # 7) decode：把生成的 token ids 转回文本
    # out_ids[0] 是整段（prompt + new tokens）的 token 序列
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    print(f"=== Device: {device} ===")
    print("=== Generated ===")
    print(text)

if __name__ == "__main__":
    main()
