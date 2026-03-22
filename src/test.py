import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

def test():
    # a=torch.tensor([[1,2,0.5,0.7]])
    # print(a[-1])
    # print("a:\n", a)
    # top_values, _ = torch.topk(a, k=2)  
    # #_表示我们不关心 topk 的索引，只要值就行了;另外从结果可以看出topk
    # # 会自动倒序排列，最大的值在前面，所以 top_values[..., -1] 就是第 k 大的值，也就是我们要的 threshold。
    # print("Top values:\n", top_values)
    # threshold = top_values[..., -1, None]
    # print("Threshold:\n", threshold)
    # filtered_logits = a.masked_fill(a < threshold, float("-inf"))    
    # print("Filtered logits:\n", filtered_logits)
    # logits=torch.tensor([[1,2,3],[2,1,5]],dtype=torch.float32)
    # labels=torch.tensor([2,2],dtype=torch.int64)
    # loss_fn=nn.CrossEntropyLoss()
    # loss=loss_fn(logits,labels)
    # print("loss:\n", loss)

    model_name_or_path = "/data1/public_checkpoints/Qwen3-1.7B"
    tokenizer=AutoTokenizer.from_pretrained(model_name_or_path)
    sample_input="Hello, how are you?"
    inputs=tokenizer(sample_input, return_tensors="pt")
    inputs2=tokenizer.encode(sample_input, return_tensors="pt")
    print("inputs:\n", inputs)
    print("inputs2:\n", inputs2)

if __name__ == "__main__":    
    test()