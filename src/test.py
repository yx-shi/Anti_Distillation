import torch

def test():
    a=torch.tensor([[1,2,0.5,0.7]])
    print(a[-1])
    print("a:\n", a)
    top_values, _ = torch.topk(a, k=2)  
    #_表示我们不关心 topk 的索引，只要值就行了;另外从结果可以看出topk
    # 会自动倒序排列，最大的值在前面，所以 top_values[..., -1] 就是第 k 大的值，也就是我们要的 threshold。
    print("Top values:\n", top_values)
    threshold = top_values[..., -1, None]
    print("Threshold:\n", threshold)
    filtered_logits = a.masked_fill(a < threshold, float("-inf"))    
    print("Filtered logits:\n", filtered_logits)

if __name__ == "__main__":    
    test()