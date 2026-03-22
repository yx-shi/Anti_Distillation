import torch

# ------------------------------
# Configuration
# ------------------------------
# HF Hub 上模型的名字，格式为 "组织名/模型名" 或直接 "模型名"（官方模型）
# distilgpt2 是 GPT-2 的知识蒸馏小版本，参数量约 82M，本地 CPU 也能跑
MODEL_NAME = "/data1/public_checkpoints/Qwen3-1.7B"
# HF Hub 上数据集的名字，格式同上
DATASET_NAME = "openai/gsm8k"
# 每条序列的最大 token 数，超过会被截断。显存有限时调小此值
MAX_LENGTH = 128
# 训练时每个 batch 包含的样本数。越大越稳定但越占显存，一般 4~16
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
# 遍历整个训练集的轮数
NUM_EPOCHS = 1
# 学习率。SFT 常用 1e-5 ~ 5e-5；太大会破坏预训练权重，太小则收敛慢
LR = 5e-5
# AdamW 的 L2 正则化系数，防止过拟合
WEIGHT_DECAY = 0.01
# warmup 占总步数的比例。warmup 期间 LR 从 0 线性升到 LR，之后再线性降
# 作用：避免训练初期梯度过大，破坏预训练权重
WARMUP_RATIO = 0.03
# 每隔多少步打印一次训练 loss
LOG_EVERY = 20
# 每隔多少步做一次验证集评估
EVAL_EVERY = 200
# 自动检测是否有 GPU；SFT 用 GPU 速度可提升 10~100 倍
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# cross_entropy 的 ignore_index 参数：标签等于此值的位置不参与 loss 计算
# -100 是 PyTorch 的约定俗成值（F.cross_entropy 的默认 ignore_index 就是 -100）
IGNORE_INDEX = -100
# 用于标记 prompt 结束、response 开始的分隔符，Dolly 格式的约定
RESPONSE_PREFIX = "### Response:\n"