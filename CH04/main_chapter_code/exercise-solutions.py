# 从importlib.metadata模块导入version函数，用于获取库的版本信息
from importlib.metadata import version

# 导入torch库
import torch
# 打印torch库的版本信息
print("torch version:", version("torch"))

# 从gpt模块导入TransformerBlock类
from gpt import TransformerBlock

# 定义GPT模型配置参数
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 768,  # 嵌入维度
    "n_heads": 12,  # 注意力头数
    "n_layers": 12,  # 层数
    "drop_rate": 0.1,  # dropout率
    "qkv_bias": False  # QKV偏置
}

# 创建TransformerBlock实例
block = TransformerBlock(GPT_CONFIG_124M)

# 计算前馈网络模块的参数总数
total_params = sum(p.numel() for p in block.ff.parameters())
print(f"Total number of parameters in feed forward module: {total_params:,}")

# 计算注意力模块的参数总数
total_params = sum(p.numel() for p in block.att.parameters())
print(f"Total number of parameters in attention module: {total_params:,}")

# 重新定义GPT模型配置参数
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# 定义get_config函数，用于根据模型名称获取对应的配置参数
def get_config(base_config, model_name="gpt2-small"):
    GPT_CONFIG = base_config.copy()

    if model_name == "gpt2-small":
        GPT_CONFIG["emb_dim"] = 768
        GPT_CONFIG["n_layers"] = 12
        GPT_CONFIG["n_heads"] = 12

    elif model_name == "gpt2-medium":
        GPT_CONFIG["emb_dim"] = 1024
        GPT_CONFIG["n_layers"] = 24
        GPT_CONFIG["n_heads"] = 16

    elif model_name == "gpt2-large":
        GPT_CONFIG["emb_dim"] = 1280
        GPT_CONFIG["n_layers"] = 36
        GPT_CONFIG["n_heads"] = 20

    elif model_name == "gpt2-xl":
        GPT_CONFIG["emb_dim"] = 1600
        GPT_CONFIG["n_layers"] = 48
        GPT_CONFIG["n_heads"] = 25

    else:
        raise ValueError(f"Incorrect model name {model_name}")

    return GPT_CONFIG

# 定义calculate_size函数，用于计算模型的参数总数和大小
def calculate_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

    # 计算模型总大小（以字节为单位，假设float32，每个参数4字节）
    total_size_bytes = total_params * 4

    # 转换为兆字节
    total_size_mb = total_size_bytes / (1024 * 1024)

    print(f"Total size of the model: {total_size_mb:.2f} MB")

# 从gpt模块导入GPTModel类
from gpt import GPTModel

# 遍历不同的模型配置，创建模型实例并计算模型大小
for model_abbrev in ("small", "medium", "large", "xl"):
    model_name = f"gpt2-{model_abbrev}"
    CONFIG = get_config(GPT_CONFIG_124M, model_name=model_name)
    model = GPTModel(CONFIG)
    print(f"\n\n{model_name}:")
    calculate_size(model)

# 定义GPT模型配置参数，包含新的dropout配置
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate_emb": 0.1,        # 新增：嵌入层的dropout
    "drop_rate_attn": 0.1,       # 新增：多头自注意力的dropout
    "drop_rate_shortcut": 0.1,   # 新增：快捷连接的dropout
    "qkv_bias": False
}

# 导入torch.nn模块
import torch.nn as nn
# 从gpt模块导入MultiHeadAttention、LayerNorm和FeedForward类
from gpt import MultiHeadAttention, LayerNorm, FeedForward

# 定义TransformerBlock类，实现Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 初始化多头自注意力模块
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate_attn"], # 新增：多头自注意力的dropout
            qkv_bias=cfg["qkv_bias"])
        # 初始化前馈网络模块
        self.ff = FeedForward(cfg)
        # 初始化两个层归一化模块
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        # 初始化快捷连接的dropout模块
        self.drop_shortcut = nn.Dropout(cfg["drop_rate_shortcut"])

    def forward(self, x):
        # 注意力块的快捷连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # 将原始输入加回来

        # 前馈网络块的快捷连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 将原始输入加回来

        return x

# 定义GPTModel类，实现GPT模型
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 定义token嵌入层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 定义位置嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 定义嵌入层的dropout模块
        self.drop_emb = nn.Dropout(cfg["drop_rate_emb"]) # 新增：嵌入层的dropout

        # 使用TransformerBlock构建Transformer层的序列
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # 定义最终的层归一化
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 定义输出层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # 前向传播函数
        batch_size, seq_len = in_idx.shape
        # 计算token嵌入
        tok_embeds = self.tok_emb(in_idx)
        # 计算位置嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 合并token嵌入和位置嵌入
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        # 通过Transformer层
        x = self.trf_blocks(x)
        # 应用最终的层归一化
        x = self.final_norm(x)
        # 计算logits
        logits = self.out_head(x)
        return logits

# 设置随机种子以确保结果可复现
torch.manual_seed(123)
# 创建GPTModel实例
model = GPTModel(GPT_CONFIG_124M)


