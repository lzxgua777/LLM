# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# "Build a Large Language Model From Scratch" 书中的代码
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码来源: https://github.com/rasbt/LLMs-from-scratch
#
# 这个文件收集了我们在第2到第5章中所涉及的所有相关代码

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch中的神经网络模块

#####################################
# Chapter 3
#####################################

# 定义多头自注意力机制（MultiHeadAttention）模块
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 确保输出维度d_out能被头数num_heads整除
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out  # 输出维度
        self.num_heads = num_heads  # 注意力头的数量
        self.head_dim = d_out // num_heads  # 每个头的维度，输出维度除以头数

        # 定义查询（Query）、键（Key）和值（Value）的线性映射层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 输出投影层，用于将多个注意力头的输出合并
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)  # Dropout层，用于防止过拟合
        # 注册掩码矩阵，用于自注意力中的因果掩码（防止未来信息泄露）
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # 获取输入的batch大小、token数目和输入维度

        # 通过线性层将输入x映射到查询、键和值
        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 将查询、键和值的维度调整以便并行计算多个头
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置维度，使得每个头的维度可以并行计算
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算缩放点积注意力（self-attention）
        attn_scores = queries @ keys.transpose(2, 3)  # 每个头的点积计算

        # 使用掩码将未来的信息置为负无穷，防止信息泄露
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # 截取合适的掩码矩阵
        attn_scores.masked_fill_(mask_bool, -torch.inf)  # 将掩码部分的注意力分数设为负无穷

        # 计算注意力权重，使用softmax归一化
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # 应用dropout，防止过拟合

        # 使用注意力权重加权值，计算上下文向量
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 将多个头的输出合并，并通过输出投影层进行映射
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 可选的输出投影层

        return context_vec  # 返回上下文向量


#####################################
# Chapter 4
#####################################

# 定义LayerNorm层（层归一化）
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除零的极小值
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可学习的缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的偏移参数

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # 计算均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算方差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 归一化
        return self.scale * norm_x + self.shift  # 缩放和平移


# 定义GELU激活函数
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # GELU激活函数的定义
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# 定义前馈网络（FeedForward）
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 定义前馈网络的层：一个线性层 + GELU激活 + 一个线性层
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 扩展维度到4倍
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 恢复维度
        )

    def forward(self, x):
        return self.layers(x)  # 返回经过前馈网络处理后的输出


# 定义Transformer块（包含自注意力和前馈网络）
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 初始化多头注意力、自注意力、前馈网络、LayerNorm和dropout层
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 自注意力模块：残差连接 + 注意力层 + Dropout
        shortcut = x
        x = self.norm1(x)  # 先进行LayerNorm
        x = self.att(x)  # 自注意力
        x = self.drop_shortcut(x)  # 应用dropout
        x = x + shortcut  # 残差连接

        # 前馈网络模块：残差连接 + 前馈层 + Dropout
        shortcut = x
        x = self.norm2(x)  # 进行LayerNorm
        x = self.ff(x)  # 前馈网络
        x = self.drop_shortcut(x)  # 应用dropout
        x = x + shortcut  # 残差连接

        return x  # 返回输出


# 定义GPT模型
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 初始化词嵌入层、位置嵌入层和dropout层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 词嵌入
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # Dropout层

        # 定义多个Transformer块
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]  # 根据层数创建多个Transformer块
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])  # 最后的LayerNorm层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 输出层，映射到词汇表大小

    def forward(self, in_idx):
        # 获取输入的批量大小（batch_size）和序列长度（seq_len）
        batch_size, seq_len = in_idx.shape

        # 通过嵌入层将输入的token索引映射到词嵌入（Embedding）
        tok_embeds = self.tok_emb(in_idx)

        # 生成位置索引，并通过位置嵌入层（pos_emb）映射到位置嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # 将词嵌入和位置嵌入相加，得到输入序列的嵌入表示
        x = tok_embeds + pos_embeds  # 形状：[batch_size, num_tokens, emb_size]

        # 对嵌入应用Dropout以减少过拟合
        x = self.drop_emb(x)

        # 将嵌入通过多个Transformer块处理
        x = self.trf_blocks(x)

        # 应用最后的层归一化（LayerNorm）
        x = self.final_norm(x)

        # 将最后的隐藏状态通过输出头（线性层），得到logits
        logits = self.out_head(x)

        # 返回logits，形状为：[batch_size, seq_len, vocab_size]
        return logits
