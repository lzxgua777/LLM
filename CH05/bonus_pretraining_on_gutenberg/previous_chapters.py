# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# 这个文件收集了我们在第2章到第4章中所涵盖的所有相关代码。
# 这个文件可以作为一个独立的脚本运行。

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

#####################################
# 第2章
#####################################

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 使用tokenizer将文本编码为token ID序列
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

        # 将token ID序列分割成多个输入和目标序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # 返回数据集中的样本数量
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 根据索引返回一个样本
        return self.input_ids[idx], self.target_ids[idx]

# 创建数据加载器的函数
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

#####################################
# 第3章
#####################################

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 确保d_out可以被num_heads整除
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # 计算每个头的维度
        self.head_dim = d_out // num_heads

        # 定义查询（Q）、键（K）和值（V）的线性层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 定义输出层，用于合并头的输出
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # 创建一个上三角矩阵，用于后续的注意力掩码
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 计算查询（Q）、键（K）和值（V）
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 将矩阵按照头进行拆分
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置以适应多头注意力计算
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算缩放点积注意力（自注意力）并使用因果掩码
        attn_scores = queries @ keys.transpose(2, 3)

        # 将掩码截断到token数量，并转换为布尔值
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码填充注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 应用softmax和dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并头
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
#####################################
# 第4章
#####################################

# 层归一化（Layer Normalization）类
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除以0的一个小值
        # 学习缩放因子，维度与嵌入维度相同
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # 学习偏移因子，维度与嵌入维度相同
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # 计算x的均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 归一化x
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 应用缩放和偏移
        return self.scale * norm_x + self.shift

# GELU激活函数类
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # GELU激活函数的计算
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# 前馈网络（Feed Forward Network）类
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 构建前馈网络的两层线性层和GELU激活函数
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        # 前馈网络的前向传播
        return self.layers(x)

# Transformer块类
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 初始化多头自注意力层
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        # 初始化前馈网络
        self.ff = FeedForward(cfg)
        # 初始化两个层归一化
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        # 初始化用于快捷连接的dropout
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 自注意力块的快捷连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 前馈网络块的快捷连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

# GPT模型类
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 初始化词汇表的嵌入层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 初始化位置的嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 初始化嵌入层的dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 构建多个Transformer块组成的序列
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # 初始化最后的层归一化
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 初始化输出层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # 计算批次大小和序列长度
        batch_size, seq_len = in_idx.shape
        # 计算token嵌入
        tok_embeds = self.tok_emb(in_idx)
        # 计算位置嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 合并token和位置嵌入
        x = tok_embeds + pos_embeds
        # 应用嵌入层的dropout
        x = self.drop_emb(x)
        # 通过所有的Transformer块
        x = self.trf_blocks(x)
        # 应用最后的层归一化
        x = self.final_norm(x)
        # 通过输出层得到logits
        logits = self.out_head(x)
        return logits

# 简单的文本生成函数
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx是当前上下文的索引数组
    for _ in range(max_new_tokens):
        # 如果当前上下文超过支持的上下文大小，则裁剪
        idx_cond = idx[:, -context_size:]

        # 获取预测结果
        with torch.no_grad():
            logits = model(idx_cond)

        # 只关注最后一个时间步
        logits = logits[:, -1, :]

        # 获取具有最高logits值的词汇表索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # 将采样的索引追加到正在运行的序列
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
#####################################
# Chapter 5
####################################

# 计算单个批次的损失函数
def calc_loss_batch(input_batch, target_batch, model, device):
    # 将输入和目标数据移到指定设备（如GPU或CPU）
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 使用模型对输入进行前向传播，获取预测的logits
    logits = model(input_batch)
    # 计算交叉熵损失，logits需要展平成二维，目标值展平成一维
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss  # 返回单个批次的损失值


# 计算整个数据加载器的平均损失
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.  # 初始化总损失
    if len(data_loader) == 0:  # 如果数据加载器为空，返回NaN
        return float("nan")
    elif num_batches is None:  # 如果未指定批次数，则使用加载器的总批次数
        num_batches = len(data_loader)
    else:  # 如果指定了批次数，则取实际批次数和指定值的最小值
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):  # 遍历数据加载器中的批次
        if i < num_batches:  # 只计算指定数量的批次
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算单个批次的损失
            total_loss += loss.item()  # 累加损失值
        else:  # 如果达到指定批次数，则停止
            break
    return total_loss / num_batches  # 返回平均损失值


# 评估模型的性能（在训练集和验证集上计算损失）
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # 将模型切换到评估模式（禁用dropout等）
    with torch.no_grad():  # 禁用梯度计算
        # 计算训练集上的平均损失，使用指定的评估批次数
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        # 计算验证集上的平均损失
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # 重新切换回训练模式
    return train_loss, val_loss  # 返回训练损失和验证损失


# 生成文本并打印样例
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()  # 将模型切换到评估模式
    context_size = model.pos_emb.weight.shape[0]  # 获取模型上下文长度
    # 将起始文本编码为token ID张量，并移动到指定设备
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():  # 禁用梯度计算
        # 使用简单生成函数生成后续文本
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size)
        # 将生成的token ID解码回文本
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # 打印解码文本，替换换行符以更紧凑显示
    model.train()  # 重新切换回训练模式


# 绘制训练和验证损失的变化图
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, output_dir):
    fig, ax1 = plt.subplots()  # 创建一个子图（用于画图）

    # 绘制训练损失和验证损失随epoch变化的曲线
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")  # 设置x轴标签为"Epochs"
    ax1.set_ylabel("Loss")  # 设置y轴标签为"Loss"
    ax1.legend(loc="upper right")  # 在右上角添加图例
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # 确保x轴的刻度为整数

    # 创建第二个x轴，用于表示已处理的tokens数量
    ax2 = ax1.twiny()  # 创建共享y轴的第二个x轴
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 绘制一个透明曲线，用于对齐刻度
    ax2.set_xlabel("Tokens seen")  # 设置第二个x轴的标签

    fig.tight_layout()  # 调整布局以防止图形重叠
    plt.savefig(output_dir / "losses.pdf")  # 保存图像为PDF文件


# 将文本转换为token ID张量
def text_to_token_ids(text, tokenizer):
    # 使用tokenizer对文本进行编码，允许特殊字符如<|endoftext|>
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # 将编码后的token ID转换为张量，并添加batch维度
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor  # 返回带有batch维度的token ID张量


# 将token ID张量解码为文本
def token_ids_to_text(token_ids, tokenizer):
    # 去除batch维度，变为一维张量
    flat = token_ids.squeeze(0)
    # 使用tokenizer将token ID列表解码为文本
    return tokenizer.decode(flat.tolist())  # 返回解码后的文本
