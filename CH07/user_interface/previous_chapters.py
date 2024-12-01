# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-5.
import json  # 导入json库，用于处理JSON数据
import os  # 导入os库，用于操作文件和目录
import urllib  # 导入urllib库，用于处理URL

import numpy as np  # 导入numpy库，用于数值计算
import tensorflow as tf  # 导入tensorflow库，用于深度学习
import torch  # 导入PyTorch库，用于深度学习
import torch.nn as nn  # 导入PyTorch的神经网络模块
from tqdm import tqdm  # 导入tqdm库，用于显示进度条


#####################################Chapter 3#####################################
class MultiHeadAttention(nn.Module):
    # 定义一个多头注意力机制类，继承自PyTorch的nn.Module
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()  # 调用父类的初始化方法
        assert d_out % num_heads == 0, "d_out必须能被num_heads整除"  # 确保d_out能被num_heads整除

        self.d_out = d_out  # 输出维度
        self.num_heads = num_heads  # 注意力头的数量
        self.head_dim = d_out // num_heads  # 每个头的维度

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 查询的线性变换
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # 键的线性变换
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 值的线性变换
        self.out_proj = nn.Linear(d_out, d_out)  # 输出的线性变换，用于合并头的输出
        self.dropout = nn.Dropout(dropout)  # dropout层
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))  # 注册一个上三角掩码，用于防止未来信息的泄露

    def forward(self, x):
        # 前向传播函数
        b, num_tokens, d_in = x.shape  # 获取输入的批大小、token数量和维度

        keys = self.W_key(x)  # 计算键
        queries = self.W_query(x)  # 计算查询
        values = self.W_value(x)  # 计算值

        # 将矩阵按num_heads维度分割
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置以适应多头注意力的计算
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算缩放点积注意力（自注意力）并使用因果掩码
        attn_scores = queries @ keys.transpose(2, 3)  # 对每个头进行点积

        # 将掩码截断到token数量，并转换为布尔型
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码填充注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)  # 计算注意力权重
        attn_weights = self.dropout(attn_weights)  # 应用dropout

        # 计算上下文向量
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并头
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 可选的投影

        return context_vec  # 返回上下文向量

#####################################
# Chapter 4
#####################################

class LayerNorm(nn.Module):
    """
    自定义的层归一化（Layer Normalization）模块。
    """
    def __init__(self, emb_dim):
        """
        初始化，定义可学习的缩放参数 scale 和偏移参数 shift。
        :param emb_dim: 嵌入维度。
        """
        super().__init__()
        self.eps = 1e-5  # 防止除以零的小偏差值
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可学习的缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的偏移参数

    def forward(self, x):
        """
        前向传播：对输入张量进行归一化。
        :param x: 输入张量，形状为 [batch_size, num_tokens, emb_dim]。
        """
        mean = x.mean(dim=-1, keepdim=True)  # 按最后一个维度计算均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 按最后一个维度计算方差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 归一化处理
        return self.scale * norm_x + self.shift  # 应用缩放和偏移


class GELU(nn.Module):
    """
    自定义 GELU 激活函数。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        前向传播：应用 GELU 激活公式。
        :param x: 输入张量。
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *  # 公式中的常数
            (x + 0.044715 * torch.pow(x, 3))  # 带有三次项的非线性
        ))


class FeedForward(nn.Module):
    """
    前馈神经网络模块。
    """
    def __init__(self, cfg):
        """
        初始化，定义两层全连接网络和激活函数。
        :param cfg: 配置字典，包含嵌入维度（emb_dim）。
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 第一个全连接层，扩大维度
            GELU(),  # GELU 激活函数
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 第二个全连接层，缩小回原维度
        )

    def forward(self, x):
        """
        前向传播：通过前馈网络处理输入。
        :param x: 输入张量。
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    单个 Transformer 块，包括注意力机制、前馈网络和归一化。
    """
    def __init__(self, cfg):
        """
        初始化 Transformer 块。
        :param cfg: 配置字典，包含嵌入维度、上下文长度、头数等信息。
        """
        super().__init__()
        # 多头注意力机制
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],  # 输入维度
            d_out=cfg["emb_dim"],  # 输出维度
            context_length=cfg["context_length"],  # 上下文长度
            num_heads=cfg["n_heads"],  # 注意力头数
            dropout=cfg["drop_rate"],  # Dropout 概率
            qkv_bias=cfg["qkv_bias"])  # 是否添加 QKV 偏置
        self.ff = FeedForward(cfg)  # 前馈网络
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 第一层归一化
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 第二层归一化
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # Dropout 层

    def forward(self, x):
        """
        前向传播：通过注意力和前馈块处理输入，同时添加残差连接。
        :param x: 输入张量。
        """
        # 注意力机制的残差连接
        shortcut = x  # 保存输入
        x = self.norm1(x)  # 层归一化
        x = self.att(x)  # 多头注意力
        x = self.drop_shortcut(x)  # Dropout
        x = x + shortcut  # 加回原始输入（残差连接）

        # 前馈网络的残差连接
        shortcut = x  # 保存中间结果
        x = self.norm2(x)  # 层归一化
        x = self.ff(x)  # 前馈网络
        x = self.drop_shortcut(x)  # Dropout
        x = x + shortcut  # 加回原始输入（残差连接）

        return x


class GPTModel(nn.Module):
    """
    基于 Transformer 的 GPT 模型。
    """
    def __init__(self, cfg):
        """
        初始化 GPT 模型。
        :param cfg: 配置字典，包含嵌入维度、词汇表大小等参数。
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 词嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入层
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # Dropout 层

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])  # 多层 Transformer 块

        self.final_norm = LayerNorm(cfg["emb_dim"])  # 最终的归一化层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 输出层（线性分类器）

    def forward(self, in_idx):
        """
        前向传播：将输入索引转换为嵌入，经过 Transformer 块和线性分类器，生成 logits。
        :param in_idx: 输入索引张量，形状 [batch_size, seq_len]。
        """
        batch_size, seq_len = in_idx.shape  # 获取批量大小和序列长度
        tok_embeds = self.tok_emb(in_idx)  # 获取词嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 获取位置嵌入
        x = tok_embeds + pos_embeds  # 将词嵌入和位置嵌入相加
        x = self.drop_emb(x)  # 应用 Dropout
        x = self.trf_blocks(x)  # 通过 Transformer 块
        x = self.final_norm(x)  # 最后归一化
        logits = self.out_head(x)  # 生成输出 logits
        return logits  # 返回 logits

#####################################
# Chapter 5
#####################################
def text_to_token_ids(text, tokenizer):
    # 使用tokenizer将文本编码为token ids
    encoded = tokenizer.encode(text)
    # 将编码后的token ids转换为PyTorch张量，并添加批次维度
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    # 移除批次维度，将token ids转换回文本
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def download_and_load_gpt2(model_size, models_dir):
    # 验证模型大小是否合法
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # 定义模型路径
    model_dir = os.path.join(models_dir, model_size)
    # 定义模型下载的基础URL
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 下载文件
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    # 加载模型设置和参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params

def download_file(url, destination):
    # 发送GET请求以下载文件
    with urllib.request.urlopen(url) as response:
        # 从响应头中获取文件总大小，默认为0
        file_size = int(response.headers.get("Content-Length", 0))

        # 检查文件是否存在且大小相同
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"文件已存在且是最新的: {destination}")
                return

        # 定义读取文件的块大小
        block_size = 1024  # 1 Kilobyte

        # 使用文件总大小初始化进度条
        progress_bar_description = os.path.basename(url)
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            # 以二进制写模式打开目标文件
            with open(destination, "wb") as file:
                # 按块读取文件并写入目标文件
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    file.write(chunk)
                    progress_bar.update(len(chunk))  # 更新进度条

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # 使用空块为每层初始化参数字典
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 遍历检查点中的每个变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载变量并移除单例维度
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 处理变量名以提取相关信息
        variable_name_parts = name.split("/")[1:]  # 跳过 'model/' 前缀

        # 确定变量的目标字典
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # 递归访问或创建嵌套字典
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # 将变量数组分配给最后一个键
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    使用 GPT 模型生成文本序列。
    :param model: GPT 模型实例。
    :param idx: 输入序列的索引张量，形状为 [batch_size, seq_len]。
    :param max_new_tokens: 生成的新 token 的最大数量。
    :param context_size: 上下文窗口的大小。
    :param temperature: 温度系数，用于控制随机性（默认值为 0，表示贪心解码）。
    :param top_k: 选择前 k 个概率最高的 token 进行采样。
    :param eos_id: 序列结束的特殊 token ID。
    """
    for _ in range(max_new_tokens):
        # 获取上下文窗口
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            # 计算模型的 logits
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # 取最后一个时间步的 logits

        # 如果指定了 top_k，进行 Top-K 采样
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)  # 获取 top_k 最大值
            min_val = top_logits[:, -1]  # 获取 top_k 中的最小值
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),  # 剔除低概率值
                logits)

        # 如果温度大于 0，进行概率采样
        if temperature > 0.0:
            logits = logits / temperature  # 调整 logits 的分布
            probs = torch.softmax(logits, dim=-1)  # 转为概率分布
            idx_next = torch.multinomial(probs, num_samples=1)  # 从分布中采样
        else:
            # 否则选择最高概率的 token（贪心解码）
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # 如果遇到结束标志且指定了 eos_id，提前结束
        if idx_next == eos_id:
            break

        # 将新生成的 token 拼接到序列中
        idx = torch.cat((idx, idx_next), dim=1)

    return idx  # 返回生成的序列
