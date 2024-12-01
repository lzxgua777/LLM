# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-5.
import json
import os
import urllib

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from tqdm import tqdm


#####################################
# 第3章
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 确保输出维度能够被头数整除
        assert d_out % num_heads == 0, "d_out必须能够被num_heads整除"

        # 保存参数
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 将输出维度减少到每个头的维度

        # 定义将输入映射为查询、键、值的线性变换
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 输出映射层，将多个头的结果结合成一个输出
        self.out_proj = nn.Linear(d_out, d_out)
        # Dropout层，用于正则化
        self.dropout = nn.Dropout(dropout)
        # 注册缓冲区，创建上三角的掩码（用于自回归）
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # 获取批量大小、token数和输入维度

        # 计算键、查询和值
        keys = self.W_key(x)  # 形状：(b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 重塑矩阵，将最后的维度分配到多个头上
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置矩阵： (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算缩放点积注意力分数
        attn_scores = queries @ keys.transpose(2, 3)  # 计算每个头的点积

        # 应用因果掩码（防止看到未来的token）
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 对注意力分数进行归一化
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # 对注意力权重应用Dropout
        attn_weights = self.dropout(attn_weights)

        # 使用注意力权重对值进行加权求和
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 重塑回原始维度，将多个头的结果合并
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        # 通过输出映射层
        context_vec = self.out_proj(context_vec)

        return context_vec


#####################################
# 第4章
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 小的数值，用于数值稳定性
        # 学习的缩放和偏移参数
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # 计算输入的均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 对输入进行归一化
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 应用缩放和偏移
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # GELU激活函数
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 定义一个简单的前馈网络，包括两层线性变换和GELU激活函数
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 第一层将维度扩展4倍
            GELU(),  # GELU激活函数
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 第二层将维度恢复到原始值
        )

    def forward(self, x):
        # 将输入通过前馈网络层
        return self.layers(x)


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

        # 初始化前馈网络层
        self.ff = FeedForward(cfg)

        # 初始化层归一化层
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # 用于残差连接的Dropout
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 注意力块的残差连接
        shortcut = x  # 保存输入，用于残差连接
        x = self.norm1(x)  # 对输入应用第一层归一化
        x = self.att(x)  # 应用多头自注意力
        x = self.drop_shortcut(x)  # 对注意力输出应用Dropout
        x = x + shortcut  # 将输入加回到输出（残差连接）

        # 前馈块的残差连接
        shortcut = x  # 保存输入，用于残差连接
        x = self.norm2(x)  # 对输入应用第二层归一化
        x = self.ff(x)  # 应用前馈网络
        x = self.drop_shortcut(x)  # 对前馈输出应用Dropout
        x = x + shortcut  # 将输入加回到输出（残差连接）

        return x


class GPTModel(nn.Module):  # 定义GPT模型类，继承自PyTorch的nn.Module
    def __init__(self, cfg):  # 初始化函数，接受一个配置字典cfg
        super().__init__()  # 调用父类的初始化函数
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 定义词嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 定义位置嵌入层
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # 定义Dropout层

        self.trf_blocks = nn.Sequential(  # 定义一个Sequential容器来存储Transformer块
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])  # 根据配置创建多个Transformer块

        self.final_norm = LayerNorm(cfg["emb_dim"])  # 定义最终的LayerNorm层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 定义输出层，没有偏置项

    def forward(self, in_idx):  # 前向传播函数，接受输入索引
        batch_size, seq_len = in_idx.shape  # 获取批次大小和序列长度
        tok_embeds = self.tok_emb(in_idx)  # 计算词嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 计算位置嵌入
        x = tok_embeds + pos_embeds  # 将词嵌入和位置嵌入相加
        x = self.drop_emb(x)  # 应用Dropout
        x = self.trf_blocks(x)  # 通过所有Transformer块
        x = self.final_norm(x)  # 应用最终的LayerNorm
        logits = self.out_head(x)  # 通过输出层得到logits
        return logits  # 返回logits

#####################################
# Chapter 5
#####################################

def text_to_token_ids(text, tokenizer):  # 将文本转换为标记ID
    encoded = tokenizer.encode(text)  # 使用tokenizer进行编码
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 将编码后的列表转换为张量并添加批次维度
    return encoded_tensor  # 返回标记ID张量

def token_ids_to_text(token_ids, tokenizer):  # 将标记ID转换回文本
    flat = token_ids.squeeze(0)  # 移除批次维度
    return tokenizer.decode(flat.tolist())  # 使用tokenizer解码回文本

def download_and_load_gpt2(model_size, models_dir):  # 下载并加载GPT-2模型
    # 验证模型大小
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")  # 如果模型大小不在允许的列表中，抛出异常

    # 定义路径
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"  # 模型文件的基础URL
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]  # 需要下载的文件列表

    # 下载文件
    os.makedirs(model_dir, exist_ok=True)  # 创建模型目录
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)  # 构造文件URL
        file_path = os.path.join(model_dir, filename)  # 构造文件路径
        download_file(file_url, file_path)  # 下载文件

    # 加载设置和参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)  # 获取最新的TensorFlow检查点路径
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))  # 加载hparams.json文件
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)  # 从TensorFlow检查点加载参数

    return settings, params  # 返回设置和参数

def download_file(url, destination):  # 下载文件的函数
    # 发送GET请求以下载文件
    with urllib.request.urlopen(url) as response:
        # 从头部获取文件总大小，默认为0如果不存在
        file_size = int(response.headers.get("Content-Length", 0))

        # 检查文件是否存在并且大小相同
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        # 定义读取文件的块大小
        block_size = 1024  # 1 Kilobyte

        # 使用总文件大小初始化进度条
        progress_bar_description = os.path.basename(url)  # 从URL中提取文件名
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            # 以二进制写模式打开目标文件
            with open(destination, "wb") as file:
                # 以块的形式读取文件并写入目标文件
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    file.write(chunk)
                    progress_bar.update(len(chunk))  # 更新进度条

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):  # 从TensorFlow检查点加载GPT-2参数
    # 使用空块为每个层初始化参数字典
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 遍历检查点中的每个变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载变量并移除单例维度
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 处理变量名以提取相关部分
        variable_name_parts = name.split("/")[1:]  # 跳过'model/'前缀

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

    return params  # 返回参数
def assign(left, right):
    # 检查两个张量的形状是否匹配，如果不匹配则抛出异常
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    # 创建一个新的PyTorch参数，其值为right张量的内容
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    # 将位置嵌入的权重分配给GPT模型
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    # 将词嵌入的权重分配给GPT模型
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    # 遍历每个Transformer块
    for b in range(len(params["blocks"])):
        # 从参数中提取查询、键、值的权重
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        # 分别将查询、键、值的权重分配给GPT模型的对应层
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 从参数中提取查询、键、值的偏置
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        # 分别将查询、键、值的偏置分配给GPT模型的对应层
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 分配注意力层输出的权重和偏置
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # 分配前馈网络层的权重和偏置
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # 分配层归一化参数
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    # 分配最终归一化层和输出层的参数
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # 生成函数，用于生成新的文本序列

    # 循环生成指定数量的新令牌
    for _ in range(max_new_tokens):
        # 使用当前索引的条件部分获取logits
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        # 只关注最后一个时间步的logits
        logits = logits[:, -1, :]

        # 如果设置了top_k，使用top_k采样过滤logits
        if top_k is not None:
            # 保留top_k个值
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            # 将小于最小值的logits设置为负无穷
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # 如果设置了温度值，应用温度缩放
        if temperature > 0.0:
            logits = logits / temperature

            # 应用softmax获取概率分布
            probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)

            # 从概率分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # 如果没有设置温度值，选择logits值最高的词汇索引
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # 如果遇到了eos_id指定的结束序列标记，提前停止生成
        if idx_next == eos_id:
            break

        # 将采样的索引追加到当前序列
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    # 返回生成的索引序列
    return idx