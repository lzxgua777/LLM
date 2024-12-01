# 版权所有 （c） Sebastian Raschka，Apache 许可证 2.0（参见 LICENSE.txt）。
# “从头开始构建大型语言模型” 的源代码
# - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch
#
# 此文件收集了我们到目前为止介绍的所有相关代码
# 贯穿第 2-5 章。
import json
import os
import urllib

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from tqdm import tqdm

##################################### Chapter 3 #####################################
# 定义多头注意力机制类
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out 必须能被 n_heads 整除"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 缩小投影维度以匹配期望的输出维度

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 查询（Q）的线性层
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # 键（K）的线性层
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 值（V）的线性层
        self.out_proj = nn.Linear(d_out, d_out)  # 结合头输出的线性层
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))  # 注册一个上三角掩码缓冲区

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # 形状：(batch_size, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 通过添加一个`num_heads`维度隐式地分割矩阵
        # 展开最后一个维度：(batch_size, num_tokens, d_out) -> (batch_size, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置：(batch_size, num_tokens, num_heads, head_dim) -> (batch_size, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算缩放点积注意力（即自注意力）并使用因果掩码
        attn_scores = queries @ keys.transpose(2, 3)  # 每个头的点积

        # 将原始掩码截断到token数量，并转换为布尔值
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码填充注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 形状：(batch_size, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并头，其中 self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 可选的投影

        return context_vec


##################################### Chapter 4 #####################################
# 定义层归一化类
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 尺度参数，可学习的
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 偏移参数，可学习的

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # 计算均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算方差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 归一化
        return self.scale * norm_x + self.shift  # 应用尺度和偏移


# 定义GELU激活函数类
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *  # GELU激活函数的计算
            (x + 0.044715 * torch.pow(x, 3))
        ))


# 定义前馈网络类
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 输入层
            GELU(),  # GELU激活函数
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 输出层
        )

    def forward(self, x):
        return self.layers(x)  # 通过前馈网络

class TransformerBlock(nn.Module):  # 定义 Transformer 块类，继承自 PyTorch 的 nn.Module
    def __init__(self, cfg):  # 初始化函数，接受配置字典 cfg
        super().__init__()  # 调用父类的初始化函数
        # 定义多头注意力模块
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],  # 输入嵌入维度
            d_out=cfg["emb_dim"],  # 输出嵌入维度
            context_length=cfg["context_length"],  # 上下文长度
            num_heads=cfg["n_heads"],  # 注意力头的数量
            dropout=cfg["drop_rate"],  # Dropout 概率
            qkv_bias=cfg["qkv_bias"])  # 是否使用 QKV 偏置

        # 定义前馈网络模块
        self.ff = FeedForward(cfg)
        # 定义两个 LayerNorm 层，用于规范化
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        # 定义 Dropout 层
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):  # 前向传播函数
        # 注意力模块的残差连接
        shortcut = x  # 保存输入作为残差
        x = self.norm1(x)  # 规范化输入
        x = self.att(x)  # 经过多头注意力层，形状 [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)  # 应用 Dropout
        x = x + shortcut  # 将残差加回

        # 前馈模块的残差连接
        shortcut = x  # 保存当前输入作为残差
        x = self.norm2(x)  # 规范化输入
        x = self.ff(x)  # 经过前馈网络层
        x = self.drop_shortcut(x)  # 应用 Dropout
        x = x + shortcut  # 将残差加回

        return x  # 返回结果


class GPTModel(nn.Module):  # 定义 GPT 模型类，继承自 nn.Module
    def __init__(self, cfg):  # 初始化函数，接受配置字典 cfg
        super().__init__()  # 调用父类初始化函数

        # 定义词嵌入层和位置嵌入层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 词嵌入
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # Dropout 层

        # 定义一系列 Transformer 块
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])  # 堆叠多个 Transformer 块

        # 定义最终的规范化层和输出线性层
        self.final_norm = LayerNorm(cfg["emb_dim"])  # 最后一个 LayerNorm
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 输出线性变换

    def forward(self, in_idx):  # 前向传播函数
        batch_size, seq_len = in_idx.shape  # 获取批量大小和序列长度
        tok_embeds = self.tok_emb(in_idx)  # 获取词嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 获取位置嵌入
        x = tok_embeds + pos_embeds  # 将词嵌入和位置嵌入相加，形状 [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)  # 应用 Dropout
        x = self.trf_blocks(x)  # 通过 Transformer 块
        x = self.final_norm(x)  # 最终的 LayerNorm
        logits = self.out_head(x)  # 线性输出层，得到 logits
        return logits  # 返回 logits


#####################################
# Chapter 5 - 第五章代码
#####################################
def text_to_token_ids(text, tokenizer):  # 将文本转化为 token ID
    encoded = tokenizer.encode(text)  # 使用分词器对文本进行编码
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加 batch 维度
    return encoded_tensor  # 返回编码后的张量


def token_ids_to_text(token_ids, tokenizer):  # 将 token ID 转化为文本
    flat = token_ids.squeeze(0)  # 去掉 batch 维度
    return tokenizer.decode(flat.tolist())  # 解码为文本并返回


def download_and_load_gpt2(model_size, models_dir):  # 下载并加载 GPT-2 模型
    # 验证模型大小是否有效
    allowed_sizes = ("124M", "355M", "774M", "1558M")  # 允许的模型大小
    if model_size not in allowed_sizes:  # 检查输入是否在允许范围
        raise ValueError(f"Model size not in {allowed_sizes}")  # 抛出错误

    # 定义路径
    model_dir = os.path.join(models_dir, model_size)  # 模型目录
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"  # 基础 URL
    filenames = [  # 要下载的文件列表
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 下载文件
    os.makedirs(model_dir, exist_ok=True)  # 确保目录存在
    for filename in filenames:  # 遍历每个文件
        file_url = os.path.join(base_url, model_size, filename)  # 构造文件 URL
        file_path = os.path.join(model_dir, filename)  # 构造本地保存路径
        download_file(file_url, file_path)  # 下载文件

    # 加载模型的超参数和权重
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)  # 获取最新的检查点路径
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))  # 加载超参数
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)  # 加载权重

    return settings, params  # 返回超参数和权重

# 定义一个下载文件的函数
def download_file(url, destination):
    # 发送GET请求以下载文件
    with urllib.request.urlopen(url) as response:
        # 从响应头中获取文件总大小，如果不存在则默认为0
        file_size = int(response.headers.get("Content-Length", 0))

        # 检查文件是否存在并且大小相同
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"文件已存在并且是最新的：{destination}")
                return

        # 定义读取文件的块大小
        block_size = 1024  # 1 千字节

        # 使用总文件大小初始化进度条
        progress_bar_description = os.path.basename(url)  # 从URL中提取文件名
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


# 定义一个从TensorFlow检查点加载GPT-2参数的函数
def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # 使用空块为每个层初始化参数字典
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 遍历检查点中的每个变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载变量并移除单例维度
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 处理变量名以提取相关部分
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


# 定义一个赋值函数，用于比较两个张量的形状并赋值
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"形状不匹配。左边：{left.shape}，右边：{right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


# 定义一个将权重加载到GPT模型中的函数
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])  # 位置嵌入权重赋值
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])  # 词嵌入权重赋值

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)  # 分割自注意力层的权重
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)  # 查询（Q）权重赋值
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)  # 键（K）权重赋值
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)  # 值（V）权重赋值

        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)  # 分割自注意力层的偏置
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)  # 查询（Q）偏置赋值
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)  # 键（K）偏置赋值
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)  # 值（V）偏置赋值

        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight,
                                                       params["blocks"][b]["attn"]["c_proj"]["w"].T)  # 自注意力输出层权重赋值
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias,
                                                     params["blocks"][b]["attn"]["c_proj"]["b"])  # 自注意力输出层偏置赋值

        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight,
                                                       params["blocks"][b]["mlp"]["c_fc"]["w"].T)  # 前馈网络第一层权重赋值
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias,
                                                     params["blocks"][b]["mlp"]["c_fc"]["b"])  # 前馈网络第一层偏置赋值
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight,
                                                       params["blocks"][b]["mlp"]["c_proj"]["w"].T)  # 前馈网络输出层权重赋值
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias,
                                                     params["blocks"][b]["mlp"]["c_proj"]["b"])  # 前馈网络输出层偏置赋值

        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale,
                                               params["blocks"][b]["ln_1"]["g"])  # 第一层层归一化尺度赋值
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift,
                                               params["blocks"][b]["ln_1"]["b"])  # 第一层层归一化偏移赋值
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale,
                                               params["blocks"][b]["ln_2"]["g"])  # 第二层层归一化尺度赋值
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift,
                                               params["blocks"][b]["ln_2"]["b"])  # 第二层层归一化偏移赋值

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])  # 最终层归一化尺度赋值
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])  # 最终层归一化偏移赋值
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])  # 输出层权重赋值
#####################################
# Chapter 6 - 第六章代码
#####################################
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    # 定义一个函数，用于分类输入文本
    # 参数说明：
    # text：要分类的文本
    # model：预训练的分类模型
    # tokenizer：分词器，用于将文本转为 token ID
    # device：运行设备（如 'cpu' 或 'cuda'）
    # max_length：输入序列的最大长度
    # pad_token_id：用于填充的 token ID，默认为 50256

    model.eval()  # 将模型设置为评估模式，关闭 dropout 等影响推理的功能

    # 准备模型的输入
    input_ids = tokenizer.encode(text)  # 使用分词器将文本编码为 token ID 列表
    supported_context_length = model.pos_emb.weight.shape[0]  # 获取模型支持的最大上下文长度

    # 如果输入序列过长，进行截断
    input_ids = input_ids[:min(max_length, supported_context_length)]  # 截断到 max_length 或支持的最大长度

    # 对序列进行填充到指定长度
    input_ids += [pad_token_id] * (max_length - len(input_ids))  # 使用 pad_token_id 填充到 max_length
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # 转为张量并增加 batch 维度

    # 模型推理
    with torch.no_grad():  # 禁用梯度计算，以加速推理并减少内存消耗
        logits = model(input_tensor.to(device))[:, -1, :]
        # 获取最后一个输出 token 的 logits
        # `[:, -1, :]` 表示只取最后一个 token 的输出结果

    predicted_label = torch.argmax(logits, dim=-1).item()
    # 使用 argmax 获取概率最大的类别索引，并转换为 Python 的整数类型

    # 返回分类结果
    return "spam" if predicted_label == 1 else "not spam"
    # 如果预测的标签为 1，返回 "spam"，否则返回 "not spam"
