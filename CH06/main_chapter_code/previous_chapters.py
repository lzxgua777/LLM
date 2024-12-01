# 版权所有 （c） Sebastian Raschka，Apache 许可证 2.0（参见 LICENSE.txt）。
# “从头开始构建大型语言模型” 的源代码
# - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch
#
# 此文件收集了我们到目前为止介绍的所有相关代码
# 贯穿第 2-5 章。
# 此文件可以作为独立脚本运行。
import numpy as np  # 导入NumPy库，用于数组和矩阵运算
import tiktoken  # 导入tiktoken库，用于GPT的tokenizer
import torch  # 导入PyTorch库，用于深度学习模型
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch.utils.data import Dataset, DataLoader  # 从torch.utils.data导入Dataset和DataLoader类，用于数据处理

#####################################
# Chapter 2
#####################################


# 定义一个GPTDatasetV1类继承自Dataset，用于处理输入文本数据
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []  # 用于存储输入的token IDs
        self.target_ids = []  # 用于存储目标的token IDs

        # 对整个文本进行分词
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分割成重叠的序列，序列长度为max_length，步幅为stride
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 当前输入片段
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 当前目标片段
            self.input_ids.append(torch.tensor(input_chunk))  # 将输入片段转为张量并添加到列表
            self.target_ids.append(torch.tensor(target_chunk))  # 将目标片段转为张量并添加到列表

    def __len__(self):
        return len(self.input_ids)  # 返回数据集的大小

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]  # 获取指定索引的输入和目标数据


# 定义一个函数来创建数据加载器
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 初始化tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader  # 返回数据加载器


#####################################
# Chapter 3
#####################################

# 定义多头自注意力机制类
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"  # 确保d_out能被num_heads整除

        self.d_out = d_out  # 输出的维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = d_out // num_heads  # 每个头的维度

        # 定义查询、键、值的线性变换
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 将多个头的输出合并
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))  # 定义上三角掩码

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # 获取输入的batch_size，token数目和输入维度

        # 计算键、查询和值
        keys = self.W_key(x)  # 形状: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 分头处理（将维度展开为多头）
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)  # 形状: (b, num_tokens, num_heads, head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算缩放点积注意力（自注意力）并应用因果掩码
        attn_scores = queries @ keys.transpose(2, 3)  # 每个头的点积

        # 原始的掩码，截断为当前token数量并转换为布尔值
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码填充注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # 应用dropout

        # 计算上下文向量
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并头，输出的维度是d_out = num_heads * head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 可选的投影层

        return context_vec  # 返回上下文向量


#####################################
# Chapter 4
#####################################

# 定义LayerNorm类，用于实现层归一化
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 为了数值稳定性，防止除以0
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 归一化的尺度参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 归一化的偏移参数

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # 计算均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算方差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 归一化
        return self.scale * norm_x + self.shift  # 应用尺度和偏移


# 定义GELU激活函数
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(  # GELU激活函数的计算
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *  # 常数部分
            (x + 0.044715 * torch.pow(x, 3))  # x的三次方部分
        ))


# 定义前馈神经网络层
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(  # 定义一个顺序的网络
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 第一个线性层，输入维度为emb_dim，输出维度为4倍的emb_dim
            GELU(),  # GELU激活函数
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 第二个线性层，输入维度为4倍的emb_dim，输出为emb_dim
        )

    def forward(self, x):
        return self.layers(x)  # 前向传播，依次通过层
# 定义一个TransformerBlock类，作为GPT模型中的基本构建块
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 多头自注意力层
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],  # 输入维度
            d_out=cfg["emb_dim"],  # 输出维度
            context_length=cfg["context_length"],  # 上下文长度
            num_heads=cfg["n_heads"],  # 注意力头数
            dropout=cfg["drop_rate"],  # dropout率
            qkv_bias=cfg["qkv_bias"]  # 是否使用偏置项
        )

        # 前馈神经网络层
        self.ff = FeedForward(cfg)

        # 两个层归一化层（分别用于注意力和前馈网络）
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # 残差连接的dropout
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 注意力块的快捷连接
        shortcut = x  # 保存输入作为shortcut连接
        x = self.norm1(x)  # 先进行LayerNorm
        x = self.att(x)  # 通过多头自注意力层
        x = self.drop_resid(x)  # 应用dropout
        x = x + shortcut  # 将原始输入加回来，形成残差连接

        # 前馈神经网络块的快捷连接
        shortcut = x  # 保存输入作为shortcut连接
        x = self.norm2(x)  # 进行LayerNorm
        x = self.ff(x)  # 通过前馈神经网络
        x = self.drop_resid(x)  # 应用dropout
        x = x + shortcut  # 将原始输入加回来，形成残差连接

        return x  # 返回经过TransformerBlock处理的结果


# 定义GPT模型类
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 词嵌入层，映射词汇表大小到嵌入维度
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # 位置嵌入层，将位置信息映射到嵌入空间
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # 嵌入层的dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 堆叠多个TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]  # 按照配置中的n_layers数目堆叠
        )

        # 最后的层归一化
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # 输出头，用于将嵌入转换回词汇表大小的logits
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape  # 获取批量大小和序列长度

        # 通过词嵌入层得到token的嵌入表示
        tok_embeds = self.tok_emb(in_idx)

        # 通过位置嵌入层获取位置编码
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # 将token嵌入和位置嵌入加起来
        x = tok_embeds + pos_embeds  # 形状 [batch_size, num_tokens, emb_size]

        # 应用dropout
        x = self.drop_emb(x)

        # 通过多个TransformerBlock进行前向传播
        x = self.trf_blocks(x)

        # 最后的层归一化
        x = self.final_norm(x)

        # 输出层，通过线性变换得到每个token的logits
        logits = self.out_head(x)

        return logits  # 返回logits


# 定义一个简单的文本生成函数
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx 是当前上下文的（B，T）形式的token索引（批次大小，序列长度）
    for _ in range(max_new_tokens):
        # 如果当前上下文超过了最大支持的上下文大小，裁剪它
        idx_cond = idx[:, -context_size:]  # 只保留最后context_size个tokens

        # 使用模型获取预测结果
        with torch.no_grad():
            logits = model(idx_cond)

        # 聚焦于最后一个时间步的预测
        # (batch, n_token, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]

        # 获取最大logits值的词汇索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 将新的token索引附加到当前序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx  # 返回生成的token索引


#####################################
# Chapter 5
#####################################

# 定义赋值函数，将权重值从右边赋给左边
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))  # 返回右边的张量作为PyTorch的参数

# 定义函数，将预训练的权重加载到GPT模型中
def load_weights_into_gpt(gpt, params):
    # 加载位置嵌入和token嵌入的权重
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    # 加载每个TransformerBlock中的权重
    for b in range(len(params["blocks"])):
        # 加载多头注意力的查询、键、值权重
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 加载偏置项
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 加载输出投影层的权重
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # 加载前馈神经网络的权重
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

        # 加载层归一化的权重
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

        # 加载最终的层归一化和输出头的权重
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


# 将文本转换为token索引
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加批次维度
    return encoded_tensor


# 将token索引转换为文本
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # 移除批次维度
    return tokenizer.decode(flat.tolist())  # 解码为文本