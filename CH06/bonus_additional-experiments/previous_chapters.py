# 版权所有 （c） Sebastian Raschka，Apache 许可证 2.0（参见 LICENSE.txt）。
# “从头开始构建大型语言模型” 的源代码
# - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch
#
# 此文件收集了我们到目前为止介绍的所有相关代码
# 贯穿第 2-5 章。
# 此文件可以作为独立脚本运行。
import numpy as np  # 导入NumPy库，用于数学运算
import tiktoken  # 导入Tiktoken库，用于文本分词
import torch  # 导入PyTorch库，用于深度学习
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch的数据集和数据加载器模块

#####################################

# Chapter 2

#####################################

class GPTDatasetV1(Dataset):  # 定义GPT数据集类，继承自PyTorch的Dataset
    def __init__(self, txt, tokenizer, max_length, stride):  # 初始化函数
        self.input_ids = []  # 存储输入ID的列表
        self.target_ids = []  # 存储目标ID的列表

        # 使用tokenizer对整个文本进行分词
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分割成长度为max_length的重叠序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 输入序列
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标序列
            self.input_ids.append(torch.tensor(input_chunk))  # 将输入序列转换为张量并添加到列表
            self.target_ids.append(torch.tensor(target_chunk))  # 将目标序列转换为张量并添加到列表

    def __len__(self):  # 返回数据集中的样本数量
        return len(self.input_ids)

    def __getitem__(self, idx):  # 根据索引获取样本
        return self.input_ids[idx], self.target_ids[idx]  # 返回输入和目标张量

def create_dataloader_v1(txt, batch_size=4, max_length=256,  # 定义创建数据加载器的函数
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

class MultiHeadAttention(nn.Module):  # 定义多头注意力类，继承自PyTorch的nn.Module
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, disable_causal_mask=False):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"  # 确保d_out能被num_heads整除

        self.d_out = d_out  # 输出维度
        self.num_heads = num_heads  # 头的数量
        self.head_dim = d_out // num_heads  # 每个头的维度

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 查询的线性层
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # 键的线性层
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 值的线性层
        self.out_proj = nn.Linear(d_out, d_out)  # 结合头输出的线性层
        self.dropout = nn.Dropout(dropout)  # Dropout层

        if not disable_causal_mask:  # 如果不禁用因果掩码
            self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))  # 注册上三角掩码
        self.disable_causal_mask = disable_causal_mask  # 是否禁用因果掩码

    def forward(self, x):  # 前向传播函数
        b, num_tokens, d_in = x.shape  # 获取输入的形状

        keys = self.W_key(x)  # 计算键
        queries = self.W_query(x)  # 计算查询
        values = self.W_value(x)  # 计算值

        # 通过添加一个`num_heads`维度隐式地分割矩阵
        # 展开最后一个维度：(b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置：(b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算缩放点积注意力（即自注意力）并使用因果掩码
        attn_scores = queries @ keys.transpose(2, 3)  # 每个头的点积

        if not self.disable_causal_mask:  # 如果不禁用因果掩码
            # 将原始掩码截断到标记数量并转换为布尔值
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

            # 使用掩码填充注意力分数
            attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)  # 应用softmax
        attn_weights = self.dropout(attn_weights)  # 应用dropout

        # 形状：(b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)  # 计算上下文向量

        # 合并头，其中self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)  # 调整形状
        context_vec = self.out_proj(context_vec)  # 可选的投影

        return context_vec  # 返回上下文向量

#####################################

# Chapter 4

#####################################

class LayerNorm(nn.Module):  # 定义层归一化类，继承自PyTorch的nn.Module
    def __init__(self, emb_dim):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        self.eps = 1e-5  # 一个很小的数，用于数值稳定性
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可学习的缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的偏移参数

    def forward(self, x):  # 前向传播函数
        mean = x.mean(dim=-1, keepdim=True)  # 计算均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算方差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 归一化
        return self.scale * norm_x + self.shift  # 缩放和平移

class GELU(nn.Module):  # 定义GELU激活函数类，继承自PyTorch的nn.Module
    def __init__(self):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数

    def forward(self, x):  # 前向传播函数
        return 0.5 * x * (1 + torch.tanh(  # GELU公式
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *  # 常数因子
            (x + 0.044715 * torch.pow(x, 3))  # x的三次方乘以一个常数
        ))
class FeedForward(nn.Module):  # 定义前馈神经网络模块
    def __init__(self, cfg):
        super().__init__()
        # 定义前馈神经网络的层：两个全连接层，中间加入GELU激活函数
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 第一个全连接层，输入维度是emb_dim，输出是4倍的emb_dim
            GELU(),  # GELU激活函数
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 第二个全连接层，输入是4倍emb_dim，输出是emb_dim
        )

    def forward(self, x):  # 前向传播函数
        return self.layers(x)  # 返回通过前馈层处理后的结果


class TransformerBlock(nn.Module):  # 定义Transformer模块
    def __init__(self, cfg, disable_causal_mask=False):  # 初始化函数
        super().__init__()
        # 初始化多头自注意力层
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],  # 输入维度为emb_dim
            d_out=cfg["emb_dim"],  # 输出维度为emb_dim
            context_length=cfg["context_length"],  # 上下文长度
            num_heads=cfg["n_heads"],  # 注意力头数
            dropout=cfg["drop_rate"],  # Dropout率
            qkv_bias=cfg["qkv_bias"],  # 是否使用QKV偏置
            disable_causal_mask=disable_causal_mask  # 是否禁用因果掩码
        )
        self.ff = FeedForward(cfg)  # 初始化前馈神经网络层
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 第一层归一化
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 第二层归一化
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # Dropout层，用于shortcut连接

    def forward(self, x):  # 前向传播函数
        # 自注意力块的shortcut连接
        shortcut = x  # 保存输入作为shortcut连接
        x = self.norm1(x)  # 对输入进行归一化
        x = self.att(x)   # 通过多头自注意力层处理输入，输出形状为[batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)  # 应用Dropout
        x = x + shortcut  # 加回原始输入（shortcut连接）

        # 前馈神经网络块的shortcut连接
        shortcut = x  # 保存输入作为shortcut连接
        x = self.norm2(x)  # 对输入进行归一化
        x = self.ff(x)  # 通过前馈神经网络层处理输入
        x = self.drop_shortcut(x)  # 应用Dropout
        x = x + shortcut  # 加回原始输入（shortcut连接）

        return x  # 返回输出


class GPTModel(nn.Module):  # 定义GPT模型
    def __init__(self, cfg, disable_causal_mask=False):  # 初始化函数
        super().__init__()
        # 初始化嵌入层：词汇表的嵌入层和位置嵌入层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 词嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入层
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # 嵌入层的Dropout

        # 初始化Transformer块，堆叠多个Transformer层
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg, disable_causal_mask) for _ in range(cfg["n_layers"])]  # 堆叠多个TransformerBlock
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])  # 最后的归一化层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 输出层，将嵌入维度转换为词汇表大小

    def forward(self, in_idx):  # 前向传播函数
        batch_size, seq_len = in_idx.shape  # 获取输入的批次大小和序列长度
        tok_embeds = self.tok_emb(in_idx)  # 获取词嵌入表示
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 获取位置嵌入表示
        x = tok_embeds + pos_embeds  # 将词嵌入和位置嵌入相加
        x = self.drop_emb(x)  # 应用Dropout
        x = self.trf_blocks(x)  # 通过多个Transformer块处理输入
        x = self.final_norm(x)  # 最后进行归一化
        logits = self.out_head(x)  # 通过输出层得到logits
        return logits  # 返回logits（每个位置的词汇表得分）


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx是当前上下文的（B, T）索引数组
    for _ in range(max_new_tokens):  # 生成max_new_tokens个新词

        # 如果当前上下文超出了支持的上下文大小，则裁剪当前上下文
        # 例如，如果LLM仅支持5个令牌，而上下文大小为10
        # 则仅使用最后5个令牌作为上下文
        idx_cond = idx[:, -context_size:]  # 裁剪上下文为context_size大小

        # 获取模型的预测结果
        with torch.no_grad():  # 在不计算梯度的情况下进行推理
            logits = model(idx_cond)  # 获取模型的输出logits

        # 仅关注最后一个时间步的输出
        # （batch, n_token, vocab_size）变为（batch, vocab_size）
        logits = logits[:, -1, :]  # 获取最后一个位置的logits

        # 获取具有最高logits值的词汇索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # 获取最大logits的索引

        # 将采样的索引附加到当前序列中
        idx = torch.cat((idx, idx_next), dim=1)  # 将新索引添加到序列中

    return idx  # 返回生成的索引

# Chapter 5

def assign(left, right):  # 定义一个函数来分配权重
    if left.shape != right.shape:  # 如果左边和右边的张量形状不匹配
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")  # 抛出形状不匹配的错误
    return torch.nn.Parameter(torch.tensor(right))  # 返回一个新的Parameter对象，其内容为右边的张量

def load_weights_into_gpt(gpt, params):  # 定义一个函数来加载权重到GPT模型
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])  # 分配位置嵌入权重
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])  # 分配词嵌入权重

    for b in range(len(params["blocks"])):  # 遍历每个Transformer块
        q_w, k_w, v_w = np.split(  # 分割查询、键、值的权重
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(  # 分配查询权重
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(  # 分配键权重
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(  # 分配值权重
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(  # 分割查询、键、值的偏置
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(  # 分配查询偏置
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(  # 分配键偏置
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(  # 分配值偏置
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(  # 分配注意力输出权重
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(  # 分配注意力输出偏置
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(  # 分配前馈网络第一层权重
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(  # 分配前馈网络第一层偏置
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(  # 分配前馈网络输出层权重
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(  # 分配前馈网络输出层偏置
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(  # 分配第一个层归一化权重
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(  # 分配第一个层归一化偏置
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(  # 分配第二个层归一化权重
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(  # 分配第二个层归一化偏置
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])  # 分配最终层归一化权重
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])  # 分配最终层归一化偏置
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])  # 分配输出层权重


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):  # 定义文本生成函数
    # 循环生成指定数量的新令牌
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # 获取当前上下文
        with torch.no_grad():  # 不计算梯度
            logits = model(idx_cond)  # 模型预测
        logits = logits[:, -1, :]  # 获取最后一个输出标记的Logits

        # 如果设置了top_k，使用top_k采样过滤Logits
        if top_k is not None:
            # 保留top_k个值
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # 如果设置了温度值，应用温度缩放
        if temperature > 0.0:
            logits = logits / temperature

            # 应用softmax获取概率分布
            probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)

            # 从概率分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # 如果没有设置温度值，选择Logits值最高的词汇索引
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # 如果遇到了eos_id指定的结束序列标记，则提前停止生成
        if idx_next == eos_id:
            break

        # 将采样的索引追加到当前序列
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx  # 返回生成的索引序列