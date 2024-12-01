# 版权所有 （c） Sebastian Raschka，Apache 许可证 2.0（参见 LICENSE.txt）。
# “从头开始构建大型语言模型” 的源代码
# - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch

# 此文件收集了我们到目前为止介绍的所有相关代码
# 贯穿第 2-4 章。
# 此文件可以作为独立脚本运行。
import tiktoken  # 导入用于文本编码的库
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch的数据集和数据加载器模块

#####################################
# Chapter 2
#####################################

class GPTDatasetV1(Dataset):  # 定义一个数据集类，用于GPT模型
    def __init__(self, txt, tokenizer, max_length, stride):  # 初始化函数
        self.input_ids = []  # 存储输入ID
        self.target_ids = []  # 存储目标ID

        # 将整个文本进行标记化
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

def create_dataloader_v1(txt, batch_size=4, max_length=256,  # 定义一个函数，用于创建数据加载器
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 初始化标记器
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

class MultiHeadAttention(nn.Module):  # 定义多头注意力类
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"  # 确保d_out能被num_heads整除

        self.d_out = d_out  # 输出维度
        self.num_heads = num_heads  # 头的数量
        self.head_dim = d_out // num_heads  # 每个头的维度

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 查询的线性层
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # 键的线性层
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 值的线性层
        self.out_proj = nn.Linear(d_out, d_out)  # 结合头输出的线性层
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))  # 注册一个上三角掩码

    def forward(self, x):  # 前向传播函数
        b, num_tokens, d_in = x.shape  # 获取输入的批次大小、标记数量和维度

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

        # 将原始掩码截断到标记数量并转换为布尔值
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码填充注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)  # 应用softmax
        attn_weights = self.dropout(attn_weights)  # 应用dropout

        # 形状：(b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 结合头，其中self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 可选的投影

        return context_vec  # 返回上下文向量

#####################################
# Chapter 4
#####################################

class LayerNorm(nn.Module):  # 定义层归一化类
    def __init__(self, emb_dim):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        self.eps = 1e-5  # 一个很小的数，用于防止除以零
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可学习的缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的偏移参数

    def forward(self, x):  # 前向传播函数
        mean = x.mean(dim=-1, keepdim=True)  # 计算均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算方差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 归一化
        return self.scale * norm_x + self.shift  # 缩放和平移

class GELU(nn.Module):  # 定义GELU激活函数类
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
        # 定义前馈神经网络结构，包括两个全连接层和GELU激活函数
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 第一个线性层，输入维度为emb_dim，输出维度为4倍的emb_dim
            GELU(),  # 激活函数
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 第二个线性层，输入维度为4倍的emb_dim，输出为emb_dim
        )

    def forward(self, x):
        return self.layers(x)  # 前馈网络的前向传播


class TransformerBlock(nn.Module):  # 定义Transformer模块
    def __init__(self, cfg):
        super().__init__()
        # MultiHeadAttention是多头注意力机制，使用给定的配置参数
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)  # 使用前馈网络
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 第一个LayerNorm层
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 第二个LayerNorm层
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # Dropout层，用于防止过拟合

    def forward(self, x):
        # 注意力块的shortcut连接
        shortcut = x
        x = self.norm1(x)  # 对输入进行LayerNorm处理
        x = self.att(x)    # 经过多头注意力机制
        x = self.drop_shortcut(x)  # Dropout
        x = x + shortcut  # 将原始输入加回来，形成残差连接

        # 前馈块的shortcut连接
        shortcut = x
        x = self.norm2(x)  # 对输入进行LayerNorm处理
        x = self.ff(x)  # 经过前馈网络
        x = self.drop_shortcut(x)  # Dropout
        x = x + shortcut  # 将原始输入加回来，形成残差连接

        return x  # 返回结果


class GPTModel(nn.Module):  # 定义GPT模型
    def __init__(self, cfg):
        super().__init__()
        # 词嵌入层和位置嵌入层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 词嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入层
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # 嵌入层的Dropout层

        # 堆叠多个TransformerBlock层
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]  # 使用给定的层数n_layers堆叠TransformerBlock
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])  # 最后的LayerNorm层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 输出层，用于生成词汇的概率分布

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape  # 获取输入的批次大小和序列长度
        tok_embeds = self.tok_emb(in_idx)  # 通过词嵌入层得到token的嵌入表示
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 获取位置嵌入
        x = tok_embeds + pos_embeds  # 将词嵌入和位置嵌入相加，作为Transformer的输入
        x = self.drop_emb(x)  # 使用Dropout防止过拟合
        x = self.trf_blocks(x)  # 经过多个TransformerBlock层
        x = self.final_norm(x)  # 进行最后的LayerNorm处理
        logits = self.out_head(x)  # 通过输出层得到logits
        return logits  # 返回logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx是当前上下文的（B, T）形状的索引数组
    for _ in range(max_new_tokens):
        # 如果当前上下文超出模型支持的最大上下文长度，则进行裁剪
        idx_cond = idx[:, -context_size:]  # 截取最后context_size个tokens作为上下文

        # 获取模型的预测结果
        with torch.no_grad():
            logits = model(idx_cond)

        # 只关注最后一个时间步的输出
        # (batch, n_token, vocab_size) 变为 (batch, vocab_size)
        logits = logits[:, -1, :]

        # 获取具有最大logits值的词汇索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 将生成的词汇索引追加到当前序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx  # 返回最终生成的token索引


if __name__ == "__main__":

    # 配置GPT模型的超参数
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # 词汇表大小
        "context_length": 1024,  # 上下文长度
        "emb_dim": 768,          # 嵌入维度
        "n_heads": 12,           # 注意力头数
        "n_layers": 12,          # Transformer层数
        "drop_rate": 0.1,        # Dropout率
        "qkv_bias": False        # 是否使用QKV偏置
    }

    torch.manual_seed(123)  # 设置随机种子以确保可复现
    model = GPTModel(GPT_CONFIG_124M)  # 初始化GPT模型
    model.eval()  # 设置模型为评估模式，禁用Dropout

    start_context = "Hello, I am"  # 定义生成文本的起始上下文

    tokenizer = tiktoken.get_encoding("gpt2")  # 使用GPT-2的tokenizer
    encoded = tokenizer.encode(start_context)  # 编码起始文本为token索引
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 转换为tensor并添加批次维度

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)  # 输出输入文本
    print("Encoded input text:", encoded)  # 输出编码后的文本
    print("encoded_tensor.shape:", encoded_tensor.shape)  # 输出tensor的形状

    # 使用模型生成文本
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,  # 生成10个新token
        context_size=GPT_CONFIG_124M["context_length"]  # 上下文大小设置为模型支持的最大长度
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())  # 解码生成的token为文本

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)  # 输出生成的token
    print("Output length:", len(out[0]))  # 输出生成的token数量
    print("Output text:", decoded_text)  # 输出解码后的文本
