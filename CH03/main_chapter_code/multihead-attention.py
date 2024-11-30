# NBVAL_IGNORE_OUTPUT
from importlib.metadata import version

# 打印出PyTorch库的版本信息
print("torch version:", version("torch"))

# 导入tiktoken库，用于文本的编码处理
import tiktoken
# 导入PyTorch库
import torch
# 导入PyTorch的神经网络模块
import torch.nn as nn
# 从PyTorch的utils包中导入数据集和数据加载器
from torch.utils.data import Dataset, DataLoader

# 定义GPTDatasetV1类，继承自PyTorch的Dataset类，用于处理GPT模型的数据
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        # 初始化两个列表，用于存储输入和目标的token ids
        self.input_ids = []
        self.target_ids = []

        # 使用tokenizer对整个文本进行编码，允许的特殊字符包括'<|endoftext|>'
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

        # 使用滑动窗口的方法，将文本分割成重叠的序列，每个序列的长度为max_length
        for i in range(0, len(token_ids) - max_length, stride):
            # 获取输入序列
            input_chunk = token_ids[i:i + max_length]
            # 获取目标序列，即输入序列的下一个token
            target_chunk = token_ids[i + 1: i + max_length + 1]
            # 将序列转换为PyTorch的tensor，并添加到对应的列表中
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # 返回数据集中样本的数量
    def __len__(self):
        return len(self.input_ids)

    # 根据索引返回对应的样本
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# 定义一个函数，用于创建数据加载器
def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True):
    # 使用gpt2的编码方式初始化tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 使用上面定义的GPTDatasetV1类创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 使用PyTorch的DataLoader创建数据加载器，设置批量大小和是否随机打乱数据
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # 返回创建的数据加载器
    return dataloader

# 打开并读取文本文件
with open("small-text-sample.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 使用gpt2的tokenizer对原始文本进行编码
tokenizer = tiktoken.get_encoding("gpt2")
encoded_text = tokenizer.encode(raw_text)

# 定义词汇表的大小、模型的输出维度、最大序列长度和上下文长度
vocab_size = 50257
output_dim = 256
max_len = 1024
context_length = max_len

# 创建token嵌入层和位置嵌入层，这两个层将用于将token ids转换为模型可以理解的嵌入向量
token_embedding_layer = nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# 使用定义好的函数创建数据加载器，设置批量大小、最大序列长度和步长
max_length = 4
dataloader = create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=max_length)

# 遍历数据加载器中的每个批次
for batch in dataloader:
    x, y = batch

    # 使用token嵌入层和位置嵌入层获取嵌入向量
    token_embeddings = token_embedding_layer(x)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))

    # 将token嵌入和位置嵌入相加，得到最终的输入嵌入
    input_embeddings = token_embeddings + pos_embeddings

    # 退出循环，因为我们只需要第一个批次的数据
    break

# 打印输入嵌入的形状，以验证其维度
print(input_embeddings.shape)

# 定义CausalSelfAttention类，这是一个因果自注意力机制的实现
class CausalSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        # 定义查询、键和值的线性变换层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 定义dropout层，用于在训练中随机丢弃部分神经元的输出
        self.dropout = nn.Dropout(dropout)
        # 注册一个上三角掩码buffer，用于在自注意力中实现因果关系
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    # 定义CausalSelfAttention类的前向传播方法
    def forward(self, x):
        # 提取输入张量x的批处理大小、token数量和特征维度
        b, n_tokens, d_in = x.shape  # New batch dimension b
        # 通过线性层计算查询（queries）、键（keys）和值（values）
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 计算查询和键的点积，然后对结果进行转置以匹配维度
        attn_scores = queries @ keys.transpose(1, 2)  # Changed transpose
        # 使用上三角掩码将未来信息的注意力分数设置为负无穷，以实现因果自注意力
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:n_tokens, :n_tokens], -torch.inf)
        # 应用softmax函数并缩放点积分数，然后应用dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # New

        # 计算加权的值（context vector）作为自注意力的输出
        context_vec = attn_weights @ values
        return context_vec

    # 定义MultiHeadAttentionWrapper类，用于实现多头自注意力机制
    class MultiHeadAttentionWrapper(nn.Module):
        def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
            super().__init__()
            # 创建一个模块列表，包含多个CausalSelfAttention实例，每个实例代表一个注意力头
            self.heads = nn.ModuleList(
                [CausalSelfAttention(d_in, d_out, context_length, dropout, qkv_bias)
                 for _ in range(num_heads)]
            )
            # 定义一个线性层，用于将多个头的输出合并
            self.out_proj = nn.Linear(d_out * num_heads, d_out * num_heads)

        def forward(self, x):
            # 遍历每个头，计算上下文向量，并将结果沿着最后一个维度拼接
            context_vec = torch.cat([head(x) for head in self.heads], dim=-1)
            # 通过输出投影层返回最终的上下文向量
            return self.out_proj(context_vec)

    # 设置随机种子以确保实验的可重复性
    torch.manual_seed(123)

    # 定义上下文长度和输入维度
    context_length = max_length
    d_in = output_dim

    # 定义头数和每个头的输出维度
    num_heads = 2
    d_out = d_in // num_heads

    # 创建MultiHeadAttentionWrapper实例
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads)

    # 使用输入嵌入计算上下文向量
    batch = input_embeddings
    context_vecs = mha(batch)

    # 打印上下文向量的形状
    print("context_vecs.shape:", context_vecs.shape)

    # 定义MultiHeadAttention类，这是一个多头自注意力机制的实现
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
            super().__init__()
            # 确保输出维度可以被头数整除
            assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

            self.d_out = d_out
            self.num_heads = num_heads
            self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

            # 定义查询、键和值的线性层
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            # 定义一个线性层，用于合并头输出
            self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
            # 定义dropout层
            self.dropout = nn.Dropout(dropout)
            # 注册一个上三角掩码buffer，用于在自注意力中实现因果关系
            self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

        def forward(self, x):
            # 提取输入张量x的批处理大小、token数量和特征维度
            b, num_tokens, d_in = x.shape

            # 通过线性层计算查询、键和值
            keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
            queries = self.W_query(x)
            values = self.W_value(x)

            # 将查询、键和值的维度展开，为多头自注意力做准备
            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
            values = values.view(b, num_tokens, self.num_heads, self.head_dim)
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

            # 转置查询、键和值，以适应多头自注意力的计算
            keys = keys.transpose(1, 2)
            queries = queries.transpose(1, 2)
            values = values.transpose(1, 2)

            # 计算每个头的点积注意力分数，并应用因果掩码
            attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

            # 将掩码截断到token数量，并转换为布尔值
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

            # 使用掩码填充注意力分数
            attn_scores.masked_fill_(mask_bool, -torch.inf)

            # 应用softmax函数并缩放点积分数，然后应用dropout
            attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # 计算加权的值（context vector）作为自注意力的输出
            context_vec = (attn_weights @ values).transpose(1, 2)

            # 合并头输出，并将结果转换为原始的输出维度
            context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
            context_vec = self.out_proj(context_vec)  # optional projection

            return context_vec

    # 设置随机种子以确保实验的可重复性
    torch.manual_seed(123)

    # 定义上下文长度、输入维度和输出维度
    context_length = max_length
    d_in = output_dim
    d_out = d_in

    # 创建MultiHeadAttention实例
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

    # 使用输入嵌入计算上下文向量
    batch = input_embeddings
    context_vecs = mha(batch)

    # 打印上下文向量的形状
    print("context_vecs.shape:", context_vecs.shape)