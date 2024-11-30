#版权所有 (c) Sebastian Raschka，根据 Apache 2.0 许可证发布（详见 LICENSE.txt）。
#本文摘自《从零开始构建大型语言模型》一书：

#书籍链接https://www.manning.com/books/build-a-large-language-model-from-scratch
#代码仓库地址：https://github.com/rasbt/LLMs-from-scratch

# 导入所需的库
import tiktoken  # 用于GPT的分词器
import torch  # PyTorch库
import torch.nn as nn  # PyTorch的神经网络模块
from torch.utils.data import Dataset, DataLoader  # 导入数据集和数据加载器模块

# 定义数据集类
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []  # 用于存储输入序列
        self.target_ids = []  # 用于存储目标序列

        # 使用tokenizer对整个文本进行分词
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})  # 对文本进行分词并编码

        # 使用滑动窗口将文本分成重叠的序列
        for i in range(0, len(token_ids) - max_length, stride):  # 遍历文本，生成不同的切片
            input_chunk = token_ids[i:i + max_length]  # 取出输入部分的序列
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 取出目标部分的序列（偏移一位）
            self.input_ids.append(torch.tensor(input_chunk))  # 将输入序列转为Tensor并加入列表
            self.target_ids.append(torch.tensor(target_chunk))  # 将目标序列转为Tensor并加入列表

    def __len__(self):
        return len(self.input_ids)  # 返回数据集的大小（即输入序列的数量）

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]  # 返回指定索引的输入和目标序列

# 定义创建DataLoader的函数
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 初始化tokenizer（GPT-2的编码器）
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)  # 使用GPTDatasetV1类创建数据集实例

    # 创建DataLoader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)  # 使用DataLoader将数据集加载成批次

    return dataloader  # 返回创建好的DataLoader

# 定义多头自注意力类
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()  # 调用父类的构造函数
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"  # 确保输出维度能够被头数整除

        self.d_out = d_out  # 输出维度
        self.num_heads = num_heads  # 头的数量
        self.head_dim = d_out // num_heads  # 每个头的维度，确保d_out可以均分给每个头

        # 定义用于生成查询（query）、键（key）和值（value）的线性变换
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 输入到查询的线性变换
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # 输入到键的线性变换
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 输入到值的线性变换
        self.out_proj = nn.Linear(d_out, d_out)  # 用于将多头的输出组合在一起
        self.dropout = nn.Dropout(dropout)  # dropout层，用于正则化
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))  # 生成一个上三角的mask，用于防止信息泄露

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # 输入的形状，b是batch size，num_tokens是序列长度，d_in是每个token的维度

        # 通过线性变换得到查询（query）、键（key）和值（value）
        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)  # Shape: (b, num_tokens, d_out)
        values = self.W_value(x)  # Shape: (b, num_tokens, d_out)

        # 将查询、键、值按照num_heads划分为多个头
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)  # Shape: (b, num_tokens, num_heads, head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  # Shape: (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)  # Shape: (b, num_tokens, num_heads, head_dim)

        # 转置，使得每个头的维度在前面
        keys = keys.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)

        # 计算缩放点积注意力（自注意力）
        attn_scores = queries @ keys.transpose(2, 3)  # 计算每个头的注意力得分，Shape: (b, num_heads, num_tokens, num_tokens)

        # 使用mask来确保每个token只能关注到自己及其之前的token
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # 截取mask，使得它只关注当前的序列长度

        # 使用mask填充注意力得分
        attn_scores.masked_fill_(mask_bool, -torch.inf)  # 将mask位置的得分设为负无穷，阻止该位置的注意力

        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)  # 使用softmax计算权重，并按头维度缩放
        attn_weights = self.dropout(attn_weights)  # 使用dropout进行正则化

        # 使用注意力权重计算上下文向量
        context_vec = (attn_weights @ values).transpose(1, 2)  # Shape: (b, num_tokens, num_heads, head_dim)

        # 将多头的输出合并起来
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # Shape: (b, num_tokens, d_out)
        context_vec = self.out_proj(context_vec)  # 可选的线性投影，用于合并所有头的输出

        return context_vec  # 返回处理后的上下文向量