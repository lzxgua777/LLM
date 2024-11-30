# 导入PyTorch库及其神经网络模块
import torch
import torch.nn as nn

# 定义CausalAttentionWithoutBuffers类，实现不带缓冲区的因果自注意力机制
class CausalAttentionWithoutBuffers(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out  # 输出维度
        # 定义线性层，用于计算查询、键和值
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)  # dropout层
        # 创建上三角掩码，用于实现因果关系（防止未来信息泄露）
        self.mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # 提取输入张量的批处理大小、token数量和特征维度
        # 通过线性层计算查询、键和值
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 计算查询和键的点积，得到注意力分数
        attn_scores = queries @ keys.transpose(1, 2)
        # 应用掩码，将未来信息的注意力分数设置为负无穷
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # 应用softmax函数并缩放点积分数，然后应用dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算加权的值（context vector）作为自注意力的输出
        context_vec = attn_weights @ values
        return context_vec

# 设置随机种子以确保结果可复现
torch.manual_seed(123)

# 创建一个示例输入张量
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]]  # step     (x^6)
)

# 将输入张量堆叠两次，形成批处理
batch = torch.stack((inputs, inputs), dim=0)
context_length = batch.shape[1]  # 上下文长度等于批次中token的数量
d_in = inputs.shape[1]  # 输入维度等于输入张量的最后一个维度
d_out = 2  # 输出维度

# 创建CausalAttentionWithoutBuffers实例
ca_without_buffer = CausalAttentionWithoutBuffers(d_in, d_out, context_length, 0.0)

# 在不计算梯度的情况下，计算上下文向量
with torch.no_grad():
    context_vecs = ca_without_buffer(batch)

# 打印上下文向量
print(context_vecs)
# 打印机器是否有GPU
print("Machine has GPU:", torch.cuda.is_available())
# 打印W_query权重的设备
print("W_query.device:", ca_without_buffer.W_query.weight.device)
# 打印掩码的设备
print("mask.device:", ca_without_buffer.mask.device)

# 打印掩码的数据类型
type(ca_without_buffer.mask)
# 将掩码发送到GPU
ca_without_buffer.mask = ca_without_buffer.mask.to("cuda")
# 打印掩码的新设备
print("mask.device:", ca_without_buffer.mask.device)

# 再次在不计算梯度的情况下，计算上下文向量
with torch.no_grad():
    context_vecs = ca_without_buffer(batch)

# 打印新的上下文向量
print(context_vecs)

# 定义CausalAttentionWithBuffer类，实现带缓冲区的因果自注意力机制
class CausalAttentionWithBuffer(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        # 定义线性层，用于计算查询、键和值
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # 注册一个缓冲区，用于存储上三角掩码
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # 通过线性层计算查询、键和值
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 计算查询和键的点积，得到注意力分数
        attn_scores = queries @ keys.transpose(1, 2)
        # 应用掩码，将未来信息的注意力分数设置为负无穷
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # 应用softmax函数并缩放点积分数，然后应用dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算加权的值（context vector）作为自注意力的输出
        context_vec = attn_weights @ values
        return context_vec

# 创建CausalAttentionWithBuffer实例，并将其发送到GPU
ca_with_buffer = CausalAttentionWithBuffer(d_in, d_out, context_length, 0.0)
ca_with_buffer.to("cuda")

# 打印W_query权重的设备
print("W_query.device:", ca_with_buffer.W_query.weight.device)
# 打印掩码的设备
print("mask.device:", ca_with_buffer.mask.device)

# 在不计算梯度的情况下，计算上下文向量
with torch.no_grad():
    context_vecs = ca_with_buffer(batch)

# 打印上下文向量
print(context_vecs)

# 打印不带缓冲区的因果自注意力机制的状态字典
ca_without_buffer.state_dict()

# 打印带缓冲区的因果自注意力机制的状态字典
ca_with_buffer.state_dict()

# 修改带缓冲区的因果自注意力机制的掩码值
ca_with_buffer.mask[ca_with_buffer.mask == 1.] = 2.
# 打印修改后的掩码
ca_with_buffer.mask

# 保存带缓冲区的因果自注意力机制的状态字典到文件
torch.save(ca_with_buffer.state_dict(), "model.pth")

# 创建新的带缓冲区的因果自注意力机制实例
new_ca_with_buffer = CausalAttentionWithBuffer(d_in, d_out, context_length, 0.0)
# 加载保存的状态字典
new_ca_with_buffer.load_state_dict(torch.load("model.pth"))

# 打印新实例的掩码
new_ca_with_buffer.mask

# 修改不带缓冲区的因果自注意力机制的掩码值
ca_without_buffer.mask[ca_without_buffer.mask == 1.] = 2.

# 保存不带缓冲区的因果自注意力机制的状态字典到文件
torch.save(ca_without_buffer.state_dict(), "model.pth")

# 创建新的不带缓冲区的因果自注意力机制实例
new_ca_without_buffer = CausalAttentionWithoutBuffers(d_in, d_out, context_length, 0.0)
# 加载保存的状态字典
new_ca_without_buffer.load_state_dict(torch.load("model.pth"))

# 打印新实例的掩码
new_ca_without_buffer.mask
