
from importlib.metadata import version  # 从importlib.metadata模块导入version函数，用于获取库的版本信息

print("torch version:", version("torch")) # 打印PyTorch库的版本

import torch        # 导入PyTorch库

# 创建一个二维张量（矩阵），其中每一行代表一个输入向量，这里用(x^1)到(x^6)表示
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]  # 选择第二个输入向量作为查询向量（query）


attn_scores_2 = torch.empty(inputs.shape[0])    # 创建一个空张量，用于存储注意力分数
for i, x_i in enumerate(inputs):                # 遍历所有输入向量
    attn_scores_2[i] = torch.dot(x_i, query)    # 计算每个输入向量与查询向量的点积（注意力分数）
print(attn_scores_2)                            # 打印注意力分数

res = 0.                                        # 初始化结果变量

for idx, element in enumerate(inputs[0]):       # 遍历第一个输入向量的每个元素
    res += inputs[0][idx] * query[idx]          # 计算第一个输入向量与查询向量的点积

print(res)                                      # 打印点积结果
print(torch.dot(inputs[0], query))              # 使用torch.dot函数计算点积，并打印结果

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum() # 将注意力分数归一化，得到注意力权重

print("Attention weights:", attn_weights_2_tmp) # 打印注意力权重
print("Sum:", attn_weights_2_tmp.sum())         # 打印注意力权重的和，理论上应该接近1

def softmax_naive(x):                               # 定义一个softmax函数的简单实现
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2) # 使用自定义的softmax函数计算注意力权重

print("Attention weights:", attn_weights_2_naive)   # 打印使用自定义softmax函数计算的注意力权重和它们的和
print("Sum:", attn_weights_2_naive.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)# 使用PyTorch内置的softmax函数计算注意力权重

print("Attention weights:", attn_weights_2)         # 打印使用PyTorch内置softmax函数计算的注意力权重和它们的和
print("Sum:", attn_weights_2.sum())

query = inputs[1]                                   # 选择第二个输入向量作为查询向量

context_vec_2 = torch.zeros(query.shape)            # 初始化一个上下文向量，其形状与查询向量相同
for i,x_i in enumerate(inputs):                     # 计算上下文向量，通过将每个输入向量乘以对应的注意力权重并求和
    context_vec_2 += attn_weights_2[i]*x_i

print(context_vec_2)                                # 打印计算得到的上下文向量

attn_scores = torch.empty(6, 6)                     # 初始化一个6x6的张量，用于存储所有输入向量之间的注意力分数

for i, x_i in enumerate(inputs):                    # 计算每对输入向量之间的点积，填充到attn_scores张量中
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)                                  # 打印所有输入向量之间的注意力分数

attn_scores = inputs @ inputs.T                     # 使用矩阵乘法计算所有输入向量之间的点积，填充到attn_scores张量中
print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)   # 使用PyTorch内置的softmax函数计算所有注意力分数的权重
print(attn_weights)

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])# 计算第二行的权重和
print("Row 2 sum:", row_2_sum)

print("All row sums:", attn_weights.sum(dim=-1))    # 计算所有行的权重和

all_context_vecs = attn_weights @ inputs            # 计算所有上下文向量
print(all_context_vecs)

print("Previous 2nd context vector:", context_vec_2) # 打印之前计算的第二个上下文向量

x_2 = inputs[1]             # 这行代码取出输入数据inputs的第二个元素，赋值给x_2
d_in = inputs.shape[1]      # 这行代码获取输入数据inputs的第二维的大小，即输入嵌入的维度d_in
d_out = 2                   # 定义输出嵌入的维度d_out为2

torch.manual_seed(123)      # 设置PyTorch的随机种子为123，以确保每次运行代码时生成的随机数是相同的

# 这三行代码创建了三个参数矩阵W_query、W_key、W_value，它们都是随机初始化的，并且设置requires_grad=False，意味着在训练过程中不会更新这些参数。
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query # 计算x_2与W_query的矩阵乘法，得到查询向量query_2。
key_2 = x_2 @ W_key     # 计算x_2与W_key的矩阵乘法，得到键向量key_2
value_2 = x_2 @ W_value # 计算x_2与W_value的矩阵乘法，得到值向量value_2

print(query_2)          # 打印查询向量query_2

keys = inputs @ W_key
values = inputs @ W_value
# 计算所有输入元素与键和值矩阵的矩阵乘法，得到所有键向量keys和所有值向量values
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
# 打印键向量keys和值向量values的形状
keys_2 = keys[1]                    # 取出所有键向量keys中的第二个元素，赋值给keys_2
attn_score_22 = query_2.dot(keys_2) # 计算查询向量query_2与键向量keys_2的点积，得到注意力分数attn_score_22
print(attn_score_22)

attn_scores_2 = query_2 @ keys.T # 计算查询向量query_2与所有键向量的转置矩阵的矩阵乘法，得到所有注意力分数attn_scores_2
print(attn_scores_2)

d_k = keys.shape[1]             # 获取键向量的维度d_k
# 对所有注意力分数attn_scores_2进行缩放（除以d_k的平方根），然后应用softmax函数，得到注意力权重attn_weights_2
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

context_vec_2 = attn_weights_2 @ values # 计算注意力权重attn_weights_2与所有值向量的矩阵乘法，得到上下文向量context_vec_2
print(context_vec_2)

import torch.nn as nn   # 导入PyTorch的神经网络模块torch.nn


class SelfAttention_v1(nn.Module):
# 定义一个名为SelfAttention_v1的类，它继承自nn.Module，这是PyTorch中所有神经网络模块的基类
    def __init__(self, d_in, d_out):
        super().__init__()
        # 定义SelfAttention_v1类的构造函数，它接受两个参数：
        # d_in（输入嵌入的维度）和d_out（输出嵌入的维度）。super().__init__()调用基类的构造函数
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        # 在构造函数中，定义了三个参数矩阵W_query、W_key和W_value，
        # 它们都是随机初始化的，并被包装成nn.Parameter，这样它们就可以在训练过程中被优化
    def forward(self, x):
        # 定义SelfAttention_v1类的前向传播函数forward，它接受一个输入x
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        # 计算输入x与键、查询和值矩阵的矩阵乘法，分别得到键向量keys、查询向量queries和值向量values。
        attn_scores = queries @ keys.T
        # 计算查询向量queries与键向量keys转置的矩阵乘法，得到所有注意力分数attn_scores
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        # 对注意力分数attn_scores进行缩放（除以键向量维度的平方根），然后应用softmax函数，得到注意力权重attn_weights
        context_vec = attn_weights @ values
        # 计算注意力权重attn_weights与值向量values的矩阵乘法，得到上下文向量context_vec
        return context_vec


torch.manual_seed(123) # 设置PyTorch的随机种子为123，以确保每次运行代码时生成的随机数是相同的
sa_v1 = SelfAttention_v1(d_in, d_out)
# 创建SelfAttention_v1类的实例sa_v1，传入输入维度d_in和输出维度d_out
print(sa_v1(inputs))
# 将输入数据inputs传递给sa_v1的前向传播函数，并打印输出结果。
# 这里假设inputs已经被定义并且是一个合适的PyTorch张量
class SelfAttention_v2(nn.Module):
# 定义一个名为SelfAttention_v2的类，它继承自nn.Module

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # d_in（输入嵌入的维度），d_out（输出嵌入的维度），以及qkv_bias（一个布尔值，用于决定是否在线性层中使用偏置项）。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 在构造函数中，定义了三个线性层W_query、W_key和W_value，它们都是具有偏置项的线性变换
    def forward(self, x):
        # 定义SelfAttention_v2类的前向传播函数forward，它接受一个输入x
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # 通过三个线性层计算输入x的键、查询和值
        attn_scores = queries @ keys.T # 计算查询和键的点积，得到注意力分数
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # 对注意力分数进行缩放和softmax操作，得到注意力权重
        context_vec = attn_weights @ values # 计算注意力权重和值的点积，得到上下文向量
        return context_vec


torch.manual_seed(789) # 设置PyTorch的随机种子为789
sa_v2 = SelfAttention_v2(d_in, d_out) # 创建SelfAttention_v2类的实例sa_v2
print(sa_v2(inputs))

# 重用前一节中的SelfAttention_v2对象的查询和键权重矩阵以方便操作
queries = sa_v2.W_query(inputs) # 分别计算查询、键，然后计算查询和键的点积，得到注意力分数
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T

attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
# 对注意力分数进行缩放和softmax操作，得到注意力权重，并打印
print(attn_weights)

# 定义上下文长度，并创建一个下三角掩码矩阵
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

# 将注意力权重与下三角掩码矩阵相乘，得到掩码后的注意力权重
masked_simple = attn_weights*mask_simple
print(masked_simple)

# 计算掩码后注意力权重的行和，并进行规范化
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

# 创建一个上三角掩码矩阵
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
# 使用上三角掩码矩阵对注意力分数进行掩码，将掩码位置的分数设置为负无穷
print(masked)

# 对掩码后的注意力分数进行缩放和softmax操作，得到最终的注意力权重，并打印
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

torch.manual_seed(123)          # 设置PyTorch的随机种子为123，以确保每次运行代码时生成的随机数是相同的
dropout = torch.nn.Dropout(0.5) # 创建一个dropout层，丢弃率为50%
example = torch.ones(6, 6)      # 创建一个6x6的全1矩阵作为示例输入

print(dropout(example))

torch.manual_seed(123)          # 再次设置随机种子，确保结果的可重复性
print(dropout(attn_weights))

batch = torch.stack((inputs, inputs), dim=0) # 将两个inputs张量堆叠起来，形成一个batch，维度为0
print(batch.shape) # 打印batch的形状，这里应该是两个输入，每个输入有6个token，每个token的嵌入维度为3。
class CausalAttention(nn.Module):
# 定义一个名为CausalAttention的类，它继承自nn.Module。
    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        # 接受输入维度d_in、输出维度d_out、上下文长度context_length、dropout率dropout以及是否在线性层中使用偏置项qkv_bias
        self.d_out = d_out      # 保存输出维度d_out。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 定义三个线性层W_query、W_key和W_value，用于计算查询、键和值
        self.dropout = nn.Dropout(dropout) # 创建一个dropout层
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        # 注册一个buffer，用于存储上三角掩码矩阵，这是因果掩码，用于防止序列中的元素关注到未来的元素

    def forward(self, x):
    # 定义CausalAttention类的前向传播函数forward，它接受一个输入x
        b, num_tokens, d_in = x.shape # 获取输入x的形状，包括batch大小b、token数量num_tokens和输入维度d_in
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # 通过三个线性层计算输入x的键、查询和值
        attn_scores = queries @ keys.transpose(1, 2)
        # 计算查询和键的点积，注意这里使用了transpose来调整键的维度
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # 使用因果掩码对注意力分数进行掩码，将掩码位置的分数设置为负无穷
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        # 对注意力分数进行缩放和softmax操作，得到注意力权重
        attn_weights = self.dropout(attn_weights)
        # 对注意力权重应用dropout

        context_vec = attn_weights @ values
        # 计算注意力权重和值的点积，得到上下文向量
        return context_vec

torch.manual_seed(123)
# 设置PyTorch的随机种子为123，以确保每次运行代码时生成的随机数是相同的

context_length = batch.shape[1]  # 获取batch张量中的token数量，作为上下文长度context_length
ca = CausalAttention(d_in, d_out, context_length, 0.0)
# 创建一个CausalAttention类的实例ca
# 传入输入维度d_in、输出维度d_out、上下文长度context_length和dropout率为0.0
context_vecs = ca(batch)       # 将batch数据传递给ca的前向传播函数，并计算上下文向量

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

class MultiHeadAttentionWrapper(nn.Module):
# 定义一个名为MultiHeadAttentionWrapper的类，它继承自nn.Module
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 接受输入维度d_in、输出维度d_out、上下文长度context_length、dropout率dropout、
        # 头数num_heads以及是否在线性层中使用偏置项qkv_bias。
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )
        # 在构造函数中，创建一个模块列表self.heads，包含多个CausalAttention实例，每个实例对应一个注意力头
    def forward(self, x):
        # 定义MultiHeadAttentionWrapper类的前向传播函数forward，它接受一个输入x。
        return torch.cat([head(x) for head in self.heads], dim=-1)
        # 对每个头计算得到的上下文向量进行拼接，返回拼接后的张量

torch.manual_seed(123)  # 再次设置随机种子，确保结果的可重复性

context_length = batch.shape[1] # 再次获取batch张量中的token数量，作为上下文长度context_length
d_in, d_out = 3, 2              # 定义输入维度d_in为3，输出维度d_out为2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)
# 创建一个MultiHeadAttentionWrapper类的实例mha，
# 传入输入维度d_in、输出维度d_out、上下文长度context_length、dropout率为0.0和头数为2
context_vecs = mha(batch)   # 将batch数据传递给mha的前向传播函数，并计算上下文向量

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


class MultiHeadAttention(nn.Module):
    # 定义一个名为MultiHeadAttention的类，它继承自nn.Module。
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 接受输入维度d_in、输出维度d_out、上下文长度context_length、dropout率dropout、
        # 头数num_heads以及是否在线性层中使用偏置项qkv_bias。
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        # 断言输出维度d_out必须能够被头数num_heads整除
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 减少投影维度以匹配期望的输出维度。
        # 保存输出维度d_out、头数num_heads和每个头的维度head_dim
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 定义三个线性层W_query、W_key和W_value，用于计算查询、键和值
        self.out_proj = nn.Linear(d_out, d_out)  # 用于合并头输出的线性层
        self.dropout = nn.Dropout(dropout)       # 创建一个dropout层
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )
        # 注册一个buffer，用于存储上三角掩码矩阵，这是因果掩码，用于防止序列中的元素关注到未来的元素。
    def forward(self, x):
        # 定义MultiHeadAttention类的前向传播函数forward，它接受一个输入x
        b, num_tokens, d_in = x.shape
        # 获取输入x的形状，包括batch大小b、token数量num_tokens和输入维度d_in。

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # 通过三个线性层计算输入x的键、查询和值
        # "我们通过添加一个num_heads维度隐式地分割矩阵"
        # "展开最后一个维度：(b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # 调整键、查询和值的形状，为多头自注意力做准备
        # 转置：(b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        # 对键、查询和值进行转置，以便于计算
        # 计算带有因果掩码的缩放点积注意力（即自注意力）
        attn_scores = queries @ keys.transpose(2, 3)  # 计算每个头的查询和键的点积，得到注意力分数

        # 将原始掩码截断到token数量，并转换为布尔值
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码填充注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)# 对注意力分数进行缩放和softmax操作，得到注意力权重，并应用dropout

        # 形状：(b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # 计算注意力权重和值的点积，得到上下文向量，并进行转置
        # 合并头，其中self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 将上下文向量的形状调整为(batch大小, token数量, 输出维度)
        context_vec = self.out_proj(context_vec)  # 可选地，通过out_proj线性层对上下文向量进行投影

        return context_vec


torch.manual_seed(123)
# 设置PyTorch的随机种子为123，以确保每次运行代码时生成的随机数是相同的
batch_size, context_length, d_in = batch.shape
# 从batch张量中提取batch大小、上下文长度和输入维度
d_out = 2 # 设置输出维度d_out为2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
# 创建一个MultiHeadAttention类的实例mha，
# 传入输入维度d_in、输出维度d_out、上下文长度context_length、dropout率为0.0和头数为2
context_vecs = mha(batch)
# 将batch数据传递给mha的前向传播函数，并计算上下文向量
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# 接下来的张量a的形状，即(batch大小, 头数, token数量, 每个头的维度)= (1, 2, 3, 4)
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],

                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])
# 创建一个具有特定值的4维张量a，代表多头自注意力机制中的查询和键的输出
print(a @ a.transpose(2, 3))
# 计算张量a与其转置的矩阵乘法，这里实际上是计算了所有头的点积注意力分数
first_head = a[0, 0, :, :]              # 提取第一个头的张量
first_res = first_head @ first_head.T   # 计算第一个头的矩阵乘法，即点积注意力分数
print("First head:\n", first_res)       # 打印第一个头的点积注意力分数

second_head = a[0, 1, :, :]             # 提取第二个头的张量
second_res = second_head @ second_head.T# 计算第二个头的矩阵乘法，即点积注意力分数
print("\nSecond head:\n", second_res)

