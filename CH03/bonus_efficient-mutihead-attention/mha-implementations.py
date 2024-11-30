# 导入PyTorch库
import torch

# 设置随机种子以确保结果可复现
torch.manual_seed(123)
# 检测是否支持CUDA，如果支持则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 打印PyTorch的版本号
print(f"PyTorch version: {torch.__version__}")

# 定义批次大小、上下文长度、嵌入维度
batch_size = 8
context_len = 1024
embed_dim = 768
# 创建一个随机初始化的嵌入张量
embeddings = torch.randn((batch_size, context_len, embed_dim), device=device)

# 导入PyTorch的神经网络模块
import torch.nn as nn

# 定义CausalAttention类，实现因果自注意力机制
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        # 定义查询、键和值的线性变换层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 定义dropout层
        self.dropout = nn.Dropout(dropout)
        # 注册一个上三角掩码buffer，用于在自注意力中实现因果关系
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # 提取输入张量x的批处理大小、token数量和特征维度
        b, num_tokens, d_in = x.shape
        # 通过线性层计算查询、键和值
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 计算查询和键的点积，然后对结果进行转置以匹配维度
        attn_scores = queries @ keys.transpose(1, 2)
        # 使用上三角掩码将未来信息的注意力分数设置为负无穷，以实现因果自注意力
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # 应用softmax函数并缩放点积分数，然后应用dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算加权的值（context vector）作为自注意力的输出
        context_vec = attn_weights @ values
        return context_vec

# 定义Ch03_MHA_Wrapper类，用于实现多头自注意力机制的包装类
class Ch03_MHA_Wrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 创建一个模块列表，包含多个CausalAttention实例，每个实例代表一个注意力头
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )
        # 定义一个线性层，用于将多个头的输出合并
        self.out_proj = nn.Linear(d_out*num_heads, d_out*num_heads)

    def forward(self, x):
        # 遍历每个头，计算上下文向量，并将结果沿着最后一个维度拼接
        context_vec = torch.cat([head(x) for head in self.heads], dim=-1)
        # 通过输出投影层返回最终的上下文向量
        return self.out_proj(context_vec)
# 创建Ch03_MHA_Wrapper实例，并将其发送到GPU或CPU
mha_ch03_wrapper = Ch03_MHA_Wrapper(
    d_in=embed_dim,
    d_out=embed_dim//12,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

# 通过多头自注意力包装器传递嵌入张量，并打印输出的形状
out = mha_ch03_wrapper(embeddings)
print(out.shape)

# 定义Ch03_MHA类，实现多头自注意力机制
class Ch03_MHA(nn.Module):
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
        self.out_proj = nn.Linear(d_out, d_out)
        # 定义dropout层
        self.dropout = nn.Dropout(dropout)
        # 注册一个上三角掩码buffer，用于在自注意力中实现因果关系
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # 提取输入张量x的批处理大小、token数量和特征维度
        b, num_tokens, d_in = x.shape

        # 通过线性层计算查询、键和值
        keys = self.W_key(x)
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
        attn_scores = queries @ keys.transpose(2, 3)
        # 将掩码截断到token数量，并转换为布尔值
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # 使用掩码填充注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 应用softmax函数并缩放点积分数，然后应用dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算加权的值（context vector）作为自注意力的输出
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并头输出，并将结果转换为原始的输出维度
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
# 创建Ch03_MHA实例，并将模型发送到GPU或CPU
mha_ch03 = Ch03_MHA(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

# 将随机初始化的嵌入张量传递给Ch03_MHA模型，并打印输出的形状
out = mha_ch03(embeddings)
print(out.shape)

# 导入PyTorch的神经网络模块
import torch.nn as nn

# 定义MultiHeadAttentionCombinedQKV类，实现合并QKV的多头自注意力机制
class MultiHeadAttentionCombinedQKV(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        # 确保嵌入维度可以被头数整除
        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        # 定义头数、上下文长度和每个头的维度
        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads

        # 定义一个线性层，用于合并QKV的计算
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        # 定义一个线性层，用于最终的投影
        self.proj = nn.Linear(d_out, d_out)
        # 定义dropout层
        self.dropout = nn.Dropout(dropout)

        # 注册一个上三角掩码buffer，用于在自注意力中实现因果关系
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        # 提取输入张量x的批处理大小、token数量和特征维度
        batch_size, num_tokens, embed_dim = x.shape

        # 通过QKV线性层计算查询、键和值
        qkv = self.qkv(x)

        # 重塑张量，将查询、键和值分开，并为多头自注意力做准备
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # 重新排序张量，以适应多头自注意力的计算
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # 分离查询、键和值
        queries, keys, values = qkv.unbind(0)

        # 计算查询和键的点积，得到注意力分数
        attn_scores = queries @ keys.transpose(-2, -1)
        # 应用因果掩码，将未来信息的注意力分数设置为负无穷
        attn_scores = attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        # 应用softmax函数并缩放点积分数，然后应用dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**-0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算加权的值（context vector）作为自注意力的输出
        context_vec = attn_weights @ values

        # 重新排序context vector，以匹配原始的输出维度
        context_vec = context_vec.transpose(1, 2)

        # 将多头自注意力的输出重塑为原始的输出维度
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, embed_dim)

        # 通过最终的投影层返回输出
        context_vec = self.proj(context_vec)

        return context_vec

# 创建MultiHeadAttentionCombinedQKV实例，并将其发送到GPU或CPU
mha_combined_qkv = MultiHeadAttentionCombinedQKV(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

# 将随机初始化的嵌入张量传递给MultiHeadAttentionCombinedQKV模型，并打印输出的形状
out = mha_combined_qkv(embeddings)
print(out.shape)
# 导入math库，用于初始化参数
import math

# 定义MHAEinsum类，使用Einsum实现多头自注意力机制
class MHAEinsum(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 确保输出维度可以被头数整除
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        # 定义输出维度、头数和每个头的维度
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # 初始化Q、K、V的参数
        self.W_query = nn.Parameter(torch.randn(d_out, d_in))
        self.W_key = nn.Parameter(torch.randn(d_out, d_in))
        self.W_value = nn.Parameter(torch.randn(d_out, d_in))

        # 如果需要偏置，则初始化偏置参数
        if qkv_bias:
            self.bias_q = nn.Parameter(torch.zeros(d_out))
            self.bias_k = nn.Parameter(torch.zeros(d_out))
            self.bias_v = nn.Parameter(torch.zeros(d_out))
        else:
            # 如果不需要偏置，则注册为None
            self.register_parameter("bias_q", None)
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        # 定义输出投影层
        self.out_proj = nn.Linear(d_out, d_out)
        # 定义dropout层
        self.dropout = nn.Dropout(dropout)
        # 注册一个上三角掩码buffer，用于在自注意力中实现因果关系
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        # 初始化参数
        self.reset_parameters()

# 初始化参数的函数
    def reset_parameters(self):
        # 使用Kaiming均匀初始化方法初始化Q、K、V参数
        nn.init.kaiming_uniform_(self.W_query, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_key, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_value, a=math.sqrt(5))
        # 如果存在偏置参数，则使用均匀分布初始化
        if self.bias_q is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_query)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_q, -bound, bound)
            nn.init.uniform_(self.bias_k, -bound, bound)
            nn.init.uniform_(self.bias_v, -bound, bound)

    # 定义MHAEinsum类的前向传播方法
    def forward(self, x):
        # 提取输入张量x的批处理大小、序列长度和特征维度
        b, n, _ = x.shape

        # 使用einsum进行线性变换，计算Q（查询）、K（键）、V（值）
        Q = torch.einsum("bnd,di->bni", x, self.W_query)
        K = torch.einsum("bnd,di->bni", x, self.W_key)
        V = torch.einsum("bnd,di->bni", x, self.W_value)

        # 如果使用偏置，则将偏置加到Q、K、V上
        if self.bias_q is not None:
            Q += self.bias_q
            K += self.bias_k
            V += self.bias_v

        # 为多头注意力重塑Q、K、V，并将它们转置以匹配期望的维度
        Q = Q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        # 使用einsum计算缩放点积注意力分数
        scores = torch.einsum("bhnd,bhmd->bhnm", Q, K) / (self.head_dim ** 0.5)

        # 应用上三角掩码，将未来信息的注意力分数设置为负无穷
        mask = self.mask[:n, :n].unsqueeze(0).unsqueeze(1).expand(b, self.num_heads, n, n)
        scores = scores.masked_fill(mask.bool(), -torch.inf)

        # 应用softmax函数并执行dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 使用einsum和注意力权重聚合上下文向量
        context_vec = torch.einsum("bhnm,bhmd->bhnd", attn_weights, V)

        # 转置并重塑context_vec以匹配原始输出维度
        context_vec = context_vec.transpose(1, 2).reshape(b, n, self.d_out)
        # 通过输出投影层返回最终的上下文向量
        context_vec = self.out_proj(context_vec)

        return context_vec

# 创建MHAEinsum实例，并将其发送到GPU或CPU
mha_einsum = MHAEinsum(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

# 将随机初始化的嵌入张量传递给MHAEinsum模型，并打印输出的形状
out = mha_einsum(embeddings)
print(out.shape)

# 定义MHAPyTorchScaledDotProduct类，使用PyTorch内置的scaled_dot_product_attention函数实现多头自注意力
class MHAPyTorchScaledDotProduct(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        # 确保输出维度可以被头数整除
        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        # 定义一个线性层，用于合并QKV的计算
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
            # 定义一个线性层，用于最终的投影
        self.proj = nn.Linear(d_out, d_out)
            # 定义dropout率
        self.dropout = dropout

        def forward(self, x):
            # 提取输入张量x的批处理大小、序列长度和特征维度
            batch_size, num_tokens, embed_dim = x.shape

            # 通过QKV线性层计算查询、键和值
            qkv = self.qkv(x)

            # 重塑张量，将查询、键和值分开，并为多头自注意力做准备
            qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

            # 重新排序张量，以适应多头自注意力的计算
            qkv = qkv.permute(2, 0, 3, 1, 4)

            # 分离查询、键和值
            queries, keys, values = qkv

            # 如果模型处于训练模式，则使用dropout，否则dropout率为0
            use_dropout = 0. if not self.training else self.dropout

            # 使用PyTorch内置的scaled_dot_product_attention函数计算多头自注意力
            context_vec = nn.functional.scaled_dot_product_attention(
                queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

            # 转置并重塑context_vec以匹配原始输出维度
            context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

            # 通过输出投影层返回最终的上下文向量
            context_vec = self.proj(context_vec)

            return context_vec

# 创建MHAPyTorchScaledDotProduct实例，并将其发送到GPU或CPU
mha_pytorch_scaled = MHAPyTorchScaledDotProduct(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)
# 将随机初始化的嵌入张量传递给MHAPyTorchScaledDotProduct模型，并打印输出的形状
out = mha_pytorch_scaled(embeddings)
print(out.shape)
# 定义MHAPyTorchScaledDotProduct类，使用PyTorch内置的scaled_dot_product_attention函数实现多头自注意力
class MHAPyTorchScaledDotProduct(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        # 确保输出维度可以被头数整除
        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        # 定义头数、上下文长度和每个头的维度
        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        # 定义一个线性层，用于合并QKV的计算
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        # 定义一个线性层，用于最终的投影
        self.proj = nn.Linear(d_out, d_out)
        # 定义dropout率
        self.dropout = dropout

    def forward(self, x):
        # 提取输入张量x的批处理大小、序列长度和特征维度
        batch_size, num_tokens, embed_dim = x.shape

        # 通过QKV线性层计算查询、键和值
        qkv = self.qkv(x)

        # 重塑张量，将查询、键和值分开，并为多头自注意力做准备
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # 重新排序张量，以适应多头自注意力的计算
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # 分离查询、键和值
        queries, keys, values = qkv

        # 如果模型处于训练模式，则使用dropout，否则dropout率为0
        use_dropout = 0. if not self.training else self.dropout

        # 使用PyTorch内置的scaled_dot_product_attention函数计算多头自注意力
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # 转置并重塑context_vec以匹配原始输出维度
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        # 通过输出投影层返回最终的上下文向量
        context_vec = self.proj(context_vec)

        return context_vec

# 创建MHAPyTorchScaledDotProduct实例，并将其发送到GPU或CPU
mha_pytorch_scaled = MHAPyTorchScaledDotProduct(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

# 将随机初始化的嵌入张量传递给MHAPyTorchScaledDotProduct模型，并打印输出的形状
out = mha_pytorch_scaled(embeddings)
print(out.shape)

# 定义MHAPyTorchSDPAWithoutFlash类，使用PyTorch内置的scaled_dot_product_attention函数实现多头自注意力，但不使用FlashAttention
class MHAPyTorchSDPAWithoutFlash(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        # 确保输出维度可以被头数整除
        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        # 定义头数、上下文长度和每个头的维度
        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        # 定义一个线性层，用于合并QKV的计算
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        # 定义一个线性层，用于最终的投影
        self.proj = nn.Linear(d_out, d_out)
        # 定义dropout率
        self.dropout = dropout
        # 注册一个上三角掩码buffer，用于在自注意力中实现因果关系
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        # 提取输入张量x的批处理大小、序列长度和特征维度
        batch_size, num_tokens, embed_dim = x.shape

        # 通过QKV线性层计算查询、键和值
        qkv = self.qkv(x)

        # 重塑张量，将查询、键和值分开，并为多头自注意力做准备
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # 重新排序张量，以适应多头自注意力的计算
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # 分离查询、键和值
        queries, keys, values = qkv

        # 如果模型处于训练模式，则使用dropout，否则dropout率为0
        use_dropout = 0. if not self.training else self.dropout

        # 确保attn_mask与预期的形状兼容，并且batch_first=True
        # 不需要手动调整num_heads；确保它对序列是正确的
        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]

        # 使用PyTorch内置的scaled_dot_product_attention函数计算多头自注意力，应用掩码
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=attn_mask, dropout_p=use_dropout, is_causal=False)

        # 转置并重塑context_vec以匹配原始输出维度
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        # 通过输出投影层返回最终的上下文向量
        context_vec = self.proj(context_vec)

        return context_vec

# 创建MHAPyTorchSDPAWithoutFlash实例，并将其发送到GPU或CPU
mha_pytorch_sdpa_no_flash = MHAPyTorchSDPAWithoutFlash(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

# 将随机初始化的嵌入张量传递给MHAPyTorchSDPAWithoutFlash模型，并打印输出的形状
out = mha_pytorch_sdpa_no_flash(embeddings)
print(out.shape)

# 导入PyTorch的神经网络模块
import torch.nn as nn

# 定义MHAPyTorchClass类，使用PyTorch内置的MultiheadAttention类实现多头自注意力
class MHAPyTorchClass(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False, need_weights=True):
        super().__init__()

        # 定义上下文长度
        self.context_length = context_length
        # 初始化MultiheadAttention模块
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_out,
            num_heads=num_heads,
            dropout=dropout,
            bias=qkv_bias,
            add_bias_kv=qkv_bias,
            batch_first=True,
        )

        # 定义是否需要返回注意力权重
        self.need_weights = need_weights
        # 定义一个线性层，用于最终的投影
        self.proj = nn.Linear(d_out, d_out)
        # 注册一个上三角掩码buffer，用于在自注意力中实现因果关系
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        # 提取输入张量x的批处理大小、序列长度和特征维度
        batch_size, num_tokens, _ = x.shape

        # 确保attn_mask与预期的形状兼容，并且batch_first=True
        # 不需要手动调整num_heads；确保它对序列是正确的
        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]

        # attn_mask广播将隐式处理batch_size维度
        # 使用MultiheadAttention模块计算多头自注意力，可能返回注意力权重
        attn_output, _ = self.multihead_attn(
            x, x, x, attn_mask=attn_mask, need_weights=self.need_weights
        )

        # 通过输出投影层返回最终的上下文向量
        output = self.proj(attn_output)

        return output

# 创建MHAPyTorchClass实例，并将其发送到GPU或CPU
mha_pytorch_class_default = MHAPyTorchClass(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

# 将随机初始化的嵌入张量传递给MHAPyTorchClass模型，并打印输出的形状
out = mha_pytorch_class_default(embeddings)
print(out.shape)

# 创建MHAPyTorchClass实例，不返回注意力权重，并将其发送到GPU或CPU
mha_pytorch_class_noweights = MHAPyTorchClass(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False,
    need_weights=False # NEW!
).to(device)

# 将随机初始化的嵌入张量传递给不返回权重的MHAPyTorchClass模型，并打印输出的形状
out = mha_pytorch_class_noweights(embeddings)
print(out.shape)

# 导入packaging.version模块，用于版本号的解析
from packaging.version import parse as parse_version

# 定义normalize_version函数，用于规范化版本号
def normalize_version(version):
    parsed_version = parse_version(version)
    return parse_version(f"{parsed_version.major}.{parsed_version.minor}.{parsed_version.micro}")

# 获取当前PyTorch版本的规范化版本号
current_version = normalize_version(torch.__version__)
MIN_TORCH_VERSION = "2.5.0"
required_version = parse_version(MIN_TORCH_VERSION)

# 检查当前PyTorch版本是否满足最小版本要求
if current_version >= required_version:
    # 如果版本满足要求，则导入flex_attention和create_block_mask函数
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# 定义causal函数，用于生成因果掩码
def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx
# 定义MHAPyTorchFlexAttention类，使用PyTorch的FlexAttention实现多头自注意力
class MHAPyTorchFlexAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        # 确保输出维度可以被头数整除
        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        # 定义头数、上下文长度和每个头的维度
        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        # 定义一个线性层，用于合并QKV的计算
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        # 定义一个线性层，用于最终的投影
        self.proj = nn.Linear(d_out, d_out)
        # 定义dropout率
        self.dropout = dropout
        # 创建一个块掩码，用于在自注意力中实现因果关系和其他掩码操作
        # self.register_buffer("block_mask", create_block_mask(causal, B=None, H=None, Q_LEN=context_length, KV_LEN=context_length))
        # `create_block_mask` function does not support buffers, yet
        self.block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=context_length, KV_LEN=context_length)

    def forward(self, x):
        # 提取输入张量x的批处理大小、序列长度和特征维度
        batch_size, num_tokens, embed_dim = x.shape

        # 通过QKV线性层计算查询、键和值
        qkv = self.qkv(x)

        # 重塑张量，将查询、键和值分开，并为多头自注意力做准备
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # 重新排序张量，以适应多头自注意力的计算
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # 分离查询、键和值
        queries, keys, values = qkv

        # 如果模型处于训练模式，则使用dropout，否则dropout率为0
        use_dropout = 0. if not self.training else self.dropout

        # 确保attn_mask与预期的形状兼容，并且batch_first=True
        # 不需要手动调整num_heads；确保它对序列是正确的
        if self.context_length >= num_tokens:
            attn_mask = self.block_mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.block_mask[:self.context_length, :self.context_length]

        # 使用flex_attention函数计算多头自注意力，应用块掩码
        context_vec = flex_attention(queries, keys, values, block_mask=attn_mask)

        # 转置并重塑context_vec以匹配原始输出维度
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        # 通过输出投影层返回最终的上下文向量
        context_vec = self.proj(context_vec)

        return context_vec

# 检查当前PyTorch版本是否满足最小版本要求，并且是否可用CUDA
if current_version >= required_version and torch.cuda.is_available():
    # 创建MHAPyTorchFlexAttention实例，并将其发送到GPU或CPU
    mha_pytorch_flex = MHAPyTorchFlexAttention(
        d_in=embed_dim,
        d_out=embed_dim,
        context_length=context_len,
        dropout=0.0,
        num_heads=12,
        qkv_bias=False
    ).to(device)

    # 将随机初始化的嵌入张量传递给MHAPyTorchFlexAttention模型，并打印输出的形状
    out = mha_pytorch_flex(embeddings)
    print(out.shape)

# 设置float32矩阵乘法的精度为“high”以启用Tensor Cores
torch.set_float32_matmul_precision("high")

# 设置随机种子以确保结果可复现
torch.manual_seed(123)
# 检测是否支持CUDA，如果支持则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 打印PyTorch的版本号
print(f"PyTorch version: {torch.__version__}")
# 打印正在运行的设备
print(f"Running on {device}")
import timeit

# 假设以下函数已经被定义：
# mha_ch03_wrapper, mha_ch03, mha_combined_qkv, mha_einsum, mha_pytorch_scaled,
# mha_pytorch_sdpa_no_flash, mha_pytorch_class_default, mha_pytorch_class_noweights, mha_pytorch_flex

# 假设embeddings是已经定义好的输入数据
# embeddings = ...
# 使用timeit模块测量不同多头自注意力实现的性能

# 1) 使用第三章中介绍的CausalAttention MHA包装类
mha_ch03_wrapper_time = timeit.timeit('mha_ch03_wrapper(embeddings)', globals=globals(), number=100)
print(f"mha_ch03_wrapper: {mha_ch03_wrapper_time}")

# 2) 使用第三章中的多头自注意力类
mha_ch03_time = timeit.timeit('mha_ch03(embeddings)', globals=globals(), number=100)
print(f"mha_ch03: {mha_ch03_time}")

# 3) 使用结合了权重的另一种多头自注意力
mha_combined_qkv_time = timeit.timeit('mha_combined_qkv(embeddings)', globals=globals(), number=100)
print(f"mha_combined_qkv: {mha_combined_qkv_time}")

# 4) 使用爱因斯坦求和符号的多头自注意力
mha_einsum_time = timeit.timeit('mha_einsum(embeddings)', globals=globals(), number=100)
print(f"mha_einsum: {mha_einsum_time}")

# 5) 使用PyTorch内置的scaled dot product attention的多头自注意力
mha_pytorch_scaled_time = timeit.timeit('mha_pytorch_scaled(embeddings)', globals=globals(), number=100)
print(f"mha_pytorch_scaled: {mha_pytorch_scaled_time}")

# 6) 没有FlashAttention的多头自注意力
mha_pytorch_sdpa_no_flash_time = timeit.timeit('mha_pytorch_sdpa_no_flash(embeddings)', globals=globals(), number=100)
print(f"mha_pytorch_sdpa_no_flash: {mha_pytorch_sdpa_no_flash_time}")

# 7) 使用PyTorch的torch.nn.MultiheadAttention类的默认设置
mha_pytorch_class_default_time = timeit.timeit('mha_pytorch_class_default(embeddings)', globals=globals(), number=100)
print(f"mha_pytorch_class_default: {mha_pytorch_class_default_time}")

# 8) 使用PyTorch的torch.nn.MultiheadAttention类，禁用`need_weights`
mha_pytorch_class_noweights_time = timeit.timeit('mha_pytorch_class_noweights(embeddings)', globals=globals(), number=100)
print(f"mha_pytorch_class_noweights: {mha_pytorch_class_noweights_time}")

# 9) 使用PyTorch的FlexAttention（需要PyTorch 2.5.0或更新版本）
# 由于代码被注释掉，所以这一部分不会被执行
#mha_pytorch_flex_time = timeit.timeit('mha_pytorch_flex(embeddings)', globals=globals(), number=100)
#print(f"mha_pytorch_flex: {mha_pytorch_flex_time}")

# 设置随机种子以确保结果可复现
torch.manual_seed(123)
# 检测是否支持CUDA，如果支持则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 打印PyTorch的版本号
print(f"PyTorch version: {torch.__version__}")
# 打印正在运行的设备
print(f"Running on {device}")

# 定义一个字典，将每种多头自注意力实现与其名称关联起来
functions = {
    "1) MHA wrapper class": mha_ch03_wrapper,
    "2) MHA Ch03": mha_ch03,
    "3) MHA with combined QKV weights": mha_combined_qkv,
    "4) MHA with Einsum": mha_einsum,
    "5) MHA with PyTorch scaled_dot_product_attention": mha_pytorch_scaled,
    "6) PyTorch's SDPA, no FlashAttention": mha_pytorch_sdpa_no_flash,
    "7) PyTorch MHA class defaults": mha_pytorch_class_default,
    "8) PyTorch MHA with need_weights=False": mha_pytorch_class_noweights
}

# 如果当前PyTorch版本支持FlexAttention，则将其添加到字典中
#if current_version >= required_version:
 #   functions["8) PyTorch's FlexAttention"] =  mha_pytorch_flex

# 导入matplotlib库用于绘图
import matplotlib.pyplot as plt

# 自定义matplotlib的配置，以适应暗色模式美学
plt.rcParams["figure.facecolor"] = "#121212"
plt.rcParams["axes.facecolor"] = "#121212"
plt.rcParams["axes.edgecolor"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["text.color"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["grid.color"] = "#444444"
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.markersize"] = 8
# 定义一个函数，用于绘制不同函数的执行时间
def plot_execution_times(functions, execution_means, execution_stds, filename):
    # 创建一个图表
    fig, ax = plt.subplots()
    # 绘制条形图，显示每个函数的平均执行时间，并添加误差棒
    bars = ax.bar(functions.keys(), execution_means, yerr=execution_stds, capsize=5, error_kw={'ecolor': 'grey'})

    # 设置y轴标签为“执行时间（毫秒）”
    plt.ylabel("Execution time (ms)")
    # 设置x轴标签旋转45度，使其更容易阅读
    plt.xticks(rotation=45, ha="right")

    # 计算新的y轴上限，增加40%的余量
    max_execution_time = max(execution_means)
    upper_ylim = max_execution_time + 0.4 * max_execution_time  # Adding a 40% margin
    plt.ylim(0, upper_ylim)

    # 在每个条形图上标注执行时间
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (0.05 * upper_ylim), round(yval, 2), ha="center", va="bottom")

    # 调整布局并保存图表
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# 以下代码是从Andrei Aksionov和GitHub上的cuda-mode/lectures共享的CUDA基准测试代码中提取的

# 导入numpy库，用于数值计算
import numpy as np

# 定义一个函数，用于测量PyTorch函数的执行时间
def time_pytorch_function(func, *input, num_repeats=1_000):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 预热，运行函数5次以减少初始化的影响
    for _ in range(5):
        func(*input)
    torch.cuda.synchronize()

    times = []
    # 重复运行函数num_repeats次，并记录每次的执行时间
    for _ in range(num_repeats):
        start.record()
        func(*input)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    # 返回平均执行时间和标准差
    return np.mean(times), np.std(times)

# 测量每个函数的执行时间，并存储平均值和标准差
execution_stats = [time_pytorch_function(fn, embeddings) for fn in functions.values()]
execution_means = [stat[0] for stat in execution_stats]
execution_stds = [stat[1] for stat in execution_stats]

# 绘制并保存执行时间图表
plot_execution_times(functions, execution_means, execution_stds, filename="1_forward-only.pdf")

# 定义一个函数，用于执行前向和反向传播
def forward_backward(func, embeddings):
    if embeddings.grad is not None:
        embeddings.grad.zero_()
    output = func(embeddings)
    loss = output.sum()
    loss.backward()

# 定义一个函数，用于测量包含前向和反向传播的PyTorch函数的执行时间
def time_pytorch_function_forward_backward(func, *input, num_repeats = 1_000):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 预热，运行前向和反向传播5次
    for _ in range(5):
        forward_backward(func, *input)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_repeats):
        start.record()
        forward_backward(func, *input)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return np.mean(times), np.std(times)

# 测量每个函数的前向和反向传播执行时间，并存储平均值和标准差
execution_stats = [time_pytorch_function_forward_backward(fn, embeddings) for fn in functions.values()]
execution_means = [stat[0] for stat in execution_stats]
execution_stds = [stat[1] for stat in execution_stats]

# 绘制并保存前向和反向传播执行时间图表
plot_execution_times(functions, execution_means, execution_stds, filename="2_forward-and-backward.pdf")

# 导入torch._dynamo模块，用于动态编译PyTorch代码
import torch._dynamo
# 配置torch._dynamo以忽略错误
torch._dynamo.config.suppress_errors = True

# 定义一个函数，用于准备函数以进行动态编译
def prepare_function(fn):
    fn = torch.compile(fn)
    return fn

# 测量每个函数的前向和反向传播执行时间（经过动态编译），并存储平均值和标准差
execution_stats = [time_pytorch_function_forward_backward(prepare_function(fn), embeddings) for fn in functions.values()]
execution_means = [stat[0] for stat in execution_stats]
execution_stds = [stat[1] for stat in execution_stats]

# 绘制并保存经过动态编译的前向和反向传播执行时间图表
plot_execution_times(functions, execution_means, execution_stds, filename="3_forward-and-backward-compiled.pdf")






