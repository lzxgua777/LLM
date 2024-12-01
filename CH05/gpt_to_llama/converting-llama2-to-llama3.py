from importlib.metadata import version  # 导入版本检查模块

pkgs = [  # 定义需要检查版本的包列表
    "blobfile",         # 用于下载预训练权重的包
    "huggingface_hub",  # 用于下载预训练权重的包
    "tiktoken",         # 用于实现分词器的包
    "torch",            # 用于实现模型的PyTorch包
]
for p in pkgs:  # 遍历包列表并打印每个包的版本
    print(f"{p} version: {version(p)}")

import os  # 导入操作系统接口模块
import sys  # 导入系统相关的参数和函数模块
import io  # 导入输入输出接口模块
import nbformat  # 导入Jupyter笔记本格式模块
import types  # 导入Python内建类型模块

# 定义从Jupyter笔记本导入函数和类的函数
def import_from_notebook():
    def import_definitions_from_notebook(fullname, names):
        current_dir = os.getcwd()  # 获取当前工作目录
        path = os.path.join(current_dir, fullname + ".py")  # 构造笔记本文件路径
        path = os.path.normpath(path)  # 规范化路径

        # 加载笔记本文件
        if not os.path.exists(path):
            raise FileNotFoundError(f"Notebook file not found at: {path}")  # 如果文件不存在，抛出文件未找到异常

        with io.open(path, "r", encoding="utf-8") as f:  # 以只读模式打开文件
            nb = nbformat.read(f, as_version=4)  # 读取笔记本内容

        # 创建一个模块来存储导入的函数和类
        mod = types.ModuleType(fullname)  # 创建模块
        sys.modules[fullname] = mod  # 将模块添加到sys.modules中

        # 遍历笔记本单元格，只执行函数或类定义
        for cell in nb.cells:
            if cell.cell_type == "code":  # 如果单元格类型是代码
                cell_code = cell.source  # 获取单元格代码
                for name in names:  # 遍历需要导入的名称列表
                    # 检查单元格代码中是否有函数或类定义
                    if f"def {name}" in cell_code or f"class {name}" in cell_code:
                        exec(cell_code, mod.__dict__)  # 执行代码，将结果添加到模块中
        return mod  # 返回创建的模块

    fullname = "converting-gpt-to-llama2"  # 笔记本的完整名称
    names = ["precompute_rope_params", "compute_rope", "SiLU", "FeedForward", "RMSNorm", "MultiHeadAttention"]  # 需要导入的函数和类名称列表

    return import_definitions_from_notebook(fullname, names)  # 调用函数从笔记本导入定义

imported_module = import_from_notebook()  # 从笔记本导入模块

# 从导入的模块中获取特定的函数和类
compute_rope = getattr(imported_module, "compute_rope", None)  # 获取compute_rope函数
SiLU = getattr(imported_module, "SiLU", None)  # 获取SiLU类
FeedForward = getattr(imported_module, "FeedForward", None)  # 获取FeedForward类
RMSNorm = getattr(imported_module, "RMSNorm", None)  # 获取RMSNorm类
MultiHeadAttention = getattr(imported_module, "MultiHeadAttention", None)  # 获取MultiHeadAttention类，仅用于比较目的

import torch  # 导入PyTorch库

# 定义预计算RoPE参数的函数
def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None):
    assert head_dim % 2 == 0, "Embedding dimension must be even"  # 确保嵌入维度是偶数

    # 计算逆频率
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # 如果提供了频率配置，则进行频率调整
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama
    # 生成位置索引
    positions = torch.arange(context_length)

    # 计算角度
    angles = positions[:, None] * inv_freq[None, :]  # 形状：(context_length, head_dim // 2)

    # 扩展角度以匹配head_dim
    angles = torch.cat([angles, angles], dim=1)  # 形状：(context_length, head_dim)

    # 预计算正弦和余弦
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin  # 返回预计算的余弦和正弦值

# 实例化RoPE参数
llama_2_context_len = 4096  # Llama 2的上下文长度
llama_3_context_len = 8192  # Llama 3的上下文长度

llama_2_theta_base = 10_000  # Llama 2的theta_base值
llama_3_theta_base = 500_000  # Llama 3的theta_base值

# 设置
batch_size = 2  # 批次大小
num_heads = 4  # 头的数量
head_dim = 16  # 每个头的维度

# 实例化RoPE参数
cos, sin = precompute_rope_params(
    head_dim=head_dim,
    theta_base=llama_3_theta_base,
    context_length=llama_3_context_len
)

# 创建虚拟的查询和键张量
torch.manual_seed(123)  # 设置随机种子
queries = torch.randn(batch_size, num_heads, llama_3_context_len, head_dim)  # 查询张量
keys = torch.randn(batch_size, num_heads, llama_3_context_len, head_dim)  # 键张量

# 应用旋转位置嵌入
queries_rot = compute_rope(queries, cos, sin)  # 应用RoPE到查询张量
keys_rot = compute_rope(keys, cos, sin)  # 应用RoPE到键张量
import torch.nn as nn
############################# NEW  #############################
class SharedBuffers:
    _buffers = {}  # 类变量，用于存储预计算的缓冲区

    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        # 创建一个键，用于标识唯一的缓冲区
        key = (context_length, head_dim, rope_base, tuple(freq_config.values()) if freq_config else freq_config, dtype)

        if key not in SharedBuffers._buffers:
            # 如果缓冲区不存在，则创建或获取缓冲区
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)  # 创建上三角掩码
            cos, sin = precompute_rope_params(head_dim, rope_base, context_length, freq_config)  # 预计算RoPE参数
            if dtype is not None:
                cos = cos.to(dtype)  # 转换数据类型
                sin = sin.to(dtype)  # 转换数据类型
            SharedBuffers._buffers[key] = (mask, cos, sin)  # 存储缓冲区

        return SharedBuffers._buffers[key]  # 返回缓冲区


############################# NEW  #############################


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads,
                 num_kv_groups,  # NEW
                 rope_base=10_000,  # NEW
                 rope_config=None,  # NEW
                 dtype=None
    ):
        super().__init__()  # 调用父类构造函数
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"  # 确保d_out能被num_heads整除
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"  # 确保num_heads能被num_kv_groups整除

        self.d_out = d_out  # 输出维度
        self.num_heads = num_heads  # 头的数量
        self.head_dim = d_out // num_heads  # 每个头的维度

        ############################# NEW  #############################
        # self.W_key = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        # self.W_value = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)  # 线性层，用于计算键
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)  # 线性层，用于计算值
        self.num_kv_groups = num_kv_groups  # 键值组的数量
        self.group_size = num_heads // num_kv_groups  # 每组的大小
        ################################################################

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)  # 线性层，用于计算查询
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)  # 输出投影层

        ############################# NEW  #############################
        # 使用SharedBuffers获取缓冲区
        mask, cos, sin = SharedBuffers.get_buffers(context_length, self.head_dim, rope_base, rope_config, dtype)
        ############################# NEW  #############################

        self.register_buffer("mask", mask)  # 注册缓冲区
        self.register_buffer("cos", cos)  # 注册缓冲区
        self.register_buffer("sin", sin)  # 注册缓冲区

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # 获取输入的形状

        queries = self.W_query(x)  # 计算查询，形状：(b, num_tokens, d_out)
        keys = self.W_key(x)  # 计算键，形状：(b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)  # 计算值，形状：(b, num_tokens, num_kv_groups * head_dim)

        # Reshape queries, keys, and values
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)  # 调整查询的形状

        ##################### NEW  #####################
        # keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        # values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)  # 调整键的形状
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)  # 调整值的形状
        ################################################

        # Transpose keys, values, and queries
        keys = keys.transpose(1, 2)  # 转置键，形状：(b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)  # 转置值，形状：(b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  # 转置查询，形状：(b, num_query_groups, num_tokens, head_dim)

        # Apply RoPE
        keys = compute_rope(keys, self.cos, self.sin)  # 应用RoPE到键
        queries = compute_rope(queries, self.cos, self.sin)  # 应用RoPE到查询

        ##################### NEW  #####################
        # 扩展键和值以匹配头的数量
        # 形状：(b, num_heads, num_tokens, head_dim)

        keys = keys.repeat_interleave(self.group_size, dim=1)  # 重复键，形状：(b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)  # 重复值，形状：(b, num_heads, num_tokens, head_dim)
        # 例如，在repeat_interleave之前沿dim=1（查询组）：
        #   [K1, K2]
        # repeat_interleave之后（每个查询组重复group_size次）：
        #   [K1, K1, K2, K2]
        # 如果我们使用常规repeat而不是repeat_interleave，我们会得到：
        #   [K1, K2, K1, K2]
        ################################################

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # 形状：(b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  # 每个头的点积

        # 原始掩码截断到标记数量并转换为布尔值
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码填充注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)  # 应用softmax
        assert keys.shape[-1] == self.head_dim  # 确保键的最后一个维度等于head_dim

        # 形状：(b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)  # 计算上下文向量

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)  # 调整形状
        context_vec = self.out_proj(context_vec)  # 可选投影

        return context_vec  # 返回上下文向量
# 设置
batch_size = 1  # 批次大小
context_len = 3000  # 上下文长度
max_context_len = 8192  # 最大上下文长度
embed_dim = 4096  # 嵌入维度
num_heads = 32  # 头的数量

example_batch = torch.randn((batch_size, context_len, embed_dim))  # 创建一个随机的示例批次数据

# 初始化多头注意力模块
mha = MultiHeadAttention(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=max_context_len,
    num_heads=num_heads
)

mha(example_batch)  # 将示例批次数据传递给多头注意力模块

# 打印权重的形状
print("W_key:", mha.W_key.weight.shape)
print("W_value:", mha.W_value.weight.shape)
print("W_query:", mha.W_query.weight.shape)

# 初始化分组查询注意力模块
gqa = GroupedQueryAttention(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=max_context_len,
    num_heads=num_heads,
    num_kv_groups=8,
    rope_base=llama_3_theta_base
)

gqa(example_batch)  # 将示例批次数据传递给分组查询注意力模块

# 打印权重的形状
print("W_key:", gqa.W_key.weight.shape)
print("W_value:", gqa.W_value.weight.shape)
print("W_query:", gqa.W_query.weight.shape)

print("Total number of parameters:")  # 打印总参数数量

mha_total_params = sum(p.numel() for p in mha.parameters())  # 计算多头注意力模块的参数总数
print(f"MHA: {mha_total_params:,}")  # 打印多头注意力模块的参数总数

gqa_total_params = sum(p.numel() for p in gqa.parameters())  # 计算分组查询注意力模块的参数总数
print(f"GQA: {gqa_total_params:,}")  # 打印分组查询注意力模块的参数总数

# 释放内存
del mha
del gqa

# 定义Transformer块模块
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()  # 调用父类构造函数
        self.att = GroupedQueryAttention(  # 使用分组查询注意力模块
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],  # 键值组的数量
            rope_base=cfg["rope_base"],  # RoPE的theta基础值
            rope_config=cfg["rope_freq"],  # RoPE频率配置
            dtype=cfg["dtype"]  # 数据类型
        )
        self.ff = FeedForward(cfg)  # 前馈网络
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-5)  # 层归一化1
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-5)  # 层归一化2

    def forward(self, x):
        # 注意力块的快捷连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x.to(torch.bfloat16))  # 传递输入到注意力模块
        x = x + shortcut  # 添加原始输入

        # 前馈网络块的快捷连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x.to(torch.bfloat16))  # 传递输入到前馈网络
        x = x + shortcut  # 添加原始输入

        return x

# 定义Llama3模型
class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()  # 调用父类构造函数
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])  # 词嵌入层

        self.trf_blocks = nn.Sequential(  # Transformer块序列
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-5)  # 最终层归一化
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])  # 输出层

    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)  # 计算词嵌入
        x = tok_embeds
        x = self.trf_blocks(x)  # 传递输入到Transformer块序列
        x = self.final_norm(x)  # 应用最终层归一化
        logits = self.out_head(x.to(torch.bfloat16))  # 传递输入到输出层
        return logits

# Llama2模型配置
LLAMA2_CONFIG_7B = {
    "vocab_size": 32_000,    # 词汇表大小
    "context_length": 4096,  # 上下文长度
    "emb_dim": 4096,         # 嵌入维度
    "n_heads": 32,           # 注意力头的数量
    "n_layers": 32,          # 层的数量
    "hidden_dim": 11_008,    # 前馈网络的中间维度大小
    "dtype": torch.bfloat16  # 数据类型，用于减少内存使用
}

# Llama3模型配置
LLAMA3_CONFIG_8B = {
    "vocab_size": 128_256,   # 更大的词汇表大小
    "context_length": 8192,  # 更大的上下文长度
    "emb_dim": 4096,         # 嵌入维度
    "n_heads": 32,           # 注意力头的数量
    "n_layers": 32,          # 层的数量
    "hidden_dim": 14_336,    # 更大的前馈网络的中间维度大小
    "n_kv_groups": 8,        # 分组查询注意力的键值组数量
    "rope_base": 500_000.0,  # RoPE的theta基础值增加到500_000
    "rope_freq": None,       # RoPE频率的额外配置
    "dtype": torch.bfloat16  # 数据类型，用于减少内存使用
}
# 创建Llama3模型实例，使用LLAMA3_CONFIG_8B配置
model = Llama3Model(LLAMA3_CONFIG_8B)

# 检查缓冲区是否共享
# 检查所有Transformer块是否共享相同的掩码、cos和sin缓冲区
# 这有助于减少内存占用，因为相同的数据不需要重复存储
print(model.trf_blocks[0].att.mask is model.trf_blocks[-1].att.mask)  # 检查第一个和最后一个块的掩码是否相同
print(model.trf_blocks[0].att.cos is model.trf_blocks[-1].att.cos)  # 检查第一个和最后一个块的cos是否相同
print(model.trf_blocks[0].att.sin is model.trf_blocks[-1].att.sin)  # 检查第一个和最后一个块的sin是否相同

# 计算模型参数总数
total_params = sum(p.numel() for p in model.parameters())  # 遍历模型的所有参数，计算总数
print(f"Total number of parameters: {total_params:,}")  # 打印模型参数总数

# 定义计算模型内存大小的函数
def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0  # 参数元素总数
    total_grads = 0  # 梯度元素总数
    for param in model.parameters():  # 遍历模型的所有参数
        # 计算每个参数的元素总数
        param_size = param.numel()
        total_params += param_size
        # 检查参数是否需要梯度
        if param.requires_grad:
            total_grads += param_size

    # 计算缓冲区大小（非参数但需要内存）
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # 每个元素的大小（以字节为单位）=（元素数量）*（每个元素的大小）
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # 将字节转换为千兆字节
    total_memory_gb = total_memory_bytes / (1024**3)

    return total_memory_gb

# 打印不同数据类型下的模型内存大小
print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")  # 以float32类型计算内存大小
print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")  # 以bfloat16类型计算内存大小

# 检查是否有可用的CUDA设备，否则使用MPS或CPU
if torch.cuda.is_available():  # 检查CUDA是否可用
    device = torch.device("cuda")
elif torch.backends.mps.is_available():  # 检查MPS是否可用
    device = torch.device("mps")
else:  # 如果两者都不可用，使用CPU
    device = torch.device("cpu")

model.to(device);  # 将模型部署到指定设备

# 导入必要的库
import os  # 操作系统接口
from pathlib import Path  # 路径操作

import tiktoken  # Tiktoken库
from tiktoken.load import load_tiktoken_bpe  # 从Tiktoken库中导入BPE加载函数

# 定义Tokenizer类
class Tokenizer:
    def __init__(self, model_path):
        assert os.path.isfile(model_path), f"Model file {model_path} not found"  # 确保模型文件存在
        mergeable_ranks = load_tiktoken_bpe(model_path)  # 加载Tiktoken BPE模型

        self.special_tokens = {
            "<|begin_of_text|>": 128000,  # 特殊标记：文本开始
            "<|end_of_text|>": 128001,  # 特殊标记：文本结束
            "<|start_header_id|>": 128006,  # 特殊标记：标题开始
            "<|end_header_id|>": 128007,  # 特殊标记：标题结束
            "<|eot_id|>": 128009,  # 特殊标记：结束
        }
        self.special_tokens.update({  # 添加额外的特殊标记
            f"<|reserved_{i}|>": 128002 + i for i in range(256) if (128002 + i) not in self.special_tokens.values()
        })

        self.model = tiktoken.Encoding(  # 创建Tiktoken编码模型
            name=Path(model_path).name,  # 模型名称
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",  # 模式字符串
            mergeable_ranks=mergeable_ranks,  # 可合并等级
            special_tokens=self.special_tokens  # 特殊标记
        )

    def encode(self, text, bos=False, eos=False, allowed_special=set(), disallowed_special=()):
        if bos:  # 如果是开始标记
            tokens = [self.special_tokens["<|begin_of_text|>"]]  # 添加开始标记
        else:
            tokens = []

        tokens += self.model.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)  # 编码文本

        if eos:  # 如果是结束标记
            tokens.append(self.special_tokens["<|end_of_text|>"])  # 添加结束标记
        return tokens

    def decode(self, tokens):
        return self.model.decode(tokens)  # 解码标记

# 从Hugging Face Hub下载分词器文件
from huggingface_hub import login  # Hugging Face Hub登录模块
import json  # JSON处理模块

with open("config.json", "r") as config_file:  # 打开配置文件
    config = json.load(config_file)  # 加载配置
    access_token = config["HF_ACCESS_TOKEN"]  # 获取访问令牌

login(token=access_token)  # 使用访问令牌登录Hugging Face Hub

from huggingface_hub import hf_hub_download  # Hugging Face Hub下载模块

tokenizer_file_path = hf_hub_download(  # 下载分词器文件
    repo_id="meta-llama/Meta-Llama-3-8B",  # 仓库ID
    filename="original/tokenizer.model",  # 文件名
    local_dir="Llama-3-8B"  # 本地目录
)

# 创建Tokenizer实例
tokenizer = Tokenizer(tokenizer_file_path)

# 导入之前章节中的文本生成相关函数
from previous_chapters import generate, text_to_token_ids, token_ids_to_text

torch.manual_seed(123)  # 设置随机种子，确保结果可重复

# 生成文本
token_ids = generate(  # 调用生成函数
    model=model,  # 模型
    idx=text_to_token_ids("Every effort", tokenizer).to(device),  # 将文本转换为标记ID并传递给模型
    max_new_tokens=30,  # 最大新标记数
    context_size=LLAMA3_CONFIG_8B["context_length"],  # 上下文长度
    top_k=1,  # top-k采样
    temperature=0.  # 温度参数
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))  # 打印生成的文本
from safetensors.torch import load_file  # 导入 safetensors 库以加载 safetensors 格式的权重文件

combined_weights = {}  # 创建一个空字典来存储合并后的所有权重

# 迭代加载多个权重文件
for i in range(1, 5):
    # 从 Hugging Face 下载指定的 safetensors 权重文件
    weights_file = hf_hub_download(
        repo_id="meta-llama/Meta-Llama-3-8B",  # 模型所在的 Hugging Face 仓库 ID
        filename=f"model-0000{i}-of-00004.safetensors",  # 权重文件的名称，按编号顺序
        local_dir="Llama-3-8B"  # 下载到本地目录
    )
    # 使用 safetensors 加载当前的权重文件
    current_weights = load_file(weights_file)
    # 将当前加载的权重更新到 combined_weights 字典中
    combined_weights.update(current_weights)

# 查看加载的权重中的前 15 个键，了解权重结构
list(combined_weights.keys())[:15]

def assign(left, right, tensor_name="unknown"):
    # 检查左侧和右侧的形状是否匹配，如果不匹配则抛出异常
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

    # 如果右侧是一个张量类型，则将其克隆并转换为 torch.nn.Parameter
    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())
    else:
        # 如果右侧不是张量，则将其转换为张量并作为参数返回
        return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_llama(model, param_config, params):
    # 加载词嵌入层的权重，将加载的权重与模型中的权重进行匹配
    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    # 循环加载每一层的权重
    for l in range(param_config["n_layers"]):
        # 加载当前层的自注意力机制中的 Q、K、V 和输出投影的权重
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        model.trf_blocks[l].att.W_key.weight = assign(
            model.trf_blocks[l].att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        model.trf_blocks[l].att.W_value.weight = assign(
            model.trf_blocks[l].att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )
        model.trf_blocks[l].att.out_proj.weight = assign(
            model.trf_blocks[l].att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # 加载前馈神经网络层的权重（包括门控、上升和下降投影的权重）
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
        model.trf_blocks[l].norm2.weight = assign(
            model.trf_blocks[l].norm2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # 加载最终层的归一化权重
    model.final_norm.weight = assign(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight")

    # 检查是否存在 "lm_head.weight"，如果有，则加载，否则使用词嵌入权重
    if "lm_head.weight" in params.keys():
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
        print("Model uses weight tying.")  # 如果未找到 "lm_head.weight"，则使用词嵌入权重并打印提示

# 将合并的权重加载到 Llama 模型中
load_weights_into_llama(model, LLAMA3_CONFIG_8B, combined_weights)

# 将模型移到指定设备（如 GPU 或 CPU）
model.to(device)

# 删除临时存储的权重，以释放内存
del combined_weights  # free up memory
torch.manual_seed(123)  # 设置随机种子以确保结果可复现

# 调用生成函数生成文本
token_ids = generate(
    model=model,  # 使用指定的模型生成文本
    idx=text_to_token_ids("Every effort", tokenizer).to(device),  # 将输入文本转为token ID
    max_new_tokens=25,  # 最大生成 token 数量为 25
    context_size=LLAMA3_CONFIG_8B["context_length"],  # 上下文长度
    top_k=1,  # top-k 样本采样
    temperature=0.  # 温度控制生成的随机性，0 表示完全确定
)

# 输出生成的文本
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# 释放内存

import gc  # 导入垃圾回收模块

del model  # 删除模型对象，释放占用的内存

gc.collect()  # 手动运行 Python 垃圾回收器，清理无用对象

if torch.cuda.is_available():  # 如果 CUDA 可用（即有 GPU）
    torch.cuda.empty_cache()  # 清空 GPU 的缓存

# 合并所有权重
combined_weights = {}

# 迭代加载多个 safetensors 权重文件
for i in range(1, 5):
    # 从 Hugging Face 下载权重文件并加载
    weights_file = hf_hub_download(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",  # 模型所在的 Hugging Face 仓库 ID
        filename=f"model-0000{i}-of-00004.safetensors",  # 权重文件名，按顺序
        local_dir="Llama-3-8B-Instruct"  # 下载到本地的目录
    )
    # 加载当前权重文件
    current_weights = load_file(weights_file)
    # 更新合并的权重字典
    combined_weights.update(current_weights)

# 初始化新的 Llama 模型
model = Llama3Model(LLAMA3_CONFIG_8B)
# 将加载的权重应用到模型中
load_weights_into_llama(model, LLAMA3_CONFIG_8B, combined_weights)
# 将模型移到指定的计算设备
model.to(device)
# 删除合并的权重数据，释放内存
del combined_weights

# 定义一个 ChatFormat 类，用于处理聊天格式的输入和输出
class ChatFormat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer  # 存储 tokenizer 用于文本编码

    def encode_header(self, message):
        # 编码消息的头部（包含角色信息）
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])  # 添加开始头标记
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))  # 编码角色
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])  # 添加结束头标记
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))  # 添加换行符
        return tokens  # 返回编码后的 tokens

    def encode(self, text):
        # 编码用户输入的文本
        message = {
            "role": "user",  # 消息角色为用户
            "content": text  # 消息内容
        }

        tokens = self.encode_header(message)  # 编码消息头部
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)  # 编码内容并去除首尾空格
        )
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])  # 添加结束标记
        return tokens  # 返回编码后的 tokens

    def decode(self, token_ids):
        # 将 token_ids 解码为文本
        return self.tokenizer.decode(token_ids)

# 初始化一个聊天格式的 tokenizer
chat_tokenizer = ChatFormat(tokenizer)

# 编码 "Hello World!" 文本
token_ids = chat_tokenizer.encode("Hello World!")
print(token_ids)  # 输出编码后的 token IDs

# 解码编码后的 token IDs
tokenizer.decode(token_ids)

# 设置随机种子以确保生成结果可复现
torch.manual_seed(123)

# 调用生成函数生成文本
token_ids = generate(
    model=model,  # 使用指定的模型
    idx=text_to_token_ids("What do llamas eat?", chat_tokenizer).to(device),  # 将输入文本转为 token ID
    max_new_tokens=150,  # 最大生成 token 数量为 150
    context_size=LLAMA3_CONFIG_8B["context_length"],  # 上下文长度
    top_k=1,  # top-k 样本采样
    temperature=0.  # 温度控制生成的随机性
)

# 将生成的 token IDs 转换为文本
output_text = token_ids_to_text(token_ids, tokenizer)

# 定义清理文本的函数，用于去除头部信息
def clean_text(text, header_end="assistant<|end_header_id|>\n\n"):
    # 查找头部结束标记 "<|end_header_id|>"
    index = text.find(header_end)

    if index != -1:
        # 如果找到标记，则返回从该位置之后的文本并去除前后的空白
        return text[index + len(header_end):].strip()
    else:
        # 如果未找到标记，则返回原始文本
        return text

# 输出清理后的文本
print("Output text:\n", clean_text(output_text))

# 配置 LLAMA3 模型的配置参数
LLAMA3_CONFIG_8B = {
    "vocab_size": 128_256,   # 词汇表大小
    "context_length": 8192,  # 上下文长度
    "emb_dim": 4096,         # 嵌入维度
    "n_heads": 32,           # 注意力头数
    "n_layers": 32,          # 层数
    "hidden_dim": 14_336,    # 前馈网络中的隐藏层维度
    "n_kv_groups": 8,        # 分组的键值对数量
    "rope_base": 500_000.0,  # RoPE (Rotary Positional Encoding) 中的基础值
    "rope_freq": None,       # RoPE 频率的额外配置
    "dtype": torch.bfloat16  # 使用较低精度的数据类型以减少内存使用
}

# 配置 LLAMA3 模型的扩展配置参数
LLAMA31_CONFIG_8B = {
    "vocab_size": 128_256,      # 词汇表大小
    "context_length": 131_072,  # 新的更大的支持上下文长度
    "emb_dim": 4096,            # 嵌入维度
    "n_heads": 32,              # 注意力头数
    "n_layers": 32,             # 层数
    "hidden_dim": 14_336,       # 前馈网络中的隐藏层维度
    "n_kv_groups": 8,           # 分组的键值对数量
    "rope_base": 500_000.0,     # RoPE 中的基础值
    "dtype": torch.bfloat16,    # 使用较低精度的数据类型以减少内存使用
    "rope_freq": {              # RoPE 频率缩放的新配置
        "factor": 8.0,          # RoPE 缩放因子
        "low_freq_factor": 1.0, # 低频因子
        "high_freq_factor": 4.0,  # 高频因子
        "original_context_length": 8192,  # 原始的上下文长度
    }
}

# 设置上下文长度为 8192，覆盖之前的较大值
old_context_length = LLAMA31_CONFIG_8B["context_length"]
LLAMA31_CONFIG_8B["context_length"] = 8192

# 定义一个函数来重新调整theta参数以适应新的上下文长度
def rescale_theta(theta_old, context_length_old, context_length_new):
    scaling_factor = context_length_new / context_length_old  # 计算缩放因子
    theta_new = theta_old * scaling_factor  # 缩放theta值
    return theta_new  # 返回新的theta值

# 调整LLAMA31_CONFIG_8B配置中的RoPE theta参数
LLAMA31_CONFIG_8B["rope_base"] = rescale_theta(
    LLAMA31_CONFIG_8B["rope_base"],
    old_context_length,
    LLAMA31_CONFIG_8B["context_length"]
)

print("New RoPE theta:", LLAMA31_CONFIG_8B["rope_base"])  # 打印新的RoPE theta值

# 释放内存
del model  # 删除模型实例以释放内存
gc.collect()  # 运行Python垃圾回收器
if torch.cuda.is_available():  # 如果CUDA可用
    torch.cuda.empty_cache()  # 清空CUDA缓存

# 从Hugging Face Hub下载分词器文件
tokenizer_file_path = hf_hub_download(
    repo_id="meta-llama/Llama-3.1-8B",
    filename="original/tokenizer.model",
    local_dir="Llama-3.1-8B"
)

tokenizer = Tokenizer(tokenizer_file_path)  # 创建Tokenizer实例

model = Llama3Model(LLAMA31_CONFIG_8B)  # 创建Llama3模型实例

# 计算模型参数总数
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")  # 打印模型参数总数

# 下载并合并权重文件
combined_weights = {}
for i in range(1, 5):
    weights_file = hf_hub_download(
        repo_id="meta-llama/Llama-3.1-8B",
        filename=f"model-0000{i}-of-00004.safetensors",
        local_dir="Llama-3.1-8B"
    )
    current_weights = load_file(weights_file)  # 加载权重文件
    combined_weights.update(current_weights)  # 合并权重

load_weights_into_llama(model, LLAMA31_CONFIG_8B, combined_weights)  # 将权重加载到模型
model.to(device);  # 将模型部署到指定设备
del combined_weights  # 释放内存

torch.manual_seed(123)  # 设置随机种子

# 生成文本
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort", tokenizer).to(device),
    max_new_tokens=25,
    context_size=LLAMA31_CONFIG_8B["context_length"],
    top_k=1,
    temperature=0.
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))  # 打印生成的文本

# LLAMA31_CONFIG_8B配置
LLAMA31_CONFIG_8B = {
    "vocab_size": 128_256,      # 词汇表大小
    "context_length": 131_072,  # 支持的上下文长度
    "emb_dim": 4096,            # 嵌入维度
    "n_heads": 32,              # 注意力头的数量
    "n_layers": 32,             # 层的数量
    "hidden_dim": 14_336,       # 前馈网络的中间维度大小
    "n_kv_groups": 8,           # 分组查询注意力的键值组数量
    "rope_base": 500_000.0,     # RoPE的theta基础值
    "dtype": torch.bfloat16,    # 数据类型，用于减少内存使用
    "rope_freq": {              # RoPE频率缩放配置
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}

# LLAMA32_CONFIG_1B配置
LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,      # 词汇表大小
    "context_length": 131_072,  # 上下文长度
    "emb_dim": 2048,            # 嵌入维度减半
    "n_heads": 32,              # 注意力头的数量
    "n_layers": 16,             # 层的数量减半
    "hidden_dim": 8192,         # 前馈网络的中间维度大小减半
    "n_kv_groups": 8,           # 分组查询注意力的键值组数量
    "rope_base": 500_000.0,     # RoPE的theta基础值
    "dtype": torch.bfloat16,    # 数据类型，用于减少内存使用
    "rope_freq": {              # RoPE频率缩放配置
        "factor": 32.0,         # 重新缩放因子的调整
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}

old_context_length = LLAMA32_CONFIG_1B["context_length"]  # 旧的上下文长度
LLAMA32_CONFIG_1B["context_length"] = 8192  # 更新上下文长度为8192

# 调整RoPE theta参数
LLAMA32_CONFIG_1B["rope_base"] = rescale_theta(
    LLAMA32_CONFIG_1B["rope_base"],
    old_context_length,
    LLAMA32_CONFIG_1B["context_length"]
)

print("New RoPE theta:", LLAMA32_CONFIG_1B["rope_base"])  # 打印新的RoPE theta值

# 释放内存
del model
gc.collect()  # 运行Python垃圾回收器
if torch.cuda.is_available():  # 如果CUDA可用
    torch.cuda.empty_cache()  # 清空CUDA缓存

# 从Hugging Face Hub下载分词器文件
tokenizer_file_path = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-1B",
    filename="original/tokenizer.model",
    local_dir="Llama-3.2-1B"
)

tokenizer = Tokenizer(tokenizer_file_path)  # 创建Tokenizer实例

model = Llama3Model(LLAMA32_CONFIG_1B)  # 创建Llama3模型实例

# 计算模型参数总数
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")  # 打印模型参数总数

# 考虑权重绑定
total_params_normalized = total_params - model.tok_emb.weight.numel()  # 减去词嵌入层的参数
print(f"\nTotal number of unique parameters: {total_params_normalized:,}")  # 打印独特参数总数

# 下载权重文件
weights_file = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-1B",
    filename=f"model.safetensors",
    local_dir="Llama-3.2-1B"
)
current_weights = load_file(weights_file)  # 加载权重文件

load_weights_into_llama(model, LLAMA32_CONFIG_1B, current_weights)  # 将权重加载到模型
model.to(device);  # 将模型部署到指定设备
del current_weights  # 释放内存

print("Weight tying:", torch.equal(model.tok_emb.weight, model.out_head.weight))  # 检查权重绑定

torch.manual_seed(123)  # 设置随机种子

# 生成文本
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort", tokenizer).to(device),
    max_new_tokens=25,
    context_size=LLAMA32_CONFIG_1B["context_length"],
    top_k=1,
    temperature=0.
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))  # 打印生成的文本