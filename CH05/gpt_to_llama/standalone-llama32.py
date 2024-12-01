from importlib.metadata import version  # 从 importlib.metadata 导入 version 函数，用于获取安装的库版本

# 定义需要检查版本的包列表
pkgs = [
    "blobfile",         # 用于下载预训练权重
    "huggingface_hub",  # 用于下载预训练权重
    "tiktoken",         # 用于实现分词器
    "torch",            # 用于实现模型
]

# 遍历包列表并打印每个包的版本
for p in pkgs:
    print(f"{p} version: {version(p)}")  # 打印每个包的版本信息

# 导入 PyTorch 库
import torch
import torch.nn as nn  # 导入 PyTorch 的神经网络模块


# 定义一个前馈神经网络类 FeedForward
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()  # 调用父类的构造函数
        # 定义三个线性层（全连接层），不使用偏置项
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        # 前向传播，依次通过每个线性层
        x_fc1 = self.fc1(x)  # 通过第一个全连接层
        x_fc2 = self.fc2(x)  # 通过第二个全连接层
        x = nn.functional.silu(x_fc1) * x_fc2  # 对第一个全连接层的输出应用 SiLU 激活函数并与第二个输出相乘
        return self.fc3(x)  # 通过第三个全连接层并返回输出


# 预计算 RoPE 参数的函数
def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None):
    assert head_dim % 2 == 0, "Embedding dimension must be even"  # 确保 head_dim 为偶数

    # 计算反频率（inverse frequency）
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # 如果有频率调整配置，则进行调整
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq  # 计算波长

        # 调整频率
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

    # 生成位置索引（位置编码的索引）
    positions = torch.arange(context_length)

    # 计算角度（用于位置编码）
    angles = positions[:, None] * inv_freq[None, :]  # 形状为 (context_length, head_dim // 2)

    # 扩展角度以匹配 head_dim
    angles = torch.cat([angles, angles], dim=1)  # 形状为 (context_length, head_dim)

    # 预计算正弦和余弦值
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin  # 返回正弦和余弦值


# 计算 RoPE（旋转位置编码）变换的函数
def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape  # 获取输入 x 的维度
    assert head_dim % 2 == 0, "Head dimension must be even"  # 确保 head_dim 为偶数

    # 将 x 切分为前半部分和后半部分
    x1 = x[..., : head_dim // 2]  # 前半部分
    x2 = x[..., head_dim // 2 :]  # 后半部分

    # 调整 sin 和 cos 的形状，以便与 x 进行广播
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # 形状调整为 (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # 应用旋转变换
    rotated = torch.cat((-x2, x1), dim=-1)  # 将 x2 和 x1 进行交换
    x_rotated = (x * cos) + (rotated * sin)  # 计算旋转后的值

    return x_rotated.to(dtype=x.dtype)  # 返回旋转后的张量，并确保数据类型一致


# 定义一个共享缓冲区的类，用于缓存 RoPE 计算的中间结果
class SharedBuffers:
    _buffers = {}  # 类变量，用于存储缓存的结果

    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        # 使用配置生成一个唯一的键值，以便缓存结果
        key = (context_length, head_dim, rope_base, tuple(freq_config.values()) if freq_config else freq_config, dtype)

        # 如果该键值的缓冲区尚未存在，则生成并缓存它们
        if key not in SharedBuffers._buffers:
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)  # 创建上三角矩阵
            cos, sin = precompute_rope_params(head_dim, rope_base, context_length, freq_config)  # 计算 RoPE 参数
            if dtype is not None:
                cos = cos.to(dtype)  # 将结果转换为所需的数据类型
                sin = sin.to(dtype)
            SharedBuffers._buffers[key] = (mask, cos, sin)  # 将计算结果存入缓存

        return SharedBuffers._buffers[key]  # 返回缓存的结果


# 定义一个分组查询注意力类 GroupedQueryAttention
class GroupedQueryAttention(nn.Module):
    def __init__(
            self, d_in, d_out, context_length, num_heads,
            num_kv_groups,
            rope_base=10_000,
            rope_config=None,
            dtype=None
        ):
        super().__init__()  # 调用父类的构造函数
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"  # 确保 d_out 可以被 num_heads 整除
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"  # 确保 num_heads 可以被 num_kv_groups 整除

        self.d_out = d_out  # 输出维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = d_out // num_heads  # 每个注意力头的维度

        # 定义键、值和查询的线性变换层
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups  # 键值对分组数
        self.group_size = num_heads // num_kv_groups  # 每个分组的头数

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)  # 查询的线性层
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)  # 输出投影层

        # 从 SharedBuffers 获取缓存的 RoPE 参数
        mask, cos, sin = SharedBuffers.get_buffers(context_length, self.head_dim, rope_base, rope_config, dtype)
        self.register_buffer("mask", mask)  # 注册 mask 缓存

        self.register_buffer("cos", cos)  # 注册 cos 缓存
        self.register_buffer("sin", sin)  # 注册 sin 缓存

    class GroupedQueryAttention(nn.Module):
        def forward(self, x):
            b, num_tokens, d_in = x.shape  # b: 批大小，num_tokens: 令牌数量，d_in: 输入维度（嵌入维度）

            queries = self.W_query(x)  # 应用查询变换（形状: b, num_tokens, d_out）
            keys = self.W_key(x)  # 应用键变换（形状: b, num_tokens, num_kv_groups * head_dim）
            values = self.W_value(x)  # 应用值变换（形状: b, num_tokens, num_kv_groups * head_dim）

            # 重塑查询、键和值的形状，以适应头数和键值组
            queries = queries.view(b, num_tokens, self.num_heads,
                                   self.head_dim)  # 重塑为 (b, num_tokens, num_heads, head_dim)
            keys = keys.view(b, num_tokens, self.num_kv_groups,
                             self.head_dim)  # 重塑为 (b, num_tokens, num_kv_groups, head_dim)
            values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)  # 同样重塑

            # 转置键、值和查询，以便进行注意力计算
            keys = keys.transpose(1, 2)  # 形状: (b, num_heads, num_tokens, head_dim)
            values = values.transpose(1, 2)  # 形状: (b, num_heads, num_tokens, head_dim)
            queries = queries.transpose(1, 2)  # 形状: (b, num_query_groups, num_tokens, head_dim)

            # 对键和值应用旋转位置编码（RoPE）
            keys = compute_rope(keys, self.cos, self.sin)  # 对键应用RoPE
            queries = compute_rope(queries, self.cos, self.sin)  # 对查询应用RoPE

            # 扩展键和值的数量，以匹配头数，通过在组轴上重复
            keys = keys.repeat_interleave(self.group_size, dim=1)  # 形状: (b, num_heads, num_tokens, head_dim)
            values = values.repeat_interleave(self.group_size, dim=1)  # 形状: (b, num_heads, num_tokens, head_dim)
            # 解释：在 `dim=1` 上重复键和值，以匹配头的数量。
            # 例如：[K1, K2] 变为 [K1, K1, K2, K2]

            # 计算缩放点积注意力（自注意力），并应用因果掩码
            attn_scores = queries @ keys.transpose(2, 3)  # 查询和转置的键之间的点积

            # 使用掩码来屏蔽注意力分数（用于实现因果掩码，例如自回归模型）
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # 获取因果掩码（最多 num_tokens 个令牌）
            attn_scores.masked_fill_(mask_bool, -torch.inf)  # 将被掩码的位置填充为负无穷大

            # 通过应用 softmax 来计算注意力权重
            attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)  # 对分数进行归一化

            # 将注意力权重应用于值，以得到最终的上下文向量
            context_vec = (attn_weights @ values).transpose(1, 2)  # 形状: (b, num_tokens, num_heads, head_dim)

            # 将注意力头合并（self.d_out = self.num_heads * self.head_dim）
            context_vec = context_vec.reshape(b, num_tokens, self.d_out)  # 形状: (b, num_tokens, d_out)
            context_vec = self.out_proj(context_vec)  # 应用输出投影（可选）

            return context_vec  # 返回经过注意力和投影处理后的上下文向量
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(  # 定义注意力机制
            d_in=cfg["emb_dim"],  # 输入维度（嵌入维度）
            d_out=cfg["emb_dim"],  # 输出维度（与输入相同）
            context_length=cfg["context_length"],  # 上下文长度（模型可以处理的令牌数）
            num_heads=cfg["n_heads"],  # 注意力头的数量
            num_kv_groups=cfg["n_kv_groups"],  # 注意力中的键值组数量
            rope_base=cfg["rope_base"],  # RoPE（旋转位置编码）的基频
            rope_config=cfg["rope_freq"],  # RoPE频率配置（例如，缩放因子）
            dtype=cfg["dtype"]  # 数据类型（精度）
        )
        self.ff = FeedForward(cfg)  # 定义前馈神经网络模块
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)  # 对注意力输出应用RMS归一化
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)  # 对前馈输出应用RMS归一化

    def forward(self, x):
        shortcut = x  # 保存输入以进行残差连接（shortcut）
        x = self.norm1(x)  # 在注意力之前对输入进行归一化
        x = self.att(x.to(torch.bfloat16))  # 应用注意力（转换为bfloat16以节省内存）
        x = x + shortcut  # 将输入添加回输出（残差连接）

        # 前馈块，带有残差连接
        shortcut = x  # 保存输入以进行残差连接（shortcut）
        x = self.norm2(x)  # 在通过前馈网络之前进行归一化
        x = self.ff(x.to(torch.bfloat16))  # 应用前馈变换（转换为bfloat16）
        x = x + shortcut  # 将输入添加回输出（残差连接）

        return x  # 返回变换后的输出

class Llama3Model(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])  # 令牌嵌入层

            # 创建一系列的 transformer 块作为模型的主体
            self.trf_blocks = nn.Sequential(
                *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]  # 堆叠多个 transformer 层
            )

            self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)  # 对最终输出应用 RMS 归一化
            self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])  # 输出投影层

        def forward(self, in_idx):
            tok_embeds = self.tok_emb(in_idx)  # 将令牌索引转换为嵌入向量
            x = tok_embeds  # 将 x 初始化为令牌嵌入
            x = self.trf_blocks(x)  # 应用 transformer 块（注意力 + 前馈）
            x = self.final_norm(x)  # 对 transformer 块的输出进行归一化
            logits = self.out_head(x.to(torch.bfloat16))  # 应用最终的线性层来得到 logits
            return logits  # 返回 logits（每个令牌的原始得分）

# Llama 3.2 1B 模型配置
LLAMA32_CONFIG = {
    "vocab_size": 128_256,  # 词汇表大小
    "context_length": 131_072,  # 最大上下文长度（模型一次可以处理的令牌数）
        "emb_dim": 2048,  # 嵌入维度（每个令牌的表示大小）
        "n_heads": 32,  # 注意力头的数量
        "n_layers": 16,  # 模型中的 transformer 层数
        "hidden_dim": 8192,  # 前馈网络中的隐藏维度
        "n_kv_groups": 8,  # 注意力中的键值组数量
        "rope_base": 500_000.0,  # RoPE（旋转位置编码）的基频
        "dtype": torch.bfloat16,  # 数据类型（降低内存使用）
        "rope_freq": {  # RoPE 频率的缩放因子
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,  # RoPE的频率缩放基于的原始上下文长度
        }
    }

# 根据配置的嵌入维度决定LLAMA模型大小的字符串表示
LLAMA_SIZE_STR = "1B" if LLAMA32_CONFIG["emb_dim"] == 2048 else "3B"

# 记录原始的上下文长度
old_context_length = LLAMA32_CONFIG["context_length"]
# 设置新的上下文长度为8192
LLAMA32_CONFIG["context_length"] = 8192


# 重新缩放theta值的函数，基于新的和旧的上下文长度
def rescale_theta(theta_old, context_length_old, context_length_new):
    scaling_factor = context_length_new / context_length_old  # 计算缩放因子
    theta_new = theta_old * scaling_factor  # 应用缩放因子来得到新的theta值
    return theta_new

# 使用重新缩放的theta更新RoPE基频
LLAMA32_CONFIG["rope_base"] = rescale_theta(
    LLAMA32_CONFIG["rope_base"],  # 原始的RoPE基频
    old_context_length,  # 原始的上下文长度
    LLAMA32_CONFIG["context_length"]  # 新的上下文长度
)

# 输出新的RoPE基频值
print("New RoPE theta:", LLAMA32_CONFIG["rope_base"])

# 创建模型实例
model = Llama3Model(LLAMA32_CONFIG)

# 检查模型中的某些参数是否共享内存
print(model.trf_blocks[0].att.mask is model.trf_blocks[-1].att.mask)  # 检查第一个和最后一个块的mask是否相同
print(model.trf_blocks[0].att.cos is model.trf_blocks[-1].att.cos)  # 检查cos是否共享
print(model.trf_blocks[0].att.sin is model.trf_blocks[-1].att.sin)  # 检查sin是否共享


# 计算并打印模型的总参数数目
total_params = sum(p.numel() for p in model.parameters())  # 计算所有参数的总数
print(f"Total number of parameters: {total_params:,}")

# 计算模型中唯一参数的总数（减去共享的嵌入层权重）
total_params_normalized = total_params - model.tok_emb.weight.numel()  # 去除嵌入层的参数
print(f"\nTotal number of unique parameters: {total_params_normalized:,}")

# 计算模型的内存占用
def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0  # 总参数数目初始化
    total_grads = 0  # 总梯度数目初始化
    for param in model.parameters():
        param_size = param.numel()  # 获取每个参数的元素数目
        total_params += param_size  # 累加参数数目
        if param.requires_grad:  # 检查参数是否需要梯度
            total_grads += param_size  # 累加需要梯度的参数数目

    # 计算模型中所有缓冲区（非参数）的内存占用
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # 计算内存总占用（以字节为单位）
    element_size = torch.tensor(0, dtype=input_dtype).element_size()  # 每个元素的大小（字节）
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size  # 总字节数

    # 转换为GB单位
    total_memory_gb = total_memory_bytes / (1024**3)  # 将字节转换为GB

    return total_memory_gb

# 计算并打印模型的内存占用（以GB为单位）
print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")

# 判断是否有可用的GPU，MPS设备，或者使用CPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # 如果有CUDA设备，使用GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # 如果有MPS设备，使用MPS
else:
    device = torch.device("cpu")  # 否则使用CPU

# 将模型移动到选定的设备
model.to(device)

# 引入所需的库
import os
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe


# 定义Tokenizer类，用于文本和token的转换
class Tokenizer:
    def __init__(self, model_path):
        assert os.path.isfile(model_path), f"Model file {model_path} not found"  # 确保模型路径存在
        mergeable_ranks = load_tiktoken_bpe(model_path)  # 加载字节对编码（BPE）

        # 定义特殊令牌
        self.special_tokens = {
            "<|begin_of_text|>": 128000,  # 文本开始标记
            "<|end_of_text|>": 128001,  # 文本结束标记
            "<|start_header_id|>": 128006,  # 标题开始标记
            "<|end_header_id|>": 128007,  # 标题结束标记
            "<|eot_id|>": 128009,  # 结束标记
        }
        # 为特殊标记添加保留的令牌
        self.special_tokens.update({
            f"<|reserved_{i}|>": 128002 + i for i in range(256) if (128002 + i) not in self.special_tokens.values()
        })

        # 初始化tiktoken的编码器
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,  # 设置模型名称
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+",  # 正则表达式模式
            mergeable_ranks=mergeable_ranks,  # 加载的BPE合并规则
            special_tokens=self.special_tokens  # 特殊标记
        )


    def encode(self, text, bos=False, eos=False, allowed_special=set(), disallowed_special=()):
        # 编码文本为token序列
        if bos:  # 如果需要添加开始标记
            tokens = [self.special_tokens["<|begin_of_text|>"]]
        else:
            tokens = []

        # 使用编码器编码文本
        tokens += self.model.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)

        # 如果需要添加结束标记
        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])
        return tokens

    def decode(self, tokens):
        # 解码token序列为文本
        return self.model.decode(tokens)


# 定义聊天格式处理类，用于处理用户消息的编码与解码
class ChatFormat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer  # 初始化时传入tokenizer

    def encode_header(self, message):
        # 编码消息头部，包含角色信息
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])  # 开始头部标记
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))  # 编码角色信息
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])  # 结束头部标记
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))  # 编码换行符
        return tokens

    def encode(self, text):
        # 编码消息内容
        message = {
            "role": "user",  # 用户角色
            "content": text  # 消息内容
        }

        tokens = self.encode_header(message)  # 编码头部信息
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)  # 编码内容
        )
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])  # 添加结束标记
        return tokens

    def decode(self, token_ids):
        # 解码tokenid为文本
        return self.tokenizer.decode(token_ids)

from huggingface_hub import login  # 导入Hugging Face Hub的登录模块

login()  # 登录到Hugging Face Hub

from huggingface_hub import hf_hub_download  # 导入Hugging Face Hub的文件下载模块

# 从Hugging Face Hub下载分词器文件
tokenizer_file_path = hf_hub_download(
    repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",  # 指定仓库ID和分词器文件名
    filename="original/tokenizer.model",  # 分词器文件名
    local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"  # 本地存储目录
)

tokenizer = Tokenizer(tokenizer_file_path)  # 创建Tokenizer实例
chat_tokenizer = ChatFormat(tokenizer)  # 创建聊天格式的分词器实例

# 定义一个函数来分配权重，确保权重的形状匹配
def assign(left, right, tensor_name="unknown"):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())  # 如果right是Tensor，克隆并分离
    else:
        return torch.nn.Parameter(torch.tensor(right))  # 否则，将right转换为Tensor

# 定义一个函数来加载权重到Llama模型
def load_weights_into_llama(model, param_config, params):
    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
    # 遍历每一层，加载注意力和前馈网络的权重
    for l in range(param_config["n_layers"]):
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        # ...（省略中间的代码以节省空间，模式与上述相同）

    # 加载输出层权重
    model.final_norm.weight = assign(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight")
    # 如果存在lm_head.weight，则将其加载到输出头，否则使用weight tying
    if "lm_head.weight" in params.keys():
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
        print("Model uses weight tying.")

from safetensors.torch import load_file  # 导入safetensors库的文件加载模块

# 根据模型大小下载权重文件并加载权重
if LLAMA_SIZE_STR == "1B":
    weights_file = hf_hub_download(
        repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
        filename=f"model.safetensors",
        local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
    )
    combined_weights = load_file(weights_file)  # 加载单个权重文件
else:
    combined_weights = {}
    for i in range(1, 3):  # 对于更大的模型，下载并合并多个权重文件
        weights_file = hf_hub_download(
            repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
            filename=f"model-0000{i}-of-00002.safetensors",
            local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
        )
        current_weights = load_file(weights_file)  # 加载当前权重文件
        combined_weights.update(current_weights)  # 合并权重
# 将预训练的权重加载到Llama模型中，并将模型移动到设备上（GPU或CPU）
load_weights_into_llama(model, LLAMA32_CONFIG, combined_weights)
model.to(device)  # 将模型移动到选择的设备（例如GPU）
del combined_weights  # 删除combined_weights变量，释放内存

# 检查权重是否共享，即嵌入层和输出层的权重是否相同
print("Weight tying:", torch.equal(model.tok_emb.weight, model.out_head.weight))

# 将文本转换为token id的函数
def text_to_token_ids(text, tokenizer):
    # 使用tokenizer编码文本
    encoded = tokenizer.encode(text)
    # 将编码后的token转换为tensor，并添加batch维度
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # unsqueeze(0)是为了添加一个batch维度
    return encoded_tensor

# 将token id转换回文本的函数
def token_ids_to_text(token_ids, tokenizer):
    # 去掉batch维度，保留单一序列
    flat = token_ids.squeeze(0)  # squeeze(0)移除batch维度
    # 解码token id为文本
    return tokenizer.decode(flat.tolist())  # 将tensor转换为list再解码为文本

# 生成文本的函数，基于给定的模型和输入token进行生成
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # 通过for循环生成max_new_tokens个新token
    for _ in range(max_new_tokens):
        # 只保留最近的context_size个token（防止上下文超出模型的上下文窗口）
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():  # 生成时不需要计算梯度
            logits = model(idx_cond)  # 获取模型的logits
        logits = logits[:, -1, :]  # 获取最后一步的logits（即下一个token的概率分布）

        # 如果进行top-k采样，筛选出top_k个最大值的logits
        if top_k is not None:
            # 获取top_k的logits
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]  # 获取top_k中的最小值
            # 将logits中小于min_val的值置为负无穷（即不再考虑这些token）
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # 如果有temperature参数，应用温度缩放来调整logits
        if temperature > 0.0:
            logits = logits / temperature  # 缩放logits以调整采样的随机性

            # 应用softmax函数计算token的概率分布
            probs = torch.softmax(logits, dim=-1)  # 计算logits的softmax，得到概率

            # 从概率分布中采样一个token
            idx_next = torch.multinomial(probs, num_samples=1)  # 使用multinomial采样，返回下一个token的id

        # 如果没有temperature，直接选择logits最大值对应的token作为下一个token
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # 选择最大值对应的索引作为下一个token

        # 如果生成的token是eos_id（结束符），则提前结束生成
        if idx_next == eos_id:  # 如果生成了结束标记，提前停止生成
            break

        # 将新生成的token添加到已有的token序列中
        idx = torch.cat((idx, idx_next), dim=1)  # 将新生成的token拼接到现有序列后

    return idx  # 返回完整的token序列

# 设置输入提示文本
PROMPT = "What do llamas eat?"

# 设置随机种子，确保结果可重现
torch.manual_seed(123)

# 生成最大为150个新token的文本，使用提供的生成函数
token_ids = generate(
    model=model,
    idx=text_to_token_ids(PROMPT, chat_tokenizer).to(device),  # 将输入提示转化为token并传到设备
    max_new_tokens=150,  # 最大生成token数量
    context_size=LLAMA32_CONFIG["context_length"],  # 使用配置的上下文长度
    top_k=1,  # 使用top-k采样，仅保留概率最高的一个token
    temperature=0.  # 设置temperature为0，完全贪婪选择最优token
)

# 将生成的token ids转换回文本
output_text = token_ids_to_text(token_ids, tokenizer)

# 定义清理生成文本的函数，移除不需要的头部信息
def clean_text(text, header_end="assistant<|end_header_id|>\n\n"):
    # 查找文本中第一个出现的header_end标记的位置
    index = text.find(header_end)

    if index != -1:  # 如果找到标记
        # 返回标记之后的文本部分，去掉前后的空白
        return text[index + len(header_end):].strip()  # 返回清理后的文本
    else:
        # 如果没有找到标记，返回原始文本
        return text

# 输出清理后的文本
print("Output text:\n", clean_text(output_text))  # 打印清理后的输出文本





