from importlib.metadata import version  # 导入版本检查模块

pkgs = [  # 定义需要检查版本的包列表
    "huggingface_hub",  # 用于下载预训练权重的包
    "sentencepiece",    # 用于实现分词器的包
    "torch",            # 用于实现模型的PyTorch包
]
for p in pkgs:  # 遍历包列表并打印每个包的版本
    print(f"{p} version: {version(p)}")

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块

####################################
# Chapter 4
####################################

class RMSNorm(nn.Module):  # 定义RMSNorm类，继承自PyTorch的nn.Module
    def __init__(self, emb_dim, eps=1e-5):  # 初始化函数，接受嵌入维度和epsilon值
        super().__init__()  # 调用父类的初始化函数
        self.eps = eps  # epsilon值，用于数值稳定性
        self.emb_dim = emb_dim  # 嵌入维度
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()  # 可学习的权重参数，初始化为1

    def forward(self, x):  # 前向传播函数，接受输入x
        means = x.pow(2).mean(dim=-1, keepdim=True)  # 计算平方的均值
        x_normed = x * torch.rsqrt(means + self.eps)  # 归一化输入x
        return (x_normed * self.weight).to(dtype=x.dtype)  # 返回归一化后的x乘以权重，并转换为x的数据类型

torch.manual_seed(123)  # 设置随机种子以确保结果的可重复性

example_batch = torch.randn(2, 3, 4)  # 创建一个随机的示例批次数据

rms_norm = RMSNorm(emb_dim=example_batch.shape[-1])  # 创建RMSNorm实例
rmsnorm_pytorch = torch.nn.RMSNorm(example_batch.shape[-1], eps=1e-5)  # 创建PyTorch的RMSNorm实例

assert torch.allclose(rms_norm(example_batch), rmsnorm_pytorch(example_batch))  # 确保两个RMSNorm实现的结果一致

####################################
# Chapter 4
####################################

class SiLU(nn.Module):  # 定义SiLU激活函数类，继承自PyTorch的nn.Module
    def __init__(self):  # 初始化函数
        super(SiLU, self).__init__()  # 调用父类的初始化函数

    def forward(self, x):  # 前向传播函数，接受输入x
        return x * torch.sigmoid(x)  # 返回x乘以x的sigmoid函数，即SiLU激活函数的计算结果

silu = SiLU()  # 创建SiLU实例

assert torch.allclose(silu(example_batch), torch.nn.functional.silu(example_batch))  # 确保SiLU实现的结果与PyTorch的SiLU一致

####################################
# Chapter 4
####################################

class FeedForward(nn.Module):  # 定义前馈网络类，继承自PyTorch的nn.Module
    def __init__(self, cfg):  # 初始化函数，接受配置字典
        super().__init__()  # 调用父类的初始化函数
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)  # 第一个全连接层
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)  # 第二个全连接层
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)  # 第三个全连接层
        self.silu = SiLU()  # SiLU激活函数

    def forward(self, x):  # 前向传播函数，接受输入x
        x_fc1 = self.fc1(x)  # 通过第一个全连接层
        x_fc2 = self.fc2(x)  # 通过第二个全连接层
        x = self.silu(x_fc1) * x_fc2  # 应用SiLU激活函数并乘以第二个全连接层的输出
        return self.fc3(x)  # 通过第三个全连接层并返回结果

def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096):  # 定义预计算RoPE参数的函数
    assert head_dim % 2 == 0, "Embedding dimension must be even"  # 确保嵌入维度是偶数

    # 计算逆频率
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # 生成位置索引
    positions = torch.arange(context_length)

    # 计算角度
    angles = positions[:, None] * inv_freq[None, :]  # 形状：(context_length, head_dim // 2)

    # 扩展角度以匹配head_dim
    angles = torch.cat([angles, angles], dim=1)  # 形状：(context_length, head_dim)

    # 预计算正弦和余弦
    cos = torch.cos(angles)  # 计算余弦值
    sin = torch.sin(angles)  # 计算正弦值

    return cos, sin  # 返回预计算的余弦和正弦值

def compute_rope(x, cos, sin):
    # 计算旋转位置编码 (Rotary Position Embeddings, RoPE)
    # 输入 x 的形状为 (batch_size, num_heads, seq_len, head_dim)

    batch_size, num_heads, seq_len, head_dim = x.shape  # 提取输入的形状
    assert head_dim % 2 == 0, "Head dimension must be even"  # 确保 head_dim 是偶数

    # 将 x 分成两部分：前一半和后一半
    x1 = x[..., : head_dim // 2]  # 取前一半
    x2 = x[..., head_dim // 2 :]  # 取后一半

    # 调整 sin 和 cos 的形状以匹配输入
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # 形状调整为 (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # 应用旋转变换
    rotated = torch.cat((-x2, x1), dim=-1)  # 将后一半旋转到前一半位置，并反转符号
    x_rotated = (x * cos) + (rotated * sin)  # 计算旋转后的值

    return x_rotated.to(dtype=x.dtype)  # 返回旋转后的结果，确保数据类型与输入一致


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dtype=None):
        # 初始化多头注意力机制模块

        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"
        # 确保输出维度可以被头数整除

        self.d_out = d_out  # 保存总的输出维度
        self.num_heads = num_heads  # 保存头数
        self.head_dim = d_out // num_heads  # 每个头的维度大小

        ################################### 新增部分 ###################################
        # 初始化线性层，bias 设置为 False，同时指定数据类型
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)  # 查询的线性投影层
        self.W_key = nn.Linear(d_in, d_out, bias=False, dtype=dtype)  # 键的线性投影层
        self.W_value = nn.Linear(d_in, d_out, bias=False, dtype=dtype)  # 值的线性投影层
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)  # 输出投影层
        # self.dropout = nn.Dropout(dropout)  # 可选：丢弃层（注释掉）

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        # 注册掩码，用于因果掩蔽 (causal masking)

        # 预计算旋转位置编码参数（RoPE）
        cos, sin = precompute_rope_params(head_dim=self.head_dim, context_length=context_length)
        self.register_buffer("cos", cos)  # 注册 cos 参数
        self.register_buffer("sin", sin)  # 注册 sin 参数



#### **`forward` 前向传播**



def forward(self, x):
    # 前向传播，输入 x 的形状为 (batch_size, seq_len, d_in)

    b, num_tokens, d_in = x.shape  # 提取输入形状

    # 计算查询、键和值
    keys = self.W_key(x)  # 使用键投影层，形状为 (b, num_tokens, d_out)
    queries = self.W_query(x)  # 使用查询投影层
    values = self.W_value(x)  # 使用值投影层

    # 将最后一维展开成 (num_heads, head_dim)
    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
    values = values.view(b, num_tokens, self.num_heads, self.head_dim)
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

    # 转置以匹配多头注意力计算
    keys = keys.transpose(1, 2)  # 转置后形状为 (b, num_heads, num_tokens, head_dim)
    queries = queries.transpose(1, 2)
    values = values.transpose(1, 2)

    ################################### 新增部分 ###################################
    # 使用旋转位置编码（RoPE）对查询和键进行位置编码增强
    keys = compute_rope(keys, self.cos, self.sin)  # 处理键
    queries = compute_rope(queries, self.cos, self.sin)  # 处理查询
    ###########################################################################

    # 计算缩放点积注意力分数（即自注意力）
    attn_scores = queries @ keys.transpose(2, 3)  # 对每个头计算点积，形状为 (b, num_heads, num_tokens, num_tokens)

    # 将掩码调整为指定的序列长度，并转换为布尔值
    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

    # 使用掩码对注意力分数进行填充
    attn_scores.masked_fill_(mask_bool, -torch.inf)  # 被掩盖的分数设置为负无穷大

    # 计算注意力权重
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)  # 缩放并归一化分数
    # attn_weights = self.dropout(attn_weights)  # 可选：对权重进行丢弃（注释掉）

    # 使用注意力权重加权值
    context_vec = (attn_weights @ values).transpose(1, 2)  # 将注意力权重应用于值向量

    # 合并所有头的输出
    context_vec = context_vec.reshape(b, num_tokens, self.d_out)  # 重新调整形状为原始输出维度
    context_vec = self.out_proj(context_vec)  # 可选：对结果进行投影

    return context_vec  # 返回最终结果

# 设置
batch_size = 1  # 批处理大小
context_len = 100  # 上下文长度（即输入序列的长度）
max_context_len = 4096  # 最大上下文长度
embed_dim = 128  # 嵌入维度
num_heads = 4  # 注意力头的数量
# 创建一个随机输入的示例批次，形状为 (batch_size, context_len, embed_dim)
example_batch = torch.randn((batch_size, context_len, embed_dim))

# 实例化一个多头注意力层
mha = MultiHeadAttention(
    d_in=embed_dim,  # 输入的嵌入维度
    d_out=embed_dim,  # 输出的嵌入维度
    context_length=max_context_len,  # 上下文的最大长度
    num_heads=num_heads  # 注意力头的数量
)

# 将示例批次传入多头注意力层进行推理
mha(example_batch)

# 删除 mha 以释放内存
del mha
class TransformerBlock(nn.Module):  # Transformer 块，用于实现每个 Transformer 层
    def __init__(self, cfg):  # 初始化方法，接受配置字典作为参数
        super().__init__()

        # 创建多头注意力层
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],  # 输入嵌入维度
            d_out=cfg["emb_dim"],  # 输出嵌入维度
            context_length=cfg["context_length"],  # 上下文长度
            num_heads=cfg["n_heads"],  # 注意力头的数量
            dtype=cfg["dtype"]  # 数据类型（例如，float32 或 bfloat16）
        )
        # 创建前馈神经网络层
        self.ff = FeedForward(cfg)

        ################################### 新增部分 ###################################
        # 使用 RMSNorm 层进行归一化
        self.norm1 = RMSNorm(cfg["emb_dim"])  # 第一层归一化
        self.norm2 = RMSNorm(cfg["emb_dim"])  # 第二层归一化
        ###########################################################################

        # self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # 可选：丢弃层（注释掉）

    def forward(self, x):  # 前向传播方法
        # 注意力块的快捷连接
        shortcut = x
        x = self.norm1(x)  # 对输入进行归一化
        x = self.att(x)  # 通过多头注意力层计算
        # x = self.drop_shortcut(x)  # 可选：丢弃层（注释掉）
        x = x + shortcut  # 将原始输入加回去，形成残差连接

        # 前馈块的快捷连接
        shortcut = x
        x = self.norm2(x)  # 对输入进行第二次归一化
        x = self.ff(x)  # 通过前馈神经网络层计算
        # x = self.drop_shortcut(x)  # 可选：丢弃层（注释掉）
        x = x + shortcut  # 将原始输入加回去，形成残差连接

        return x  # 返回处理后的输出
# GPT 模型类
class Llama2Model(nn.Module):  # 定义一个 Llama2 模型类（类似于 GPT 模型）
    def __init__(self, cfg):  # 初始化方法，接受配置字典作为参数
        super().__init__()

        # 创建词嵌入层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        # 创建多个 Transformer 块（由 TransformerBlock 组成）
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]  # 堆叠多个 Transformer 层
        )

        ################################### 新增部分 ###################################
        # 使用 RMSNorm 进行最终的归一化
        self.final_norm = RMSNorm(cfg["emb_dim"])
        ###########################################################################

        # 输出线性层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, in_idx):  # 前向传播方法
        # in_idx 是输入的 token 索引
        # 词嵌入：将输入的 token 索引转化为嵌入表示
        tok_embeds = self.tok_emb(in_idx)

        x = tok_embeds  # 这里没有使用位置嵌入（注释掉的位置嵌入）

        # 通过多个 Transformer 块进行处理
        x = self.trf_blocks(x)

        # 对最终的输出进行归一化
        x = self.final_norm(x)

        # 通过输出头（线性层）映射到词汇空间，得到每个 token 的 logits
        logits = self.out_head(x)

        return logits  # 返回模型输出的 logits（每个 token 对应的 logits）

# GPT 模型配置（124M 参数）
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 768,          # 嵌入维度
    "n_heads": 12,           # 注意力头数
    "n_layers": 12,          # Transformer 层数
    "drop_rate": 0.1,        # 丢弃率
    "qkv_bias": False        # 是否使用 Query-Key-Value 偏置
}

# GPT 模型配置（1558M 参数）
GPT_CONFIG_1558M = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 1600,         # 嵌入维度
    "n_heads": 25,           # 注意力头数
    "n_layers": 48,          # Transformer 层数
    "drop_rate": 0.1,        # 丢弃率
    "qkv_bias": False        # 是否使用 Query-Key-Value 偏置
}

# Llama2 模型配置（7B 参数）
LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # 词汇表大小
    "context_length": 4096,  # 上下文长度
    "emb_dim": 4096,         # 嵌入维度
    "n_heads": 32,           # 注意力头数
    "n_layers": 32,          # Transformer 层数
    "hidden_dim": 11008,     # 前馈网络中的中间维度
    "dtype": torch.bfloat16  # 使用低精度类型来减少内存使用
}

model = Llama2Model(LLAMA2_CONFIG_7B)  # 创建一个Llama2模型实例，使用给定的配置

total_params = sum(p.numel() for p in model.parameters())  # 计算模型参数的总数
print(f"Total number of parameters: {total_params:,}")  # 打印模型参数总数

# 定义一个函数来计算模型在内存中的大小
def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0  # 参数元素总数
    total_grads = 0  # 梯度元素总数
    for param in model.parameters():  # 遍历模型的所有参数
        # 计算每个参数的元素总数
        param_size = param.numel()
        total_params += param_size
        # 检查此参数是否存储梯度
        if param.requires_grad:
            total_grads += param_size

    # 计算缓冲区大小（需要内存的非参数）
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # 每个元素的大小（以字节为单位）=（元素数量）*（每个元素的大小）
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size  # 总内存大小（以字节为单位）

    # 将字节转换为千兆字节
    total_memory_gb = total_memory_bytes / (1024**3)
    return total_memory_gb  # 返回总内存大小（以千兆字节为单位）

# 打印不同数据类型下的模型内存大小
print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")

# 检查是否有可用的CUDA设备，否则使用MPS或CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device);  # 将模型部署到指定设备

# 导入Hugging Face Hub的登录模块
from huggingface_hub import login
import json

# 从配置文件中读取访问令牌
with open("config.json", "r") as config_file:
    config = json.load(config_file)
    access_token = config["HF_ACCESS_TOKEN"]

login(token=access_token)  # 使用访问令牌登录Hugging Face Hub

# 从Hugging Face Hub下载分词器模型
from huggingface_hub import hf_hub_download

tokenizer_file = hf_hub_download(
    repo_id="meta-llama/Llama-2-7b",
    filename="tokenizer.model",
    local_dir="Llama-2-7b"
)

# 导入SentencePiece库
import sentencepiece as spm

# 定义LlamaTokenizer类，用于加载和使用分词器
class LlamaTokenizer:
    def __init__(self, tokenizer_file):
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_file)
        self.tokenizer = sp

    def encode(self, text):
        return self.tokenizer.encode_as_ids(text)

    def decode(self, ids):
        return self.tokenizer.decode_pieces(ids)

tokenizer = LlamaTokenizer(tokenizer_file)  # 创建分词器实例

# 导入之前章节中的文本生成相关函数
from previous_chapters import generate, text_to_token_ids, token_ids_to_text

torch.manual_seed(123)  # 设置随机种子

# 生成文本
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves", tokenizer).to(device),
    max_new_tokens=30,
    context_size=LLAMA2_CONFIG_7B["context_length"],
    top_k=1,
    temperature=0.
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))  # 打印生成的文本

# 从Hugging Face Hub下载模型权重文件
weights_file = hf_hub_download(
   repo_id="meta-llama/Llama-2-7b",
   filename="consolidated.00.pth",
   local_dir="Llama-2-7b"
)
# 加载权重文件，并仅加载权重参数
weights = torch.load(weights_file, weights_only=True)

# 检查权重文件中的键（前 15 个键）
list(weights.keys())[:15]

def assign(left, right):
    # 检查左右张量形状是否匹配，如果不匹配，抛出异常
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")

    # 如果右侧是张量，返回复制后的张量并转换为参数
    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())
    else:
        # 如果右侧不是张量，则将其转换为张量
        return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_llama(model, param_config, params):
    # 加载词嵌入层的权重
    model.tok_emb.weight = assign(model.tok_emb.weight, params["tok_embeddings.weight"])

    # 循环加载每一层的权重
    for l in range(param_config["n_layers"]):

        # 加载注意力机制的权重
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            params[f"layers.{l}.attention.wq.weight"]
        )
        model.trf_blocks[l].att.W_key.weight = assign(
            model.trf_blocks[l].att.W_key.weight,
            params[f"layers.{l}.attention.wk.weight"]
        )
        model.trf_blocks[l].att.W_value.weight = assign(
            model.trf_blocks[l].att.W_value.weight,
            params[f"layers.{l}.attention.wv.weight"]
        )
        model.trf_blocks[l].att.out_proj.weight = assign(
            model.trf_blocks[l].att.out_proj.weight,
            params[f"layers.{l}.attention.wo.weight"]
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"layers.{l}.attention_norm.weight"]
        )

        # 加载前馈神经网络层的权重
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"layers.{l}.feed_forward.w1.weight"]
        )
        # 由于某些原因，w2 和 w3 的权重顺序在文件中提供的顺序是反的，因此需要调整
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"layers.{l}.feed_forward.w3.weight"]
        )
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"layers.{l}.feed_forward.w2.weight"]
        )
        model.trf_blocks[l].norm2.weight = assign(
            model.trf_blocks[l].norm2.weight,
            params[f"layers.{l}.ffn_norm.weight"]
        )

    # 加载输出层的权重
    model.final_norm.weight = assign(model.final_norm.weight, params["norm.weight"])
    model.out_head.weight = assign(model.out_head.weight, params["output.weight"])

# 将预训练权重加载到 Llama 模型
load_weights_into_llama(model, LLAMA2_CONFIG_7B, weights)
model.to(device)  # 将模型移到指定的设备上（如 GPU 或 CPU）

# 设置随机种子，保证每次生成结果可复现
torch.manual_seed(123)

# 调用生成文本的函数，生成与 "Every effort" 相关的文本
token_ids = generate(
    model=model,  # 使用加载权重的 Llama 模型进行文本生成
    idx=text_to_token_ids("Every effort", tokenizer).to(device),  # 将输入文本转换为 token IDs，并转移到设备
    max_new_tokens=25,  # 最多生成 25 个新 token
    context_size=LLAMA2_CONFIG_7B["context_length"],  # 上下文大小，通常为 4096
    top_k=1,  # 使用 Top-K 采样，仅考虑概率最高的 1 个候选 token
    temperature=0.  # 温度设置为 0，选择概率最高的 token，确保结果确定性
)

# 输出生成的文本
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

del model  # 删除之前加载的 `model` 对象，释放内存，避免内存溢出。

weights_file = hf_hub_download(
   repo_id="meta-llama/Llama-2-7b-chat",  # Hugging Face 仓库 ID，用于指定权重文件来源。
   filename="consolidated.00.pth",       # 要下载的权重文件名。
   local_dir="Llama-2-7b-chat"           # 本地存储目录。
)
# 下载 Llama-2-7b 的权重文件，路径保存在变量 `weights_file` 中。

model = Llama2Model(LLAMA2_CONFIG_7B)  # 初始化 Llama2 模型实例，使用 7B 参数配置。

load_weights_into_llama(model, LLAMA2_CONFIG_7B, weights)
# 将下载的预训练权重加载到模型参数中：
# - `model` 是初始化的 Llama2 模型。
# - `LLAMA2_CONFIG_7B` 提供模型层数、头数等配置信息。
# - `weights` 包含权重数据。

model.to(device)  # 将模型移动到指定设备（如 GPU），以加速计算。

torch.manual_seed(123)  # 设置随机数种子，确保生成的结果可复现。

token_ids = generate(
    model=model,  # 使用加载权重的 Llama2 模型进行生成。
    idx=text_to_token_ids("What do llamas eat?", tokenizer).to(device),
    # 将输入文本 "What do llamas eat?" 转换为 token IDs 并移动到设备。
    max_new_tokens=25,  # 最大生成 25 个新 token。
    context_size=LLAMA2_CONFIG_7B["context_length"],
    # 模型上下文长度，从 7B 配置中读取（通常为 4096）。
    top_k=1,  # 使用 Top-K 采样，仅考虑概率最高的 1 个候选 token。
    temperature=0.  # 温度设置为 0，选择概率最高的 token，确保结果确定性。
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
# 将生成的 token IDs 转换为可读文本并打印输出。





















