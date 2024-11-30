# 从importlib.metadata模块导入version函数，用于获取库的版本信息
from importlib.metadata import version

# 导入matplotlib、tiktoken和torch库
import matplotlib
import tiktoken
import torch

# 打印matplotlib、torch和tiktoken库的版本信息
print("matplotlib version:", version("matplotlib"))
print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))

# 定义GPT模型配置参数
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 词汇表大小
    "context_length": 1024, # 上下文长度
    "emb_dim": 768,         # 嵌入维度
    "n_heads": 12,          # 注意力头数
    "n_layers": 12,         # 层数
    "drop_rate": 0.1,       # dropout率
    "qkv_bias": False       # QKV偏置
}

# 导入torch和torch.nn模块
import torch
import torch.nn as nn

# 定义DummyGPTModel类，这是一个模拟GPT模型的占位符
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 定义token嵌入层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 定义位置嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 定义dropout层
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 使用DummyTransformerBlock的序列作为Transformer块的占位符
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # 使用DummyLayerNorm作为LayerNorm的占位符
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        # 定义输出层
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        # 前向传播函数
        batch_size, seq_len = in_idx.shape
        # 计算token嵌入
        tok_embeds = self.tok_emb(in_idx)
        # 计算位置嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 合并token嵌入和位置嵌入
        x = tok_embeds + pos_embeds
        # 应用dropout
        x = self.drop_emb(x)
        # 通过Transformer块
        x = self.trf_blocks(x)
        # 应用LayerNorm
        x = self.final_norm(x)
        # 计算 logits
        logits = self.out_head(x)
        return logits

# 定义DummyTransformerBlock类，这是一个模拟Transformer块的占位符
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 这个块只是一个简单的占位符

    def forward(self, x):
        # 这个块什么也不做，只是返回它的输入
        return x

# 定义DummyLayerNorm类，这是一个模拟LayerNorm层的占位符
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # 这里的参数只是为了模拟LayerNorm接口

    def forward(self, x):
        # 这层什么也不做，只是返回它的输入
        return x

# 导入tiktoken库
import tiktoken

# 初始化tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# 创建一个空的批次列表
batch = []

# 定义两个文本样本
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

# 将文本编码为token，并添加到批次列表
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
# 将批次列表堆叠成一个张量
batch = torch.stack(batch, dim=0)
# 打印批次张量
print(batch)

# 设置随机种子以确保结果可复现
torch.manual_seed(123)
# 创建DummyGPTModel实例
model = DummyGPTModel(GPT_CONFIG_124M)

# 计算logits
logits = model(batch)
# 打印输出形状
print("Output shape:", logits.shape)
# 打印logits
print(logits)

# 设置随机种子以确保结果可复现
torch.manual_seed(123)

# 创建两个训练样本，每个样本有5个维度（特征）
batch_example = torch.randn(2, 5)

# 定义一个包含线性层和ReLU激活函数的序列模型
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
# 通过模型计算输出
out = layer(batch_example)
# 打印输出
print(out)
# 计算输出的平均值和方差，保持维度以便后续计算
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)

# 打印平均值和方差
print("Mean:\n", mean)
print("Variance:\n", var)

# 标准化输出，使其均值为0，方差为1
out_norm = (out - mean) / torch.sqrt(var)
print("Normalized layer outputs:\n", out_norm)

# 重新计算标准化后的平均值和方差
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

# 设置PyTorch打印选项，关闭科学计数法
torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean)
print("Variance:\n", var)

# 定义LayerNorm类，实现层归一化
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除以0的一个小值
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 学习缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 学习偏移参数

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # 计算均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算方差，不使用无偏估计
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 归一化
        return self.scale * norm_x + self.shift  # 应用缩放和偏移

# 创建LayerNorm实例并应用到输入
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)

# 计算LayerNorm输出的平均值和方差
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

# 打印LayerNorm输出的平均值和方差
print("Mean:\n", mean)
print("Variance:\n", var)

# 定义GELU类，实现GELU激活函数
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# 导入matplotlib库用于绘图
import matplotlib.pyplot as plt

# 创建GELU和ReLU激活函数实例
gelu, relu = GELU(), nn.ReLU()

# 生成一些样本数据
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)

# 绘制GELU和ReLU激活函数的图形
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
plt.show()

# 定义FeedForward类，实现前馈网络
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

# 打印GPT模型配置中的嵌入维度
print(GPT_CONFIG_124M["emb_dim"])

# 创建FeedForward实例并应用到输入
ffn = FeedForward(GPT_CONFIG_124M)

# 生成随机输入并应用前馈网络
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)

# 定义ExampleDeepNeuralNetwork类，实现一个深度神经网络示例
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # 计算当前层的输出
            layer_output = layer(x)
            # 检查是否可以应用快捷连接
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

# 定义一个函数，用于打印模型参数的梯度
def print_gradients(model, x):
    # 前向传播
    output = model(x)
    # 定义目标张量，这里只是一个形状为[[0.]]的张量
    target = torch.tensor([[0.]])

    # 使用均方误差损失函数计算损失
    loss = nn.MSELoss()
    loss = loss(output, target)

    # 反向传播，计算梯度
    loss.backward()

    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        # 如果参数名中包含'weight'，则打印其梯度的绝对值的均值
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

# 定义网络层的大小
layer_sizes = [3, 3, 3, 3, 3, 1]

# 创建一个样本输入张量
sample_input = torch.tensor([[1., 0., -1.]])

# 设置随机种子以确保结果可复现
torch.manual_seed(123)
# 创建一个不使用快捷连接的深度神经网络模型
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)
# 打印不使用快捷连接的模型的梯度
print_gradients(model_without_shortcut, sample_input)

# 设置随机种子以确保结果可复现
torch.manual_seed(123)
# 创建一个使用快捷连接的深度神经网络模型
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)
# 打印使用快捷连接的模型的梯度
print_gradients(model_with_shortcut, sample_input)

# 从特定模块导入MultiHeadAttention类
from ch03.main_chapter_code.ch03 import MultiHeadAttention

# 定义TransformerBlock类，实现Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 初始化多头自注意力模块
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        # 初始化前馈网络
        self.ff = FeedForward(cfg)
        # 初始化两个层归一化模块
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        # 初始化dropout模块
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 注意力块的快捷连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # 将原始输入加回来

        # 前馈网络块的快捷连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 将原始输入加回来

        return x

# 设置随机种子以确保结果可复现
torch.manual_seed(123)

# 创建一个随机输入张量
x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
# 创建TransformerBlock实例
block = TransformerBlock(GPT_CONFIG_124M)
# 计算输出
output = block(x)

# 打印输入和输出的形状
print("Input shape:", x.shape)
print("Output shape:", output.shape)
# 定义GPTModel类，实现GPT模型
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 定义token嵌入层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 定义位置嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 定义dropout层
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 使用TransformerBlock构建Transformer层的序列
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # 定义最终的层归一化
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 定义输出层
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    # 前向传播函数
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        # 计算token嵌入
        tok_embeds = self.tok_emb(in_idx)
        # 计算位置嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 合并token嵌入和位置嵌入
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        # 通过Transformer层
        x = self.trf_blocks(x)
        # 应用最终的层归一化
        x = self.final_norm(x)
        # 计算logits
        logits = self.out_head(x)
        return logits

# 设置随机种子以确保结果可复现
torch.manual_seed(123)
# 创建GPTModel实例
model = GPTModel(GPT_CONFIG_124M)

# 通过模型计算输出
out = model(batch)
# 打印输入批次和输出形状
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)
# 计算模型参数总数
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# 打印token嵌入层和输出层的形状
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)
# 计算考虑权重共享的可训练参数数
total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

# 计算模型总大小（以字节为单位，假设float32，每个参数4字节）
total_size_bytes = total_params * 4
# 转换为兆字节
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")

# 定义一个函数，用于简单生成文本
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx是当前上下文的索引数组
    for _ in range(max_new_tokens):
        # 如果当前上下文超过支持的上下文大小，则裁剪
        idx_cond = idx[:, -context_size:]
        # 不计算梯度，获取预测
        with torch.no_grad():
            logits = model(idx_cond)
        # 只关注最后一个时间步
        logits = logits[:, -1, :]
        # 应用softmax获取概率
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
        # 获取概率最高的词汇表索引
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        # 将采样的索引追加到运行序列
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)
    return idx

# 定义起始上下文
start_context = "Hello, I am"
# 编码起始上下文
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)

# 将编码后的文本转换为张量
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

# 设置模型为评估模式，禁用dropout
model.eval()

# 生成文本
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

# 打印输出和输出长度
print("Output:", out)
print("Output length:", len(out[0]))

# 解码输出
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)

