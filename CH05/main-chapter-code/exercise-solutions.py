# 导入用于获取包版本的模块
from importlib.metadata import version

# 定义要检查版本的包列表
pkgs = ["numpy",
        "tiktoken",
        "torch",
        "tensorflow" # For OpenAI's pretrained weights
       ]
# 遍历包列表并打印每个包的版本
for p in pkgs:
    print(f"{p} version: {version(p)}")

# 导入PyTorch库
import torch

# 定义一个简单的词汇表和其逆向词汇表
vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}

# 定义一个张量，表示下一个token的logits
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

# 定义一个函数，用于打印采样的tokens
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

# 定义一个函数，用于计算softmax并应用温度参数
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

# 定义不同的温度参数
temperatures = [1, 0.1, 5]  # Original, higher, and lower temperature
# 计算不同温度下的softmax概率分布
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]

# 遍历不同温度下的概率分布，并打印采样的tokens
for i, probas in enumerate(scaled_probas):
    print("\n\nTemperature:", temperatures[i])
    print_sampled_tokens(probas)

# 获取特定温度下特定token的索引
temp5_idx = 2
pizza_idx = 6

# 获取温度为5时，'pizza' token的概率
scaled_probas[temp5_idx][pizza_idx]

# 导入tiktoken库和GPTModel类
import tiktoken
import torch
from ch04.main_chapter_code.ch04 import GPTModel

# 定义GPT模型的配置
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,       # Shortened context length (orig: 1024)
    "emb_dim": 768,       # Embedding dimension
    "n_heads": 12,        # Number of attention heads
    "n_layers": 12,       # Number of layers
    "drop_rate": 0.1,     # Dropout rate
    "qkv_bias": False     # Query-key-value bias
}

# 设置PyTorch的随机种子
torch.manual_seed(123)

# 获取GPT-2模型的tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# 创建GPT模型实例并加载预训练权重
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval();

# 导入文本生成相关的函数
from gpt_generate import generate, text_to_token_ids, token_ids_to_text
from ch04.main_chapter_code.ch04 import generate_text_simple

# 使用确定性函数生成文本
start_context = "Every effort moves you"

# 将文本转换为token IDs并生成新的文本
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)

# 打印生成的文本
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# 使用确定性行为生成文本：无top_k，无温度缩放
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=None,
    temperature=0.0
)

# 打印生成的文本
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# 重复上一步操作，再次生成文本
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=None,
    temperature=0.0
)

# 打印生成的文本
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# 导入必要的库和GPTModel类
import tiktoken
import torch
from ch04.main_chapter_code.ch04 import GPTModel

# 定义GPT模型的配置
GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

# 设置设备，优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取GPT-2模型的tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# 加载模型和优化器的状态
checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)

# 创建优化器并加载状态
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train();

# 导入必要的库
import os
import urllib.request
from ch04.main_chapter_code.gpt import create_dataloader_v1

# 读取文本数据
with open("the-verdict.txt", "r", encoding="utf-8") as file:
    text_data = file.read()

# 设置训练/验证数据的比例
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]
# 设置PyTorch的随机种子以确保结果可复现
torch.manual_seed(123)

# 使用create_dataloader_v1函数创建训练数据加载器，设置批量大小、最大序列长度等参数
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

# 使用create_dataloader_v1函数创建验证数据加载器，设置批量大小、最大序列长度等参数
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# 从gpt_train模块导入train_model_simple函数，用于简单训练模型
from gpt_train import train_model_simple

# 定义训练的轮数
num_epochs = 1
# 使用train_model_simple函数训练模型，并返回训练损失、验证损失和看到的token数量
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

# 导入tiktoken库和torch库，以及GPTModel类
import tiktoken
import torch
from ch04.main_chapter_code.ch04 import GPTModel

# 定义GPT模型的配置参数
GPT_CONFIG_124M = {
    "vocab_size": 50257,   # 词汇表大小
    "context_length": 256, # 缩短的上下文长度（原始：1024）
    "emb_dim": 768,        # 嵌入维度
    "n_heads": 12,         # 注意力头数
    "n_layers": 12,        # 层数
    "drop_rate": 0.1,      # Dropout率
    "qkv_bias": False      # Query-key-value偏置
}

# 设置PyTorch的随机种子以确保结果可复现
torch.manual_seed(123)

# 获取GPT-2模型的tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# 从gpt_download模块导入download_and_load_gpt2函数，用于下载和加载GPT2模型
from gpt_download import download_and_load_gpt2

# 使用download_and_load_gpt2函数下载并加载124M大小的GPT2模型
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# 定义不同GPT模型的配置字典，以便于管理和更新配置
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# 复制基础配置，并更新为特定模型的设置
model_name = "gpt2-small (124M)"  # 示例模型名称
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

# 创建GPT模型实例，并使用更新后的配置
gpt = GPTModel(NEW_CONFIG)
gpt.eval();

# 从gpt_generate模块导入load_weights_into_gpt函数，用于将权重加载到GPT模型
from gpt_generate import load_weights_into_gpt

# 设置设备，优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将权重加载到GPT模型，并发送模型到设备（GPU或CPU）
load_weights_into_gpt(gpt, params)
gpt.to(device);
# 导入必要的Python模块，用于文件操作和网络请求
import os
import urllib.request
from ch04.main_chapter_code.gpt import create_dataloader_v1

# 定义要下载的文件的本地路径和URL
file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

# 检查本地文件是否存在，如果不存在，则从URL下载
if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

# 设置训练/验证数据集的比例
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# 设置PyTorch的随机种子以确保结果可复现
torch.manual_seed(123)

# 使用create_dataloader_v1函数创建训练数据加载器
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

# 使用create_dataloader_v1函数创建验证数据加载器
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# 从gpt_train模块导入calc_loss_loader函数，用于计算数据加载器上的损失
from gpt_train import calc_loss_loader

# 再次设置PyTorch的随机种子以确保结果可复现，特别是由于数据加载器中的洗牌
torch.manual_seed(123)
train_loss = calc_loss_loader(train_loader, gpt, device)
val_loss = calc_loss_loader(val_loader, gpt, device)

# 打印训练和验证损失
print("Training loss:", train_loss)
print("Validation loss:", val_loss)

# 从gpt_download模块导入download_and_load_gpt2函数，用于下载和加载GPT2模型
settings, params = download_and_load_gpt2(model_size="1558M", models_dir="gpt2")

# 定义新模型的名称和配置
model_name = "gpt2-xl (1558M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

# 创建GPT模型实例并设置为评估模式
gpt = GPTModel(NEW_CONFIG)
gpt.eval()

# 将预训练权重加载到GPT模型中
load_weights_into_gpt(gpt, params)
gpt.to(device)

# 再次设置PyTorch的随机种子
torch.manual_seed(123)
train_loss = calc_loss_loader(train_loader, gpt, device)
val_loss = calc_loss_loader(val_loader, gpt, device)

# 打印训练和验证损失
print("Training loss:", train_loss)
print("Validation loss:", val_loss)

# 导入必要的库和GPTModel类
import tiktoken
import torch
from ch04.main_chapter_code.ch04 import GPTModel

# 定义GPT模型的配置参数
GPT_CONFIG_124M = {
    "vocab_size": 50257,   # 词汇表大小
    "context_length": 256, # 缩短的上下文长度（原始：1024）
    "emb_dim": 768,        # 嵌入维度
    "n_heads": 12,         # 注意力头数
    "n_layers": 12,        # 层数
    "drop_rate": 0.1,      # Dropout率
    "qkv_bias": False      # Query-key-value偏置
}

# 获取GPT-2模型的tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# 从gpt_download和gpt_generate模块导入相关函数
from gpt_download import download_and_load_gpt2
from gpt_generate import load_weights_into_gpt

# 定义不同GPT模型的配置字典
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# 定义新模型的名称和配置
model_name = "gpt2-xl (1558M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

# 创建GPT模型实例并设置为评估模式
gpt = GPTModel(NEW_CONFIG)
gpt.eval()

# 从gpt_download模块导入download_and_load_gpt2函数，用于下载和加载GPT2模型
settings, params = download_and_load_gpt2(model_size="1558M", models_dir="gpt2")
load_weights_into_gpt(gpt, params)

# 从gpt_generate模块导入文本生成相关的函数
from gpt_generate import generate, text_to_token_ids, token_ids_to_text

# 设置PyTorch的随机种子
torch.manual_seed(123)

# 使用GPT模型生成文本
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)

# 打印生成的文本
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))