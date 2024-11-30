# 从importlib.metadata模块导入version函数，用于获取库的版本信息
from importlib.metadata import version

# 创建一个包含库名称的列表
pkgs = ["matplotlib",
        "numpy",
        "tiktoken",
        "torch",
        "tensorflow"]  # For OpenAI's pretrained weights
# 遍历列表，打印每个库的版本信息
for p in pkgs:
    print(f"{p} version: {version(p)}")

# 导入PyTorch库
import torch
# 从ch04.main_chapter_code.ch04模块导入GPTModel类
from ch04.main_chapter_code.ch04 import GPTModel

# 定义GPT模型配置参数
GPT_CONFIG_124M = {
    "vocab_size": 50257,   # 词汇表大小
    "context_length": 256, # 缩短的上下文长度（原始：1024）
    "emb_dim": 768,        # 嵌入维度
    "n_heads": 12,         # 注意力头数
    "n_layers": 12,        # 层数
    "drop_rate": 0.1,      # dropout率
    "qkv_bias": False      # QKV偏置
}

# 设置随机种子以确保结果可复现
torch.manual_seed(123)
# 创建GPTModel实例
model = GPTModel(GPT_CONFIG_124M)
# 设置模型为评估模式，禁用dropout
model.eval();

# 导入tiktoken库
import tiktoken
# 从ch04.main_chapter_code.ch04模块导入generate_text_simple函数
from ch04.main_chapter_code.ch04 import generate_text_simple

# 定义一个函数，将文本转换为token ids
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加批次维度
    return encoded_tensor

# 定义一个函数，将token ids转换为文本
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # 移除批次维度
    return tokenizer.decode(flat.tolist())

# 定义起始上下文
start_context = "Every effort moves you"
# 初始化tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# 使用generate_text_simple函数生成文本
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

# 打印输出文本
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# 创建输入张量
inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

# 创建目标张量
targets = torch.tensor([[3626, 6100, 345],  # [" effort moves you",
                        [1107,  588, 11311]]) #  " really like chocolate"]

# 不计算梯度，获取预测
with torch.no_grad():
    logits = model(inputs)

# 计算每个token在词汇表中的概率
probas = torch.softmax(logits, dim=-1) # 每个token的概率
print(probas.shape) # 形状：(batch_size, num_tokens, vocab_size)

# 获取概率最高的token ids
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)

# 打印目标文本和输出文本
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

# 计算两个文本的目标概率
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

# 计算所有token概率的对数
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

# 计算每个token的平均概率
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)

# 打印logits和targets的形状
print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)

# 将logits和targets展平
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()

print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)

# 计算交叉熵损失
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)

# 计算困惑度
perplexity = torch.exp(loss)
print(perplexity)

# 导入os和urllib.request模块
import os
import urllib.request

# 打开并读取文件"the-verdict.txt"
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    text_data = f.read()
# 打印前100个字符
print(text_data[:99])

# 打印最后100个字符
print(text_data[-99:])

# 计算总字符数和token数
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print("Characters:", total_characters)
print("Tokens:", total_tokens)

# 从ch04.main_chapter_code.gpt模块导入create_dataloader_v1函数
from ch04.main_chapter_code.gpt import create_dataloader_v1

# 设置训练/验证集比例
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# 设置随机种子以确保结果可复现
torch.manual_seed(123)

# 创建训练数据加载器
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

# 创建验证数据加载器
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)
# 校验代码的合理性

# 检查训练集所需的 token 数量是否足够
if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("训练数据集的 token 数量不足。"
          "尝试降低 `GPT_CONFIG_124M['context_length']` 或增加 `training_ratio`")

# 检查验证集所需的 token 数量是否足够
if total_tokens * (1 - train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("验证数据集的 token 数量不足。"
          "尝试降低 `GPT_CONFIG_124M['context_length']` 或减少 `training_ratio`")

# 输出训练数据加载器的批次形状
print("训练数据加载器：")
for x, y in train_loader:
    print(x.shape, y.shape)

# 输出验证数据加载器的批次形状
print("\n验证数据加载器：")
for x, y in val_loader:
    print(x.shape, y.shape)

# 计算训练集中的 token 数量
train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

# 计算验证集中的 token 数量
val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

# 输出训练集、验证集和所有数据的 token 总数
print("训练集 token 数量:", train_tokens)
print("验证集 token 数量:", val_tokens)
print("所有 token 总数:", train_tokens + val_tokens)


# 计算单个批次的损失函数
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)  # 获取模型输出
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())  # 计算交叉熵损失
    return loss


# 计算数据加载器中所有批次的损失函数
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")  # 如果数据加载器为空，返回 NaN
    elif num_batches is None:
        num_batches = len(data_loader)  # 默认计算所有批次
    else:
        # 如果 num_batches 超过数据加载器中的批次数量，减少批次数量
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算单个批次的损失
            total_loss += loss.item()  # 累加损失
        else:
            break
    return total_loss / num_batches  # 返回平均损失


# 设置设备为 GPU（如果可用）或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 注意：
# 如果你使用的是苹果的 M1/M2 芯片（Apple Silicon），可以启用 MPS 加速，
# 这样在 Apple Silicon 上的性能约为 CPU 的 2 倍（以 M3 MacBook Air 为例测得）。
# 但是，训练的损失值可能会略有不同。

# 如果有 CUDA 支持（GPU 可用），使用 GPU 否则使用 MPS 或 CPU
# if torch.cuda.is_available():
#    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#    device = torch.device("mps")
# else:
#    device = torch.device("cpu")
#
# print(f"使用 {device} 设备。")

# 将模型加载到指定设备
model.to(device)

# 设置随机种子以确保结果可复现（特别是数据加载时的随机打乱）
torch.manual_seed(123)

# 禁用梯度追踪（因为我们只是在评估模型，不训练）
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)  # 计算训练集损失
    val_loss = calc_loss_loader(val_loader, model, device)  # 计算验证集损失

# 输出训练集和验证集的损失
print("训练集损失:", train_loss)
print("验证集损失:", val_loss)


# 简单的训练模型函数
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # 初始化列表来记录损失和训练过程中看到的 token 数量
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # 主训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 清除上一个批次的梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算当前批次的损失
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新模型权重
            tokens_seen += input_batch.numel()  # 累计训练过程中看到的 token 数量
            global_step += 1  # 记录全局训练步数

            # 每隔 eval_freq 步进行一次评估
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)  # 评估当前模型
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch + 1} (Step {global_step:06d}): "
                      f"训练损失 {train_loss:.3f}, 验证损失 {val_loss:.3f}")

        # 每个 epoch 结束后，打印一个生成的样本文本
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


# 评估模型的函数
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度追踪
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)  # 计算训练集的损失
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)  # 计算验证集的损失
    model.train()  # 恢复训练模式
    return train_loss, val_loss


# 根据提供的上下文生成并打印样本文本
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()  # 设置模型为评估模式
    context_size = model.pos_emb.weight.shape[0]  # 获取模型的位置嵌入维度
    encoded = text_to_token_ids(start_context, tokenizer).to(device)  # 将输入文本转为 token IDs
    with torch.no_grad():  # 禁用梯度追踪
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size  # 生成最大 50 个新 token
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)  # 将 token IDs 转为文本
    print(decoded_text.replace("\n", " "))  # 格式化打印生成的文本
    model.train()  # 恢复训练模式


# Note:
# Uncomment the following code to calculate the execution time
# import time
# start_time = time.time()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)
# 注意：
# 取消以下代码的注释以计算执行时间
# import time
# start_time = time.time()

# 设置随机种子以确保结果可复现
torch.manual_seed(123)
# 创建GPTModel实例
model = GPTModel(GPT_CONFIG_124M)
# 将模型发送到指定设备
model.to(device)
# 创建AdamW优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

# 定义训练的轮数
num_epochs = 10
# 训练模型并返回训练和验证损失
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

# 注意：
# 取消以下代码的注释以显示执行时间
# end_time = time.time()
# execution_time_minutes = (end_time - start_time) / 60
# print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# 导入matplotlib库用于绘图
import matplotlib.pyplot as plt
# 导入matplotlib的MaxNLocator用于设置x轴刻度
from matplotlib.ticker import MaxNLocator


# 定义一个函数，用于绘制训练和验证损失
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 绘制训练和验证损失随轮数的变化
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # x轴仅显示整数标签

    # 创建第二个x轴以显示已看到的token数
    ax2 = ax1.twiny()  # 创建第二个x轴，共享相同的y轴
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 绘制不可见的图以对齐刻度
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # 调整布局以留出空间
    plt.savefig("loss-plot.pdf")
    plt.show()

# 创建一个表示轮数的张量
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# 绘制损失图
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# 将模型移至CPU并设置为评估模式
model.to("cpu")
model.eval()

# 初始化tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# 使用generate_text_simple函数生成文本
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)

# 打印输出文本
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# 定义词汇表和逆向词汇表
vocab = {
    "closer": 0,
    "every": 3,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}

# 假设输入是"every effort moves you"，LLM返回以下logits用于下一个token：
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

# 计算概率
probas = torch.softmax(next_token_logits, dim=0)
# 获取概率最高的token id
next_token_id = torch.argmax(probas).item()

# 下一个生成的token如下：
print(inverse_vocab[next_token_id])

# 设置随机种子以确保结果可复现
torch.manual_seed(123)
# 从概率中采样下一个token id
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])

# 定义一个函数，用于打印采样的token
def print_sampled_tokens(probas):
    torch.manual_seed(123) # 手动设置随机种子以确保可复现性
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

# 打印采样的token
print_sampled_tokens(probas)

# 定义一个函数，用于计算具有温度参数的softmax
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)
# 温度值
temperatures = [1, 0.1, 5]  # 原始温度、更高的置信度、更低的置信度

# 计算加权后的概率
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]

# 绘图
x = torch.arange(len(vocab))  # x 轴为词汇表的索引
bar_width = 0.15  # 设置每个条形图的宽度

fig, ax = plt.subplots(figsize=(5, 3))  # 创建一个图形和坐标轴
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')  # 绘制不同温度下的条形图

# 设置 y 轴标签、x 轴标签和图例
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)  # 将词汇表的词作为 x 轴标签，旋转90度
ax.legend()

# 调整图形布局并保存为 PDF 文件
plt.tight_layout()
plt.savefig("temperature-plot.pdf")
plt.show()  # 展示图形

# 打印不同温度下采样的 token
print_sampled_tokens(scaled_probas[1])
print_sampled_tokens(scaled_probas[2])

# Top-k 采样：选择 top_k 个最大 logits
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)

print("Top logits:", top_logits)
print("Top positions:", top_pos)

# 将 logits 中小于 top_k 最小值的部分设置为 -inf，确保只有 top_k 个最大值参与计算
new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float("-inf")),
    other=next_token_logits
)

print(new_logits)

# 对新的 logits 进行 softmax 归一化，得到概率分布
topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)

# 定义一个生成文本的函数
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # 循环生成新 token：获取模型的 logits，并只关注最后一个时间步的输出
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # 限制上下文长度
        with torch.no_grad():
            logits = model(idx_cond)  # 获取 logits
        logits = logits[:, -1, :]  # 只取最后一步的 logits

        # 新增：使用 top-k 采样进行过滤，只保留 top_k 个最大 logits
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # 新增：应用温度缩放
        if temperature > 0.0:
            logits = logits / temperature  # 对 logits 进行缩放

            # 对 logits 进行 softmax，得到概率分布
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # 从概率分布中采样下一个 token
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # 如果没有温度缩放，则选择 logits 最大值对应的索引
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # 如果遇到结束符（eos_id），则提前停止生成
        if idx_next == eos_id:
            break

        # 将采样的 token 索引添加到已生成的序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

# 设置随机种子，确保生成结果可复现
torch.manual_seed(123)

# 使用生成函数生成文本
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,  # 最大生成 15 个新 token
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,  # 选择 top 25 的 logits
    temperature=1.4  # 使用温度 1.4
)

# 打印生成的文本
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# 保存训练后的模型权重
torch.save(model.state_dict(), "model.pth")

# 重新加载模型
model = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
model.eval();  # 切换为评估模式

# 保存模型和优化器的状态字典
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth"
)

# 重新加载模型和优化器的状态字典
checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)

# 加载模型
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])

# 加载优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train();  # 切换回训练模式

# 打印 TensorFlow 和 tqdm 的版本
# pip install tensorflow tqdm
print("TensorFlow version:", version("tensorflow"))
print("tqdm version:", version("tqdm"))

# 从本地文件下载并加载 GPT-2 模型
from gpt_download import download_and_load_gpt2

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# 输出加载的模型设置和参数字典的键
print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())

# 打印 token 嵌入层的参数并输出维度
print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)

# 定义一个包含不同 GPT-2 模型配置的字典
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# 复制基础配置，并更新为特定模型的设置
model_name = "gpt2-small (124M)"  # 示例使用的模型名称
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])  # 更新为特定模型的配置
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})  # 设置上下文长度和 qkv_bias

# 创建新的模型并切换为评估模式
gpt = GPTModel(NEW_CONFIG)
gpt.eval();

# 定义一个简单的函数，确保两者的形状匹配并返回一个新的参数
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))  # 将 right 转换为一个新的可训练参数


import numpy as np


# 定义一个加载模型权重的函数
def load_weights_into_gpt(gpt, params):
    # 加载位置嵌入的权重
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    # 加载词嵌入的权重
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    # 遍历每一层的 Transformer 块
    for b in range(len(params["blocks"])):
        # 从注意力机制中获取 Q、K、V 权重（Q、K、V 是自注意力机制中的查询、键和值）
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)

        # 将权重矩阵赋值给 GPT 模型的对应层
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 获取 Q、K、V 的偏置
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)

        # 将偏置赋值给 GPT 模型的对应层
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 加载自注意力的输出投影层的权重和偏置
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # 加载前馈神经网络（FFN）中的权重和偏置
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # 加载 LayerNorm 层的缩放因子和偏置
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    # 加载最终的 LayerNorm 层的缩放因子和偏置
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])

    # 加载输出层的权重
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


# 将权重加载到 GPT 模型中
load_weights_into_gpt(gpt, params)
gpt.to(device);  # 将模型移动到指定设备（如 GPU）

# 设置随机种子，确保生成结果可复现
torch.manual_seed(123)

# 生成文本
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),  # 将输入的文本转换为 token ID，并移到设备
    max_new_tokens=25,  # 最大生成 25 个新 token
    context_size=NEW_CONFIG["context_length"],  # 使用模型的上下文长度
    top_k=50,  # 选择 top 50 的 logits
    temperature=1.5  # 使用温度 1.5 生成文本
)

# 打印生成的文本
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))  # 将 token ID 转换为文本并输出
