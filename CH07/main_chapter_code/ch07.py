from importlib.metadata import version

pkgs = [
    "matplotlib",  # 绘图库
    "tiktoken",    # 分词器
    "torch",       # 深度学习库
    "tqdm",        # 进度条
    "tensorflow",  # OpenAI 预训练权重
]
for p in pkgs:
    print(f"{p} version: {version(p)}")
    # 输出每个包的版本信息

import json
import os
import urllib.request  # Python 3 的正确导入方式

# 定义下载并加载文件的函数
def download_and_load_file(file_path, url):

    # 如果文件不存在，则从 URL 下载文件并保存
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        # 如果文件已经存在，则读取文件内容
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    # 将文件内容转换为 JSON 格式的数据
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

# 下载文件的 URL 和保存路径
file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

# 调用函数下载并加载文件数据
data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))  # 输出数据条目数量

# 打印数据的部分示例
print("Example entry:\n", data[50])
print("Another example entry:\n", data[999])

# 定义格式化输入的函数
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # 如果有输入文本，添加到格式化的文本中
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text  # 返回格式化后的文本

# 获取第 50 条和第 999 条数据，格式化为输入和期望的输出
model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"

print(model_input + desired_response)

model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"

print(model_input + desired_response)

# 划分训练集、验证集和测试集
train_portion = int(len(data) * 0.85)  # 85% 用于训练
test_portion = int(len(data) * 0.1)    # 10% 用于测试
val_portion = len(data) - train_portion - test_portion  # 剩余 5% 用于验证

# 划分数据集
train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

# 输出各数据集的长度
print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))

# 导入 PyTorch 和 Dataset
import torch
from torch.utils.data import Dataset

# 定义数据集类
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # 对文本进行预处理（分词）
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)  # 格式化指令和输入
            response_text = f"\n\n### Response:\n{entry['output']}"  # 格式化输出
            full_text = instruction_plus_input + response_text
            # 对每个完整文本进行编码并保存
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]  # 返回指定索引的编码文本

    def __len__(self):
        return len(self.data)  # 返回数据集大小

# 使用 tiktoken 分词器
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

# 打印结束符 token 的编码
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

# 自定义的批处理函数
def custom_collate_draft_1(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    # 找到批次中最长的序列，并将最大长度加1
    # 这会在填充时加入一个额外的 padding token
    batch_max_length = max(len(item)+1 for item in batch)

    # 创建输入序列列表
    inputs_lst = []

    for item in batch:
        new_item = item.copy()
        # 添加 <|endoftext|> token
        new_item += [pad_token_id]
        # 将序列填充到 batch_max_length 长度
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        # 使用 padded[:-1] 去掉添加的额外 padding token
        # 这个额外的 padding token 在后续代码中会有作用
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    # 将列表转换为张量并转移到目标设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor

# 创建一些示例数据
inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]

batch = (
    inputs_1,
    inputs_2,
    inputs_3
)

# 调用自定义的批处理函数并打印结果
print(custom_collate_draft_1(batch))


print(custom_collate_draft_1(batch))
def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    # 找到批次中最长的序列，并将其长度加 1（为添加特殊 token 做准备）
    batch_max_length = max(len(item)+1 for item in batch)

    # 准备用于输入和目标（标签）的列表
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()  # 复制序列，避免直接修改原数据
        # 添加特殊 token `<|endoftext|>`
        new_item += [pad_token_id]
        # 将序列填充到统一长度
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        # `inputs` 不包含最后一个 token
        inputs = torch.tensor(padded[:-1])
        # `targets` 向右移动 1 位（用于生成任务的目标）
        targets = torch.tensor(padded[1:])
        # 将处理好的输入和目标分别添加到对应列表中
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 将列表转换为张量，并移动到指定设备上
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

# 使用示例数据调用 `custom_collate_draft_2`
inputs, targets = custom_collate_draft_2(batch)
print(inputs)
print(targets)
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # 找到批次中最长的序列长度并加 1
    batch_max_length = max(len(item)+1 for item in batch)

    # 准备输入和目标列表
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # 添加特殊 token `<|endoftext|>`
        new_item += [pad_token_id]
        # 填充序列到统一长度
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        # 定义输入和目标序列
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        # 替换目标中的填充值为 `ignore_index`
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 可选：截断序列到允许的最大长度
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        # 添加处理好的输入和目标
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 转换为张量并移动到设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

# 使用示例数据调用函数
inputs, targets = custom_collate_fn(batch)
print(inputs)
print(targets)
logits_1 = torch.tensor(
    [[-1.0, 1.0],  # 第一条样本的 logits
     [-0.5, 1.5]]  # 第二条样本的 logits
)
targets_1 = torch.tensor([0, 1])  # 第一条样本目标为类 0，第二条为类 1

# 计算交叉熵损失
loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
print(loss_1)

logits_2 = torch.tensor(
    [[-1.0, 1.0],     # 第一条样本的 logits
     [-0.5, 1.5],     # 第二条样本的 logits
     [-0.5, 1.5]]     # 第三条样本的 logits
)
targets_2 = torch.tensor([0, 1, 1])  # 对应的目标

# 再次计算损失
loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
print(loss_2)

# 目标中包含 `-100`，用于忽略某些 token 的计算
targets_3 = torch.tensor([0, 1, -100])

# 计算损失，忽略 `-100` 的部分
loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
print(loss_3)
print("loss_1 == loss_3:", loss_1 == loss_3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 提供对 Apple Silicon 的支持
#if torch.cuda.is_available():
#    device = torch.device("cuda")
#elif torch.backends.mps.is_available():
#    device = torch.device("mps")
#else:
#    device = torch.device("cpu")
from functools import partial

# 使用 `partial` 提供自定义 collate 函数
customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

from torch.utils.data import DataLoader

# 设置 DataLoader 的参数
num_workers = 0
batch_size = 8
torch.manual_seed(123)  # 设置随机种子

# 加载训练集
train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,      # 打乱数据
    drop_last=True,    # 丢弃最后一个不足 batch 的数据
    num_workers=num_workers
)

# 加载验证集
val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,     # 验证集不打乱
    drop_last=False,
    num_workers=num_workers
)

# 加载测试集
test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,     # 测试集不打乱
    drop_last=False,
    num_workers=num_workers
)

print("Train loader:")
# 遍历训练数据加载器，打印输入和目标的张量形状
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)

# 打印第一个批次的输入
print(inputs[0])

# 打印第一个批次的目标
print(targets[0])
# 导入下载和加载 GPT-2 模型的工具函数
from ch05.main_chapter_code.gpt_download import download_and_load_gpt2
# 导入 GPT 模型及权重加载的相关方法
from ch06.main_chapter_code.ch06 import GPTModel, load_weights_into_gpt
BASE_CONFIG = {
    "vocab_size": 50257,     # GPT-2 的词汇表大小
    "context_length": 1024,  # 最大上下文长度
    "drop_rate": 0.0,        # Dropout 概率，设为 0 表示禁用 Dropout
    "qkv_bias": True         # 是否启用查询-键-值的偏置项
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},    # GPT-2 小模型
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},  # GPT-2 中模型
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},   # GPT-2 大模型
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},     # GPT-2 超大模型
}
CHOOSE_MODEL = "gpt2-medium (355M)"  # 选择 GPT-2 的中等模型

# 更新基础配置，添加选择模型的嵌入维度、层数和注意力头数
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
# 获取选择的模型大小，例如 "355M"
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

# 下载和加载 GPT-2 模型权重及相关配置
settings, params = download_and_load_gpt2(
    model_size=model_size,  # 模型大小
    models_dir="gpt2"       # 模型存储目录
)
# 初始化 GPT 模型，使用更新后的基础配置
model = GPTModel(BASE_CONFIG)

# 加载权重到模型中
load_weights_into_gpt(model, params)

# 将模型设置为评估模式
model.eval()
# 设置随机数种子，以确保结果可复现
torch.manual_seed(123)

# 格式化第一个验证数据的输入
input_text = format_input(val_data[0])

# 打印格式化的输入文本
print(input_text)

# 从第 5 章导入文本生成相关函数
from ch05.main_chapter_code.ch05 import (
    generate,           # 文本生成函数
    text_to_token_ids,  # 将文本转换为 token IDs 的函数
    token_ids_to_text   # 将 token IDs 转换回文本的函数
)


# 使用模型生成文本
token_ids = generate(
    model=model,                           # 使用的 GPT 模型
    idx=text_to_token_ids(input_text, tokenizer),  # 输入文本转换为 token IDs
    max_new_tokens=35,                     # 最大生成的 token 数
    context_size=BASE_CONFIG["context_length"],  # 上下文长度
    eos_id=50256,                          # 结束符号的 token ID
)

# 将生成的 token IDs 转换为文本
generated_text = token_ids_to_text(token_ids, tokenizer)
# 提取生成的响应文本
response_text = (
    generated_text[len(input_text):]       # 去掉输入部分，仅保留生成的部分
    .replace("### Response:", "")          # 去掉 "### Response:" 标记
    .strip()                               # 去掉首尾的空格
)

# 打印最终生成的响应文本
print(response_text)
# 导入计算损失和简单模型训练的函数
from ch05.main_chapter_code.ch05 import (
    calc_loss_loader,   # 用于计算数据加载器的损失
    train_model_simple  # 简化的模型训练函数
)

# 将模型移动到指定设备（CPU 或 GPU）
model.to(device)

# 设置随机种子以确保结果的可复现性
torch.manual_seed(123)

# 使用训练集和验证集分别计算前 5 个批次的平均损失
with torch.no_grad():  # 在计算损失时禁用梯度计算，节省内存
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

# 打印训练和验证集的损失
print("Training loss:", train_loss)
print("Validation loss:", val_loss)

# 导入时间模块用于计算训练时间
import time

# 记录训练开始时间
start_time = time.time()

# 设置随机种子
torch.manual_seed(123)

# 定义优化器，使用 AdamW 优化算法
optimizer = torch.optim.AdamW(
    model.parameters(),  # 模型参数
    lr=0.00005,          # 学习率
    weight_decay=0.1     # 权重衰减，避免过拟合
)

# 定义训练的轮次
num_epochs = 2

# 使用简单训练函数训练模型，并记录训练和验证损失
train_losses, val_losses, tokens_seen = train_model_simple(
    model,                # 训练的模型
    train_loader,         # 训练数据加载器
    val_loader,           # 验证数据加载器
    optimizer,            # 优化器
    device,               # 运行设备
    num_epochs=num_epochs,  # 训练的轮次
    eval_freq=5,          # 每 5 次迭代评估一次
    eval_iter=5,          # 评估时的迭代次数
    start_context=format_input(val_data[0]),  # 初始上下文（来自验证数据的格式化输入）
    tokenizer=tokenizer   # 分词器
)

# 记录训练结束时间
end_time = time.time()

# 计算训练所需的总时间（以分钟为单位）
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# 导入绘图函数以可视化训练和验证损失
from ch05.main_chapter_code.ch05 import plot_losses

# 创建一个张量，表示从 0 到训练轮次数的线性区间，用于绘制 x 轴
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))

# 绘制训练和验证损失随时间的变化图
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# 再次设置随机种子以确保结果一致
torch.manual_seed(123)

# 测试模型，打印生成的文本
for entry in test_data[:3]:  # 对测试数据的前 3 个条目进行测试

    # 格式化输入
    input_text = format_input(entry)

    # 使用模型生成响应
    token_ids = generate(
        model=model,                           # 使用的模型
        idx=text_to_token_ids(input_text, tokenizer).to(device),  # 输入文本转换为 token IDs
        max_new_tokens=256,                    # 最大生成 token 数
        context_size=BASE_CONFIG["context_length"],  # 上下文长度
        eos_id=50256                           # 结束标记
    )
    # 将生成的 token IDs 转换为文本
    generated_text = token_ids_to_text(token_ids, tokenizer)

    # 提取模型生成的响应部分，去掉格式化标签并去除空格
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )

    # 打印输入文本、正确的响应和模型的响应
    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")

# 导入 tqdm 进度条工具，用于显示进度
from tqdm import tqdm

# 遍历整个测试集，为每个条目生成响应并保存
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

    # 格式化输入
    input_text = format_input(entry)

    # 使用模型生成响应
    token_ids = generate(
        model=model,                           # 使用的模型
        idx=text_to_token_ids(input_text, tokenizer).to(device),  # 输入文本转换为 token IDs
        max_new_tokens=256,                    # 最大生成 token 数
        context_size=BASE_CONFIG["context_length"],  # 上下文长度
        eos_id=50256                           # 结束标记
    )
    # 将生成的 token IDs 转换为文本
    generated_text = token_ids_to_text(token_ids, tokenizer)

    # 提取模型生成的响应部分，去掉格式化标签并去除空格
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    # 将生成的响应添加到测试数据中
    test_data[i]["model_response"] = response_text

# 将带有模型响应的测试数据保存为 JSON 文件
with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)  # 使用缩进格式化输出，便于阅读

# 打印测试数据的第一个条目，查看模型的响应
print(test_data[0])
import re  # 导入正则表达式模块

# 将模型状态保存到文件
file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"  # 去除模型名称中的空格和括号以生成文件名
torch.save(model.state_dict(), file_name)  # 保存模型的状态字典
print(f"Model saved as {file_name}")  # 打印保存的文件名

# 加载模型的方法
# model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))

import psutil  # 导入 psutil 模块以检查运行中的进程

# 定义检查特定进程是否运行的函数
def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):  # 遍历当前运行的所有进程
        if process_name in proc.info["name"]:  # 如果进程名称匹配
            running = True
            break
    return running

# 检查 Ollama 服务是否正在运行
ollama_running = check_if_running("ollama")

if not ollama_running:  # 如果 Ollama 未运行，抛出异常
    raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
print("Ollama running:", check_if_running("ollama"))  # 打印 Ollama 的运行状态

# 可选：允许重启 notebook 后仅运行指定部分代码
import json  # 导入 JSON 模块
from tqdm import tqdm  # 导入 tqdm 模块，用于显示进度条

# 加载包含响应的测试数据文件
file_path = "instruction-data-with-response.json"

with open(file_path, "r") as file:
    test_data = json.load(file)  # 将 JSON 文件加载为 Python 字典

# 定义输入格式化函数
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    # 如果输入存在，附加到格式化文本中
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

import urllib.request  # 导入用于发送 HTTP 请求的模块

# 定义查询模型生成响应的函数
def query_model(
    prompt,  # 提示文本
    model="llama3",  # 使用的模型名称
    url="http://localhost:11434/api/chat"  # 本地服务的 URL
):
    # 构建数据负载
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}  # 用户的输入内容
        ],
        "options": {     # 配置生成响应的选项
            "seed": 123,             # 随机种子，确保响应一致性
            "temperature": 0,        # 温度设为 0，生成确定性响应
            "num_ctx": 2048          # 上下文最大长度
        }
    }

    # 将数据负载转为 JSON 格式并编码为字节
    payload = json.dumps(data).encode("utf-8")

    # 创建 HTTP POST 请求并添加必要的头部信息
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )
    request.add_header("Content-Type", "application/json")

    # 发送请求并获取响应
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # 逐行读取并解码响应
        while True:
            line = response.readline().decode("utf-8")
            if not line:  # 如果没有更多内容，则退出循环
                break
            response_json = json.loads(line)  # 解析响应为 JSON
            response_data += response_json["message"]["content"]  # 累加生成的内容

    return response_data  # 返回生成的响应

# 测试查询函数
model = "llama3"  # 模型名称
result = query_model("What do Llamas eat?", model)  # 发送简单问题到模型
print(result)  # 打印响应

# 对测试数据中的前三个条目进行评分
for entry in test_data[:3]:
    prompt = (
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{entry['model_response']}`"
        f" on a scale from 0 to 100, where 100 is the best score. "
    )
    print("\nDataset response:")
    print(">>", entry['output'])  # 打印数据集中正确的响应
    print("\nModel response:")
    print(">>", entry["model_response"])  # 打印模型生成的响应
    print("\nScore:")
    print(">>", query_model(prompt))  # 打印模型评分
    print("\n-------------------------")

# 定义一个函数对数据集中的模型响应进行评分
def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []  # 初始化空列表以存储评分
    for entry in tqdm(json_data, desc="Scoring entries"):  # 显示评分进度条
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)  # 发送提示到模型并获取响应
        try:
            scores.append(int(score))  # 将响应转换为整数并添加到评分列表
        except ValueError:  # 如果转换失败，打印错误并跳过
            print(f"Could not convert score: {score}")
            continue

    return scores  # 返回评分列表

# 使用评分函数对测试数据评分
scores = generate_model_scores(test_data, "model_response")
print(f"Number of scores: {len(scores)} of {len(test_data)}")  # 打印评分数量
print(f"Average score: {sum(scores)/len(scores):.2f}\n")  # 打印平均评分
