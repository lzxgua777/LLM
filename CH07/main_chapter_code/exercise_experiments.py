# 版权声明：Sebastian Raschka 根据 Apache License 2.0（参见 LICENSE.txt）持有版权。
# "Build a Large Language Model From Scratch" 的来源
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch
#
# 运行练习的代码；更多信息请参见 exercise-solutions.ipynb

from functools import partial  # 导入partial函数，用于部分应用一个函数
from importlib.metadata import version  # 导入version函数，用于获取库的版本信息
import json  # 导入json模块，用于处理JSON数据
import math  # 导入math模块，用于数学计算
import os  # 导入os模块，用于操作系统相关功能
import re  # 导入re模块，用于正则表达式操作
import time  # 导入time模块，用于时间相关功能
import urllib  # 导入urllib模块，用于URL请求

import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘图
from matplotlib.ticker import MaxNLocator  # 导入matplotlib的MaxNLocator，用于设置坐标轴刻度
import tiktoken  # 导入tiktoken库，用于处理GPT模型的分词
import torch  # 导入PyTorch库，用于深度学习模型的构建和训练
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch的数据集和数据加载器模块
from tqdm import tqdm  # 导入tqdm模块，用于显示进度条

# 从当前文件夹中的本地文件导入
from gpt_download import download_and_load_gpt2  # 导入下载和加载GPT-2模型的函数
from previous_chapters import (
    calc_loss_loader,  # 导入计算数据加载器损失的函数
    generate,  # 导入文本生成函数
    GPTModel,  # 导入GPT模型类
    load_weights_into_gpt,  # 导入将权重加载到GPT模型的函数
    text_to_token_ids,  # 导入文本到标记ID的转换函数
    train_model_simple,  # 导入简单模型训练函数
    token_ids_to_text  # 导入标记ID到文本的转换函数
)

# 定义一个用于处理指令数据集的类
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data  # 存储数据

        # 预标记化文本
        self.encoded_texts = []  # 存储编码后的文本
        for entry in data:
            instruction_plus_input = format_input(entry)  # 格式化输入
            response_text = f"\n\n### Response:\n{entry['output']}"  # 格式化响应
            full_text = instruction_plus_input + response_text  # 拼接完整的文本
            self.encoded_texts.append(
                tokenizer.encode(full_text)  # 使用tokenizer对文本进行编码
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]  # 根据索引返回编码后的文本

    def __len__(self):
        return len(self.data)  # 返回数据集中的样本数量
# 定义一个带有掩码的指令数据集类
class InstructionDatasetWithMasking(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data  # 存储数据

        # 新增：用于存储指令长度的列表
        self.instruction_lengths = []
        self.encoded_texts = []  # 存储编码后的文本列表

        for entry in data:
            instruction_plus_input = format_input(entry)  # 格式化输入
            response_text = f"\n\n### Response:\n{entry['output']}"  # 格式化响应文本
            full_text = instruction_plus_input + response_text  # 拼接完整的文本

            self.encoded_texts.append(
                tokenizer.encode(full_text)  # 使用tokenizer对文本进行编码并添加到列表
            )

            # 新增：收集指令长度
            instruction_length = len(tokenizer.encode(instruction_plus_input))
            self.instruction_lengths.append(instruction_length)  # 将指令长度添加到列表

    def __getitem__(self, index):
        # 新增：分别返回指令长度和文本
        return self.instruction_lengths[index], self.encoded_texts[index]

    def __len__(self):
        return len(self.data)  # 返回数据集中的样本数量


# 定义一个Phi版本的指令数据集类
class InstructionDatasetPhi(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data  # 存储数据

        # 预标记化文本
        self.encoded_texts = []  # 存储编码后的文本列表
        for entry in data:
            # 使用新的format_input_phi函数和调整后的响应文本模板
            instruction_plus_input = format_input_phi(entry)
            response_text = f"\n<|assistant|>:\n{entry['output']}"
            full_text = instruction_plus_input + response_text  # 拼接完整的文本
            self.encoded_texts.append(
                tokenizer.encode(full_text)  # 使用tokenizer对文本进行编码并添加到列表
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]  # 根据索引返回编码后的文本

    def __len__(self):
        return len(self.data)  # 返回数据集中的样本数量


# 定义一个带有LoRA的线性层类
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear  # 原始线性层
        self.lora = LoRALayer(  # LoRA层
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)  # 将线性层和LoRA层的输出相加


# 定义LoRA层类
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))  # 参数A
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # 初始化参数A
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))  # 参数B
        self.alpha = alpha  # 缩放因子alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)  # 计算LoRA层的输出
        return x


# 定义一个函数，用于将模型中的线性层替换为带有LoRA的线性层
def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # 将线性层替换为带有LoRA的线性层
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 递归地对子模块应用相同的函数
            replace_linear_with_lora(module, rank, alpha)


# 定义一个自定义的批处理函数
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # 找到批次中最长的序列
    batch_max_length = max(len(item)+1 for item in batch)

    # 填充并准备输入和目标
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # 添加一个 <|endoftext|> 标记
        new_item += [pad_token_id]
        # 将序列填充到最大长度
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # 截断最后一个标记作为输入
        targets = torch.tensor(padded[1:])  # 向右移动一个位置作为目标

        # 将目标中除第一个填充标记外的所有标记替换为ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 可选地截断到最大序列长度
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 将输入和目标列表转换为张量并传输到目标设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor
# 定义一个自定义的批处理函数，用于数据填充和掩码处理
def custom_collate_with_masking_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # 找到批次中最长的序列
    batch_max_length = max(len(item)+1 for instruction_length, item in batch)  # 现在batch是一个元组

    # 填充并准备输入和目标
    inputs_lst, targets_lst = [], []

    for instruction_length, item in batch:  # 现在batch是一个元组
        new_item = item.copy()
        # 添加一个 <|endoftext|> 标记
        new_item += [pad_token_id]
        # 将序列填充到最大长度
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # 截断最后一个标记作为输入
        targets = torch.tensor(padded[1:])  # 向右移动一个位置作为目标

        # 将目标中除第一个填充标记外的所有标记替换为ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 掩码处理所有输入和指令标记在目标中
        targets[:instruction_length-1] = -100

        # 可选地截断到最大序列长度
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 将输入和目标列表转换为张量并传输到目标设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


# 定义一个下载并加载文件的函数
def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r") as file:
        data = json.load(file)

    return data


# 定义一个格式化输入的函数，用于Phi版本的数据
def format_input_phi(entry):
    instruction_text = (
        f"<|user|>\n{entry['instruction']}"
    )

    input_text = f"\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


# 定义一个格式化输入的函数
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


# 定义一个绘制损失图的函数
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, plot_name):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制训练和验证损失随epoch的变化
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # x轴只显示整数标签

    # 创建第二个x轴用于显示看到的标记数
    ax2 = ax1.twiny()  # 创建第二个x轴，共享y轴
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 绘制一个不可见的图，用于对齐刻度
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # 调整布局以留出空间
    print(f"Plot saved as {plot_name}")
    plt.savefig(plot_name)
    # plt.show()


# 定义主函数，用于根据不同的参数执行不同的操作
def main(mask_instructions=False, alpaca52k=False, phi3_prompt=False, lora=False):
    #######################################
    # 打印包版本
    #######################################
    print()
    pkgs = [
        "matplotlib",  # 绘图库
        "tiktoken",    # 分词器
        "torch",       # 深度学习库
        "tqdm",        # 进度条
        "tensorflow",  # 用于OpenAI的预训练权重
    ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")
    print(50*"-")
    #######################################
    # Download and prepare dataset
    #######################################

    file_path = "instruction-data.json"  # 设置数据集文件路径

    # 根据选择的条件（是否使用Alpaca52k数据集）选择下载的URL
    if alpaca52k:
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json&#34"  # Alpaca52k数据集URL
    else:
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json&#34"  # 默认数据集URL

    # 调用下载并加载数据集的函数
    data = download_and_load_file(file_path, url)

    # 将数据集划分为训练集（85%）、测试集（10%）和验证集（5%）
    train_portion = int(len(data) * 0.85)  # 85%用于训练
    test_portion = int(len(data) * 0.1)  # 10%用于测试

    train_data = data[:train_portion]  # 训练集
    test_data = data[train_portion:train_portion + test_portion]  # 测试集
    val_data = data[train_portion + test_portion:]  # 验证集

    # 打印数据集的长度信息
    print("Training set length:", len(train_data))  # 打印训练集长度
    print("Validation set length:", len(val_data))  # 打印验证集长度
    print("Test set length:", len(test_data))  # 打印测试集长度
    print(50 * "-")  # 打印分割线

    # 初始化tokenizer，并根据是否有CUDA设备决定模型的设备
    tokenizer = tiktoken.get_encoding("gpt2")  # 使用GPT-2的tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用GPU（如果有）或CPU
    print("Device:", device)  # 打印设备信息
    print(50 * "-")  # 打印分割线

    # 根据数据集类型设置最大序列长度
    if alpaca52k:
        allowed_max_length = 512  # Alpaca52k数据集最大长度512
    else:
        allowed_max_length = 1024  # 默认最大长度1024

    # 检查是否同时启用了instruction masking和Phi-3 prompt模板
    if mask_instructions and phi3_prompt:
        raise ValueError(
            "Simultaneous support for instruction masking and the Phi-3 prompt template has not been implemented, yet.")  # 抛出错误，不能同时启用这两个功能

    # 根据不同的条件选择collate_fn和数据集类
    if mask_instructions:
        # 如果启用了mask_instructions，则使用带masking的collate_fn
        customized_collate_fn = partial(custom_collate_with_masking_fn, device=device,
                                        allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDatasetWithMasking  # 使用带masking的数据集类
    elif phi3_prompt:
        # 如果启用了phi3_prompt，则使用标准的collate_fn
        customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDatasetPhi  # 使用phi3_prompt数据集类
    else:
        # 否则，使用默认的collate_fn和数据集类
        customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDataset  # 默认数据集类

    num_workers = 0  # 设置数据加载时使用的子进程数

    # 根据是否使用Alpaca52k数据集设置batch_size
    if alpaca52k:
        batch_size = 4  # 使用Alpaca52k时batch_size设为4
    else:
        batch_size = 8  # 默认batch_size设为8

    torch.manual_seed(123)  # 设置随机种子

    # 创建训练集DataLoader
    train_dataset = CustomDataset(train_data, tokenizer)  # 创建训练集数据集对象
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # 设置批大小
        collate_fn=customized_collate_fn,  # 设置collate函数
        shuffle=True,  # 打乱数据顺序
        drop_last=True,  # 如果数据量不是batch_size的整数倍，丢弃最后一个不完整的batch
        num_workers=num_workers  # 设置数据加载时的子进程数
    )

    # 创建验证集DataLoader
    val_dataset = CustomDataset(val_data, tokenizer)  # 创建验证集数据集对象
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # 设置批大小
        collate_fn=customized_collate_fn,  # 设置collate函数
        shuffle=False,  # 验证集不打乱
        drop_last=False,  # 不丢弃最后一个batch
        num_workers=num_workers  # 设置数据加载时的子进程数
    )

    #######################################
    # Load pretrained model
    #######################################

    # 基本的模型配置
    BASE_CONFIG = {
        "vocab_size": 50257,  # 词汇表大小
        "context_length": 1024,  # 上下文长度
        "drop_rate": 0.0,  # Dropout率
        "qkv_bias": True  # 是否使用QKV偏置
    }

    # 不同大小的GPT2模型配置
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},  # GPT2-small配置
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},  # GPT2-medium配置
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},  # GPT2-large配置
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},  # GPT2-xl配置
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"  # 选择模型为GPT2-medium

    # 更新基础配置，包含所选模型的配置
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # 提取模型的大小（如355M）
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    # 下载并加载预训练的GPT2模型
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    # 创建GPT模型并加载权重
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)  # 加载模型权重
    model.eval()  # 设置模型为评估模式
    model.to(device)  # 将模型转移到设备（GPU或CPU）

    print("Loaded model:", CHOOSE_MODEL)  # 打印加载的模型名称
    print(50 * "-")  # 打印分割线

    # 如果启用了LoRA（Low-Rank Adaptation），则进行LoRA参数替换
    if lora:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 计算训练参数的总数
        print(f"Total trainable parameters before: {total_params:,}")  # 打印训练参数总数

        # 将所有模型参数的`requires_grad`设为False，冻结所有参数
        for param in model.parameters():
            param.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 计算冻结后的训练参数总数
        print(f"Total trainable parameters after: {total_params:,}")  # 打印冻结后训练参数的总数

        # 用LoRA替换线性层
        replace_linear_with_lora(model, rank=16, alpha=16)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 计算LoRA参数总数
        print(f"Total trainable LoRA parameters: {total_params:,}")  # 打印LoRA参数的总数

        model.to(device)  # 将LoRA模型转移到设备（GPU或CPU）

    #######################################
    # Finetuning the model
    #######################################
    print("Initial losses")  # 打印初始的损失值

    with torch.no_grad():  # 在不计算梯度的情况下进行损失计算
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)  # 计算训练集的损失（前5个批次）
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)  # 计算验证集的损失（前5个批次）

    print("   Training loss:", train_loss)  # 打印训练集损失
    print("   Validation loss:", val_loss)  # 打印验证集损失

    start_time = time.time()  # 记录训练开始的时间

    num_epochs = 2  # 设置训练的轮数
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)  # 设置AdamW优化器，学习率为0.00005，权重衰减为0.1

    torch.manual_seed(123)  # 设置随机种子以保证结果的可重复性

    # 根据不同的参数选择格式化输入函数
    start_context = format_input_phi(val_data[0]) if phi3_prompt else format_input(val_data[0])

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,  # 设置训练轮数，评估频率等
        start_context=start_context, tokenizer=tokenizer
    )

    end_time = time.time()  # 记录训练结束的时间
    execution_time_minutes = (end_time - start_time) / 60  # 计算训练用时（分钟）
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")  # 打印训练总时长

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))  # 生成训练轮数对应的张量

    # 根据不同的标志条件修改绘图文件名
    plot_name = "loss-plot.pdf"
    if mask_instructions:
        plot_name = plot_name.replace(".pdf", "-mask-instructions.pdf")
    if alpaca52k:
        plot_name = plot_name.replace(".pdf", "-alpaca52k.pdf")
    if phi3_prompt:
        plot_name = plot_name.replace(".pdf", "-phi3-prompt.pdf")
    if lora:
        plot_name = plot_name.replace(".pdf", "-lora.pdf")
    if not any([mask_instructions, alpaca52k, phi3_prompt, lora]):
        plot_name = plot_name.replace(".pdf", "-baseline.pdf")

    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, plot_name)  # 绘制损失曲线并保存为文件
    print(50 * "-")  # 打印分割线

    #######################################
    # Saving results
    #######################################
    print("Generating responses")  # 打印生成模型响应的提示信息

    # 遍历测试数据生成模型输出
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):  # tqdm用于显示进度条

        input_text = format_input_phi(entry) if phi3_prompt else format_input(entry)  # 根据选择的提示格式生成输入文本

        # 使用模型生成token
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),  # 将输入文本转换为token IDs
            max_new_tokens=256,  # 限制生成的最大token数为256
            context_size=BASE_CONFIG["context_length"],  # 上下文窗口大小
            eos_id=50256  # 结束符ID
        )

        generated_text = token_ids_to_text(token_ids, tokenizer)  # 将生成的token转换为文本

        # 根据提示格式处理生成的文本
        if phi3_prompt:
            response_text = generated_text[len(input_text):].replace("<|assistant|>:",
                                                                     "").strip()  # 对phi3_prompt格式的响应进行处理
        else:
            response_text = generated_text[len(input_text):].replace("### Response:", "").strip()  # 对其他格式的响应进行处理

        test_data[i]["model_response"] = response_text  # 将生成的响应保存到测试数据中

    # 根据不同的标志条件修改保存的文件名
    test_data_path = "instruction-data-with-response.json"
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"

    if mask_instructions:
        test_data_path = test_data_path.replace(".json", "-mask-instructions.json")
        file_name = file_name.replace(".pth", "-mask-instructions.pth")
    if alpaca52k:
        test_data_path = test_data_path.replace(".json", "-alpaca52k.json")
        file_name = file_name.replace(".pth", "-alpaca52k.pth")
    if phi3_prompt:
        test_data_path = test_data_path.replace(".json", "-phi3-prompt.json")
        file_name = file_name.replace(".pth", "-phi3-prompt.pth")
    if lora:
        test_data_path = test_data_path.replace(".json", "-lora.json")
        file_name = file_name.replace(".pth", "-lora.pth")
    if not any([mask_instructions, alpaca52k, phi3_prompt, lora]):
        test_data_path = test_data_path.replace(".json", "-baseline.json")
        file_name = file_name.replace(".pth", "-baseline.pth")

    with open(test_data_path, "w") as file:  # 保存测试数据及模型响应
        json.dump(test_data, file, indent=4)  # "indent"参数用于美化输出
    print(f"Responses saved as {test_data_path}")  # 打印响应保存路径

    torch.save(model.state_dict(), file_name)  # 保存模型的状态字典
    print(f"Model saved as {file_name}")  # 打印模型保存路径

    if __name__ == "__main__":  # 主函数入口

        import argparse  # 导入argparse模块用于命令行参数解析

        parser = argparse.ArgumentParser(
            description="Instruction finetune a GPT model"  # 命令行工具描述
        )
        options = {"baseline", "mask_instructions", "alpaca_52k", "phi3_prompt", "lora"}  # 预定义的选项
        parser.add_argument(
            "--exercise_solution",
            type=str,
            default="last_block",  # 默认值设置为 "last_block"
            help=(
                f"Which experiment to run. Options: {options}."  # 说明参数的功能
            )
        )
        args = parser.parse_args()  # 解析命令行参数

        # 根据命令行输入的参数值选择执行的实验
        if args.exercise_solution == "baseline":
            main()  # 执行基础模型训练
        elif args.exercise_solution == "mask_instructions":
            main(mask_instructions=True)  # 执行带mask指令的训练
        elif args.exercise_solution == "alpaca_52k":
            main(alpaca52k=True)  # 执行alpaca52k训练
        elif args.exercise_solution == "phi3_prompt":
            main(phi3_prompt=True)  # 执行phi3_prompt训练
        elif args.exercise_solution == "lora":
            main(lora=True)  # 执行lora训练
        else:
            raise ValueError(
                f"{args.exercise_solution} is not a valid --args.exercise_solution option. Options: {options}")  # 如果输入无效，抛出错误
