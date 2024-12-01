# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# 本文件基于《Build a Large Language Model From Scratch》一书的代码
# 图书链接：https://www.manning.com/books/build-a-large-language-model-from-scratch
# 源代码仓库：https://github.com/rasbt/LLMs-from-scratch
#
# 一个基于第7章的最小指令微调代码示例

from functools import partial  # 从 functools 导入 partial 函数，用于创建部分函数
from importlib.metadata import version  # 用于获取库的版本信息
import json  # 导入 JSON 模块，用于处理 JSON 数据
import os  # 操作系统相关功能，如文件路径
import re  # 正则表达式模块
import time  # 时间模块，用于测量和格式化时间
import urllib  # 用于处理 URL 相关功能

import matplotlib.pyplot as plt  # 导入 Matplotlib 绘图工具
import tiktoken  # 用于分词和编码
import torch  # 深度学习框架 PyTorch
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载工具
from tqdm import tqdm  # 导入 tqdm 用于显示进度条

# 从本地的相关文件导入函数和类
from ch06.main_chapter_code.gpt_download import download_and_load_gpt2  # 下载和加载 GPT2 模型
from previous_chapters import (
    calc_loss_loader,  # 计算数据加载器中的损失
    generate,  # 生成文本函数
    GPTModel,  # 定义 GPT 模型的类
    load_weights_into_gpt,  # 加载预训练权重到 GPT 模型
    text_to_token_ids,  # 将文本转化为 Token ID
    train_model_simple,  # 简单训练函数
    token_ids_to_text  # 将 Token ID 转化为文本
)


# 定义指令数据集类，继承 PyTorch 的 Dataset 类
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        """
        初始化数据集对象。

        参数：
        - data: 包含指令、输入和输出的字典列表
        - tokenizer: 用于编码文本的分词器
        """
        self.data = data  # 保存数据

        # 预先对文本进行编码
        self.encoded_texts = []  # 用于存储编码后的文本
        for entry in data:  # 遍历数据集中的每一条数据
            # 格式化指令和输入文本
            instruction_plus_input = format_input(entry)
            # 格式化响应文本
            response_text = f"\n\n### Response:\n{entry['output']}"
            # 合并指令、输入和响应文本
            full_text = instruction_plus_input + response_text
            # 使用分词器对完整文本进行编码，并存储到列表中
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        """
        根据索引返回数据。

        参数：
        - index: 数据索引
        返回：
        - 编码后的文本数据
        """
        return self.encoded_texts[index]

    def __len__(self):
        """
        返回数据集的大小。

        返回：
        - 数据集的长度
        """
        return len(self.data)
def custom_collate_fn(
    batch,  # 批次数据
    pad_token_id=50256,  # 用于填充的特殊 token ID (<|endoftext|>)
    ignore_index=-100,  # 忽略索引的值，用于计算损失时跳过
    allowed_max_length=None,  # 允许的最大序列长度（可选）
    device="cpu"  # 目标设备（默认为 CPU）
):
    # 找到批次中最长的序列长度，并加 1（为末尾添加的特殊 token）
    batch_max_length = max(len(item)+1 for item in batch)

    # 初始化输入和目标列表
    inputs_lst, targets_lst = [], []

    for item in batch:  # 遍历批次中的每个样本
        new_item = item.copy()  # 复制样本数据
        # 在序列末尾添加一个 <|endoftext|> token
        new_item += [pad_token_id]
        # 将序列填充到批次中最长的长度
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # 输入数据去掉最后一个 token
        targets = torch.tensor(padded[1:])  # 目标数据右移一位

        # 新功能：将目标序列中除第一个填充 token 之外的其他填充值替换为 ignore_index
        mask = targets == pad_token_id  # 找到目标中的填充值位置
        indices = torch.nonzero(mask).squeeze()  # 获取填充值索引
        if indices.numel() > 1:  # 如果有多个填充值
            targets[indices[1:]] = ignore_index  # 将除第一个之外的填充值替换为 ignore_index

        # 新功能：可选地将序列截断到最大长度
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]  # 截断输入序列
            targets = targets[:allowed_max_length]  # 截断目标序列

        # 将处理好的输入和目标添加到列表中
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 将输入和目标列表转换为张量并转移到目标设备上
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    # 返回输入和目标张量
    return inputs_tensor, targets_tensor


def download_and_load_file(file_path, url):
    """
    下载文件并加载数据。

    参数：
    - file_path: 文件保存路径
    - url: 文件下载的 URL
    返回：
    - 加载的 JSON 数据
    """
    if not os.path.exists(file_path):  # 检查文件是否存在
        # 如果不存在，从 URL 下载文件内容
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")  # 读取并解码数据
        # 将下载的数据写入文件
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        # 如果文件存在，直接读取文件内容
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    # 加载文件内容为 JSON 格式
    with open(file_path, "r") as file:
        data = json.load(file)

    # 返回解析后的数据
    return data


def format_input(entry):
    """
    格式化输入数据，包括指令和输入内容。

    参数：
    - entry: 包含指令和输入的字典
    返回：
    - 格式化后的字符串
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."  # 提供任务描述的指令
        f"\n\n### Instruction:\n{entry['instruction']}"  # 添加指令内容
    )

    # 如果有输入数据，则添加输入部分；否则为空
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    # 返回格式化后的指令和输入文本
    return instruction_text + input_text


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    绘制训练损失和验证损失图。

    参数：
    - epochs_seen: 经过的训练轮数
    - tokens_seen: 处理的 Token 数量
    - train_losses: 训练损失值列表
    - val_losses: 验证损失值列表
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))  # 创建一个图形和主轴，指定图大小

    # 绘制训练和验证损失曲线
    ax1.plot(epochs_seen, train_losses, label="Training loss")  # 绘制训练损失曲线
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")  # 绘制验证损失曲线
    ax1.set_xlabel("Epochs")  # 设置 X 轴标签（训练轮数）
    ax1.set_ylabel("Loss")  # 设置 Y 轴标签（损失值）
    ax1.legend(loc="upper right")  # 添加图例，放置在右上角

    # 创建第二个 X 轴，用于显示处理的 Token 数量
    ax2 = ax1.twiny()  # 创建一个共享 Y 轴的双 X 轴
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 绘制隐形曲线，用于对齐刻度
    ax2.set_xlabel("Tokens seen")  # 设置 X 轴标签（Token 数量）

    fig.tight_layout()  # 调整图形布局，使所有元素不重叠
    plot_name = "loss-plot-standalone.pdf"  # 定义保存图形的文件名
    print(f"Plot saved as {plot_name}")  # 打印保存路径
    plt.savefig(plot_name)  # 保存图形为 PDF 格式
    # plt.show()  # 可选：显示图形
def main(test_mode=False):  # 主函数，test_mode 参数用于控制是否进入测试模式
    #######################################
    # 打印依赖包版本
    #######################################
    print()
    pkgs = [
        "matplotlib",  # 绘图库
        "tiktoken",    # 分词器
        "torch",       # 深度学习库
        "tqdm",        # 进度条工具
        "tensorflow",  # 用于加载 OpenAI 的预训练权重
    ]
    for p in pkgs:  # 遍历包名
        print(f"{p} version: {version(p)}")  # 打印每个包的版本号
    print(50 * "-")  # 分隔线

    #######################################
    # 下载和准备数据集
    #######################################
    file_path = "instruction-data.json"  # 数据集文件路径
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    # 下载并加载 JSON 数据
    data = download_and_load_file(file_path, url)

    train_portion = int(len(data) * 0.85)  # 数据集中 85% 用于训练
    test_portion = int(len(data) * 0.1)    # 数据集中 10% 用于测试

    # 分割数据集
    train_data = data[:train_portion]  # 训练数据
    test_data = data[train_portion:train_portion + test_portion]  # 测试数据
    val_data = data[train_portion + test_portion:]  # 验证数据

    # 如果处于测试模式，仅取小部分数据
    if test_mode:
        train_data = train_data[:10]
        val_data = val_data[:10]
        test_data = test_data[:10]

    print("Training set length:", len(train_data))  # 打印训练集大小
    print("Validation set length:", len(val_data))  # 打印验证集大小
    print("Test set length:", len(test_data))  # 打印测试集大小
    print(50 * "-")  # 分隔线

    # 初始化分词器和设备
    tokenizer = tiktoken.get_encoding("gpt2")  # 使用 GPT-2 分词器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有 GPU
    print("Device:", device)  # 打印使用的设备
    print(50 * "-")  # 分隔线

    # 自定义的批次处理函数，指定设备和最大长度
    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

    num_workers = 0  # 数据加载线程数
    batch_size = 8  # 批次大小

    torch.manual_seed(123)  # 设置随机种子以保证结果可复现

    # 创建训练数据集和加载器
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,  # 使用自定义批次处理函数
        shuffle=True,  # 打乱数据
        drop_last=True,  # 如果最后一个批次大小不足，则丢弃
        num_workers=num_workers  # 加载线程数
    )

    # 创建验证数据集和加载器
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,  # 使用自定义批次处理函数
        shuffle=False,  # 不打乱数据
        drop_last=False,  # 保留最后一个批次
        num_workers=num_workers  # 加载线程数
    )

    #######################################
    # 加载预训练模型
    #######################################

    # 测试模式下使用小模型
    if test_mode:
        BASE_CONFIG = {  # 基础配置
            "vocab_size": 50257,     # 词汇表大小
            "context_length": 120,  # 上下文长度
            "drop_rate": 0.0,       # Dropout 率
            "qkv_bias": False,      # 是否添加 QKV 偏置
            "emb_dim": 12,          # 嵌入维度
            "n_layers": 1,          # 层数
            "n_heads": 2            # 注意力头数
        }
        model = GPTModel(BASE_CONFIG)  # 初始化小模型
        model.eval()  # 设置模型为评估模式
        device = "cpu"  # 测试模式仅使用 CPU
        CHOOSE_MODEL = "Small test model"  # 选择的模型名称

    # 使用主代码中配置加载模型
    else:
        BASE_CONFIG = {  # GPT 模型基础配置
            "vocab_size": 50257,     # 词汇表大小
            "context_length": 1024,  # 上下文长度
            "drop_rate": 0.0,        # Dropout 率
            "qkv_bias": True         # 是否添加 QKV 偏置
        }

        # 定义不同 GPT-2 模型的配置
        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        CHOOSE_MODEL = "gpt2-medium (355M)"  # 选择 GPT-2 中等模型

        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])  # 更新配置为所选模型的参数

        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")  # 提取模型大小
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")  # 下载预训练权重

        model = GPTModel(BASE_CONFIG)  # 初始化模型
        load_weights_into_gpt(model, params)  # 加载预训练权重到模型
        model.eval()  # 设置模型为评估模式
        model.to(device)  # 将模型移动到指定设备

    print("Loaded model:", CHOOSE_MODEL)  # 打印加载的模型
    print(50 * "-")  # 分隔线
    #######################################
    # 微调模型
    #######################################
    print("Initial losses")  # 打印初始损失
    with torch.no_grad():  # 在不计算梯度的情况下评估损失
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)  # 计算训练集损失
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)  # 计算验证集损失

    print("   Training loss:", train_loss)  # 打印训练集损失
    print("   Validation loss:", val_loss)  # 打印验证集损失

    start_time = time.time()  # 记录训练开始时间
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)  # 使用 AdamW 优化器

    num_epochs = 2  # 设置训练轮数

    torch.manual_seed(123)  # 设置随机种子以保证结果可复现
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,  # 调用训练函数
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,  # 每隔 5 个批次评估一次
        start_context=format_input(val_data[0]), tokenizer=tokenizer  # 设置初始上下文
    )

    end_time = time.time()  # 记录训练结束时间
    execution_time_minutes = (end_time - start_time) / 60  # 计算训练时间（分钟）
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")  # 打印训练完成时间

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))  # 生成训练轮数的张量
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)  # 绘制损失曲线
    print(50 * "-")  # 分隔线

    #######################################
    # 保存结果
    #######################################
    print("Generating responses")  # 打印生成响应的消息
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):  # 遍历测试数据集

        input_text = format_input(entry)  # 格式化输入文本

        token_ids = generate(  # 使用模型生成预测结果
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,  # 最大生成长度
            context_size=BASE_CONFIG["context_length"],  # 上下文大小
            eos_id=50256  # EOS (结束标志) token ID
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)  # 将生成的 token IDs 转换为文本
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()  # 获取模型响应

        test_data[i]["model_response"] = response_text  # 保存生成的响应到测试数据

    test_data_path = "instruction-data-with-response-standalone.json"  # 设置保存文件路径
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)  # 使用 pretty-print 格式保存响应数据
    print(f"Responses saved as {test_data_path}")  # 打印响应数据保存路径

    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft-standalone.pth"  # 设置保存模型的文件名
    torch.save(model.state_dict(), file_name)  # 保存模型参数
    print(f"Model saved as {file_name}")  # 打印保存的模型文件名


if __name__ == "__main__":  # 如果是主程序，则执行以下代码

    import argparse  # 导入 argparse 用于命令行参数解析

    parser = argparse.ArgumentParser(
        description="Finetune a GPT model for classification"  # 设置命令行参数说明
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",  # 添加一个 test_mode 参数，默认为 False
        help=("This flag runs the model in test mode for internal testing purposes. "
              "Otherwise, it runs the model as it is used in the chapter (recommended).")
    )
    args = parser.parse_args()  # 解析命令行参数

    main(args.test_mode)  # 调用主函数，传入 test_mode 参数
