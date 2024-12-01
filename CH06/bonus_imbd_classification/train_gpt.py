# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# 来源于 "Build a Large Language Model From Scratch"
# 书籍链接: https://www.manning.com/books/build-a-large-language-model-from-scratch
# 源代码链接: https://github.com/rasbt/LLMs-from-scratch

# 导入必要的库
import argparse  # 用于解析命令行参数
from pathlib import Path  # 提供方便的文件路径操作
import time  # 计算时间消耗

import pandas as pd  # 用于处理数据集
import tiktoken  # 一个用于 GPT 模型的高效文本编码库
import torch  # PyTorch，用于深度学习
from torch.utils.data import DataLoader  # 用于数据加载
from torch.utils.data import Dataset  # 用于创建自定义数据集

# 导入自定义模块
from ch05.main_chapter_code.gpt_download import download_and_load_gpt2  # 下载并加载 GPT-2 权重
from ch04.main_chapter_code.ch04 import GPTModel  # GPT 模型的定义
from ch05.main_chapter_code.ch05 import load_weights_into_gpt  # 加载权重到 GPT 模型


# 定义自定义数据集类，用于加载 IMDB 数据集
class IMDBDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        """
        初始化 IMDB 数据集
        :param csv_file: 数据集的 CSV 文件路径
        :param tokenizer: 文本编码器，用于将文本转换为令牌序列
        :param max_length: 最大序列长度，如果为 None，则自动计算最长序列长度
        :param pad_token_id: 用于填充序列的填充令牌 ID，默认为 GPT 的 <|endoftext|> 令牌
        """
        self.data = pd.read_csv(csv_file)  # 读取 CSV 数据
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)
        # 如果未提供最大长度，则通过 _longest_encoded_length 方法计算最长序列长度

        # 预先对文本进行编码
        self.encoded_texts = [
            tokenizer.encode(text)[:self.max_length]  # 对每个文本进行编码并截断到最大长度
            for text in self.data["text"]
        ]
        # 填充序列到固定长度
        self.encoded_texts = [
            et + [pad_token_id] * (self.max_length - len(et))  # 用填充令牌填充序列
            for et in self.encoded_texts
        ]

    def __getitem__(self, index):
        """
        获取数据集中的一个样本
        :param index: 样本索引
        :return: 一个元组，包含编码后的文本和标签
        """
        encoded = self.encoded_texts[index]  # 获取编码后的文本
        label = self.data.iloc[index]["label"]  # 获取对应的标签
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)
        # 将数据转换为 PyTorch 的张量格式

    def __len__(self):
        """
        获取数据集的总样本数
        :return: 数据集的长度
        """
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        """
        计算数据集中最长的编码序列长度
        :param tokenizer: 文本编码器
        :return: 最长序列长度
        """
        max_length = 0  # 初始化最大长度为 0
        for text in self.data["text"]:
            encoded_length = len(tokenizer.encode(text))  # 对文本进行编码并计算其长度
            if encoded_length > max_length:
                max_length = encoded_length  # 更新最大长度
        return max_length  # 返回最大长度


# 实例化模型
def instantiate_model(choose_model, load_weights):
    """
    初始化 GPT 模型
    :param choose_model: 要选择的模型名称，例如 "gpt2-small (124M)"
    :param load_weights: 是否加载预训练权重
    :return: 实例化的 GPT 模型
    """
    # 基础配置
    BASE_CONFIG = {
        "vocab_size": 50257,     # 词汇表大小
        "context_length": 1024,  # 上下文长度
        "drop_rate": 0.0,        # Dropout 率
        "qkv_bias": True         # Query-key-value 的偏置
    }

    # 不同模型的参数配置
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # 更新基础配置为选择的模型配置
    BASE_CONFIG.update(model_configs[choose_model])

    if not load_weights:
        torch.manual_seed(123)  # 设置随机种子以确保结果可复现
    model = GPTModel(BASE_CONFIG)  # 实例化模型

    if load_weights:
        # 解析模型大小并加载预训练权重
        model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")  # 提取模型大小，例如 "124M"
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
        # 下载并加载 GPT-2 权重
        load_weights_into_gpt(model, params)  # 将权重加载到模型中

    model.eval()  # 设置模型为评估模式
    return model


# 计算批量数据的损失
def calc_loss_batch(input_batch, target_batch, model, device,
                    trainable_token_pos=-1, average_embeddings=False):
    """
    计算单个批量的损失
    :param input_batch: 输入的批量数据
    :param target_batch: 目标标签的批量数据
    :param model: GPT 模型
    :param device: 运行设备（CPU 或 GPU）
    :param trainable_token_pos: 可训练的令牌位置，默认为 -1（最后一个令牌）
    :param average_embeddings: 是否对嵌入进行平均
    :return: 当前批量的损失值
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将数据迁移到指定设备

    model_output = model(input_batch)  # 模型前向传播，得到输出
    if average_embeddings:
        # 如果选择对嵌入进行平均，则对序列维度进行平均
        logits = model_output.mean(dim=1)
    else:
        # 否则，选择指定位置的嵌入
        logits = model_output[:, trainable_token_pos, :]

    loss = torch.nn.functional.cross_entropy(logits, target_batch)  # 使用交叉熵计算损失
    return loss
# 定义用于计算整个数据加载器的损失的函数
def calc_loss_loader(data_loader, model, device,
                     num_batches=None, trainable_token_pos=-1,
                     average_embeddings=False):
    """
    计算数据加载器的平均损失
    :param data_loader: 数据加载器
    :param model: 模型
    :param device: 使用的设备（CPU 或 GPU）
    :param num_batches: 要计算的批次数量
    :param trainable_token_pos: 可训练令牌的位置，默认为 -1（最后一个令牌）
    :param average_embeddings: 是否对嵌入进行平均
    :return: 数据加载器的平均损失
    """
    total_loss = 0.  # 初始化总损失
    if len(data_loader) == 0:  # 如果数据加载器为空
        return float("nan")  # 返回 NaN
    elif num_batches is None:  # 如果未指定批次数量
        num_batches = len(data_loader)  # 使用数据加载器中的总批次数
    else:
        # 如果指定的批次数量超过数据加载器中的批次数，则取较小值
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:  # 只处理指定数量的批次
            # 计算每个批次的损失并累加
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
            )
            total_loss += loss.item()  # 累加损失值
        else:
            break  # 如果达到指定批次数量，则退出循环
    return total_loss / num_batches  # 返回平均损失


# 定义计算整个数据加载器准确率的函数
@torch.no_grad()  # 禁用梯度计算以提高效率
def calc_accuracy_loader(data_loader, model, device,
                         num_batches=None, trainable_token_pos=-1,
                         average_embeddings=False):
    """
    计算数据加载器的准确率
    :param data_loader: 数据加载器
    :param model: 模型
    :param device: 使用的设备（CPU 或 GPU）
    :param num_batches: 要计算的批次数量
    :param trainable_token_pos: 可训练令牌的位置，默认为 -1（最后一个令牌）
    :param average_embeddings: 是否对嵌入进行平均
    :return: 数据加载器的准确率
    """
    model.eval()  # 设置模型为评估模式
    correct_predictions, num_examples = 0, 0  # 初始化正确预测数和总样本数

    if num_batches is None:
        num_batches = len(data_loader)  # 如果未指定批次数量，则使用加载器的总批次数
    else:
        num_batches = min(num_batches, len(data_loader))  # 取指定批次与加载器总批次的最小值
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:  # 只处理指定数量的批次
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将数据移至指定设备

            model_output = model(input_batch)  # 模型前向传播
            if average_embeddings:
                # 如果需要平均嵌入，则对序列维度（dim=1）取平均
                logits = model_output.mean(dim=1)
            else:
                # 否则选择指定位置的嵌入
                logits = model_output[:, trainable_token_pos, :]

            predicted_labels = torch.argmax(logits, dim=-1)  # 获取预测标签

            num_examples += predicted_labels.shape[0]  # 更新总样本数
            correct_predictions += (predicted_labels == target_batch).sum().item()
            # 累加正确预测数
        else:
            break  # 达到指定批次数退出循环
    return correct_predictions / num_examples  # 返回准确率


# 定义模型评估函数
def evaluate_model(model, train_loader, val_loader, device, eval_iter,
                   trainable_token_pos=-1, average_embeddings=False):
    """
    评估模型在训练集和验证集上的损失
    :param model: 模型
    :param train_loader: 训练数据加载器
    :param val_loader: 验证数据加载器
    :param device: 设备（CPU 或 GPU）
    :param eval_iter: 要评估的迭代次数
    :param trainable_token_pos: 可训练令牌位置
    :param average_embeddings: 是否对嵌入进行平均
    :return: 训练损失和验证损失
    """
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        # 计算训练和验证集的平均损失
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
    model.train()  # 恢复模型为训练模式
    return train_loss, val_loss  # 返回训练和验证损失


# 定义简单的分类器训练函数
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None, trainable_token_pos=-1,
                            average_embeddings=False):
    """
    使用简单分类器训练 GPT 模型
    :param model: GPT 模型
    :param train_loader: 训练数据加载器
    :param val_loader: 验证数据加载器
    :param optimizer: 优化器
    :param device: 设备（CPU 或 GPU）
    :param num_epochs: 训练的总轮数
    :param eval_freq: 评估的频率（每多少步评估一次）
    :param eval_iter: 每次评估的迭代次数
    :param max_steps: 最大训练步数
    :param trainable_token_pos: 可训练令牌位置
    :param average_embeddings: 是否对嵌入进行平均
    """
    # 初始化用于跟踪损失和准确率的列表
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1  # 初始化全局变量

    # 主训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 重置梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device,
                                   trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings)
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 使用梯度更新模型权重
            examples_seen += input_batch.shape[0]  # 跟踪已处理的样本数
            global_step += 1  # 更新全局步数

            # 可选的评估步骤
            if global_step % eval_freq == 0:
                # 评估训练和验证损失
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter,
                    trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
                )
                train_losses.append(train_loss)  # 保存训练损失
                val_losses.append(val_loss)  # 保存验证损失
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")  # 打印评估结果

            if max_steps is not None and global_step > max_steps:
                break  # 如果达到最大步数，退出训练
                # 新增功能：在每个 epoch 结束后计算训练和验证的准确率
                train_accuracy = calc_accuracy_loader(
                    train_loader, model, device, num_batches=eval_iter,
                    trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
                )
                # 计算训练集的准确率

                val_accuracy = calc_accuracy_loader(
                    val_loader, model, device, num_batches=eval_iter,
                    trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
                )
                # 计算验证集的准确率

                print(f"Training accuracy: {train_accuracy * 100:.2f}% | ", end="")
                # 打印训练准确率，保留两位小数

                print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
                # 打印验证准确率，保留两位小数

                train_accs.append(train_accuracy)
                val_accs.append(val_accuracy)
                # 将训练和验证的准确率分别存储到对应的列表中

                if max_steps is not None and global_step > max_steps:
                    break
                # 如果全局训练步数超过最大步数，则退出循环

            return train_losses, val_losses, train_accs, val_accs, examples_seen
            # 返回训练和验证的损失、准确率，以及已处理的样本数

        # 主程序入口
        if __name__ == "__main__":

            # 定义命令行参数解析器
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "--model_size",
                type=str,
                default="gpt2-small (124M)",
                help=(
                    "选择要使用的 GPT 模型。"
                    "可选值：'gpt2-small (124M)', 'gpt2-medium (355M)',"
                    "'gpt2-large (774M)', 'gpt2-xl (1558M)'。"
                )
            )
            # 参数 `--model_size` 指定 GPT 模型大小，默认是 "gpt2-small (124M)"

            parser.add_argument(
                "--weights",
                type=str,
                default="pretrained",
                help=(
                    "选择使用 'pretrained'（预训练权重）或 'random'（随机初始化权重）。"
                )
            )
            # 参数 `--weights` 指定模型的初始化方式，默认是预训练权重

            parser.add_argument(
                "--trainable_layers",
                type=str,
                default="last_block",
                help=(
                    "选择要训练的层。"
                    "可选值：'all'（全部层）、'last_block'（最后一个块）、'last_layer'（最后一层）。"
                )
            )
            # 参数 `--trainable_layers` 指定训练哪些层，默认是最后一个块

            parser.add_argument(
                "--trainable_token_pos",
                type=str,
                default="last",
                help=(
                    "选择要训练的令牌位置。"
                    "可选值：'first'（第一个令牌）、'last'（最后一个令牌）。"
                )
            )
            # 参数 `--trainable_token_pos` 指定要训练的令牌位置，默认是最后一个令牌

            parser.add_argument(
                "--average_embeddings",
                action='store_true',
                default=False,
                help=(
                    "是否对所有令牌的输出嵌入取平均值，"
                    "而不是只使用由 `--trainable_token_pos` 指定位置的嵌入。"
                )
            )
            # 参数 `--average_embeddings` 指定是否平均嵌入，默认不启用（使用单个位置的嵌入）

            parser.add_argument(
                "--context_length",
                type=str,
                default="256",
                help=(
                    "指定数据输入的上下文长度。"
                    "可选值：'longest_training_example'（最长训练样本）、"
                    "'model_context_length'（模型支持的最大长度）或整数值。"
                )
            )
            # 参数 `--context_length` 指定输入数据的上下文长度，默认是 256

            parser.add_argument(
                "--num_epochs",
                type=int,
                default=1,
                help=(
                    "训练的轮数。"
                )
            )
            # 参数 `--num_epochs` 指定训练的轮数，默认是 1

            parser.add_argument(
                "--learning_rate",
                type=float,
                default=5e-5,
                help=(
                    "学习率。"
                )
            )
            # 参数 `--learning_rate` 指定学习率，默认是 5e-5

            args = parser.parse_args()
            # 解析命令行参数并存储到变量 `args` 中

            # 根据参数设置训练令牌的位置
            if args.trainable_token_pos == "first":
                args.trainable_token_pos = 0
                # 如果指定为 "first"，则位置设置为 0（第一个令牌）
            elif args.trainable_token_pos == "last":
                args.trainable_token_pos = -1
                # 如果指定为 "last"，则位置设置为 -1（最后一个令牌）
            else:
                raise ValueError("Invalid --trainable_token_pos argument")
                # 如果参数值无效，则抛出错误
    ###############################
    # Load model
    ###############################

    if args.weights == "pretrained":
        load_weights = True
    elif args.weights == "random":
        load_weights = False
    else:
        raise ValueError("Invalid --weights argument.")
    # 根据命令行参数选择是否加载预训练权重。如果是 "pretrained" 加载预训练权重，
    # 如果是 "random" 使用随机初始化。如果参数无效则抛出错误。

    model = instantiate_model(args.model_size, load_weights)
    # 根据选择的模型大小和权重类型初始化模型。

    for param in model.parameters():
        param.requires_grad = False
    # 冻结模型中的所有参数（不更新权重）

    # 根据模型大小设置输入特征的维度
    if args.model_size == "gpt2-small (124M)":
        in_features = 768
    elif args.model_size == "gpt2-medium (355M)":
        in_features = 1024
    elif args.model_size == "gpt2-large (774M)":
        in_features = 1280
    elif args.model_size == "gpt2-xl (1558M)":
        in_features = 1600
    else:
        raise ValueError("Invalid --model_size argument")
    # 根据所选模型大小设置输入层的特征维度。

    torch.manual_seed(123)
    model.out_head = torch.nn.Linear(in_features=in_features, out_features=2)
    # 设置随机种子，确保结果可复现，并为模型添加输出层（2 个分类输出）

    # 根据命令行参数选择要训练的层
    if args.trainable_layers == "last_layer":
        pass
    elif args.trainable_layers == "last_block":
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True
    elif args.trainable_layers == "all":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Invalid --trainable_layers argument.")
    # 根据选择的层类型，解冻相应层的参数。

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 检查是否有 GPU 可用，若有则使用 GPU 进行训练，否则使用 CPU。

    ###############################
    # Instantiate dataloaders
    ###############################

    base_path = Path(".")
    # 基础路径，假设数据文件位于当前目录

    tokenizer = tiktoken.get_encoding("gpt2")
    # 使用 GPT2 模型的 tokenizer 对输入文本进行编码

    train_dataset = None
    # 初始化训练数据集

    # 根据上下文长度参数确定最大长度
    if args.context_length == "model_context_length":
        max_length = model.pos_emb.weight.shape[0]
    elif args.context_length == "longest_training_example":
        train_dataset = IMDBDataset(base_path / "train.csv", max_length=None, tokenizer=tokenizer)
        max_length = train_dataset.max_length
    else:
        try:
            max_length = int(args.context_length)
        except ValueError:
            raise ValueError("Invalid --context_length argument")
    # 根据命令行输入参数设置上下文长度，若是模型上下文长度则获取模型的上下文长度，若是最长训练样本长度则从数据集中计算。

    if train_dataset is None:
        train_dataset = IMDBDataset(base_path / "train.csv", max_length=max_length, tokenizer=tokenizer)
    val_dataset = IMDBDataset(base_path / "validation.csv", max_length=max_length, tokenizer=tokenizer)
    test_dataset = IMDBDataset(base_path / "test.csv", max_length=max_length, tokenizer=tokenizer)
    # 加载训练、验证和测试数据集

    num_workers = 0
    batch_size = 8
    # 设置数据加载器的参数：工作线程数为 0（表示不使用多线程），批大小为 8

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    # 初始化训练数据加载器，打乱数据并在最后一个批次丢弃不足一个批次的数据

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    # 初始化验证数据加载器

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    # 初始化测试数据加载器

    ###############################
    # Train model
    ###############################

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    # 记录训练开始时间，设置随机种子，初始化优化器 AdamW，学习率为命令行参数传入的学习率，权重衰减为 0.1。

    # 开始训练模型
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs, eval_freq=50, eval_iter=20,
        max_steps=None, trainable_token_pos=args.trainable_token_pos,
        average_embeddings=args.average_embeddings
    )
    # 使用 `train_classifier_simple` 函数开始训练，传入模型、训练/验证加载器、优化器、设备和相关训练参数。

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    # 计算训练时间并打印训练完成时间

    ###############################
    # Evaluate model
    ###############################

    print("\nEvaluating on the full datasets ...\n")
    # 在完整的数据集上进行评估

    # 计算训练集、验证集和测试集的准确率
    train_accuracy = calc_accuracy_loader(
        train_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )
    val_accuracy = calc_accuracy_loader(
        val_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )
    test_accuracy = calc_accuracy_loader(
        test_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )

    # 打印训练、验证和测试准确率
    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
