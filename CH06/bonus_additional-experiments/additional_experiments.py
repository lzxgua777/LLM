# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
import argparse
import math
import os
from pathlib import Path
import time
import urllib.request
import zipfile

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from ch06.main_chapter_code.gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt


# 定义 LoRA 层（低秩适应层）
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        # 初始化 LoRA 层的参数 A 和 B
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))  # A 是输入维度与秩的矩阵
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # 使用 Kaiming 初始化
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))  # B 是秩与输出维度的矩阵
        self.alpha = alpha  # alpha 是一个超参数，用于调整 LoRA 层的影响力

    def forward(self, x):
        # 在前向传播中，计算 x @ A @ B 的结果，并乘以 alpha
        x = self.alpha * (x @ self.A @ self.B)  # 低秩适应计算
        return x


# 这是 LoRA 代码的另一种实现形式，等价于 LinearWithLoRA
class LinearWithLoRAMerged(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear  # 线性层
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )  # 初始化 LoRA 层

    def forward(self, x):
        # 将 LoRA 层的 A 和 B 相乘，然后与线性层的权重相加，形成新的加权矩阵
        lora = self.lora.A @ self.lora.B  # 计算低秩矩阵 A 和 B 的乘积
        combined_weight = self.linear.weight + self.lora.alpha * lora.T  # 加权矩阵
        return torch.nn.functional.linear(x, combined_weight, self.linear.bias)  # 使用新的加权矩阵进行前向传播


# 自定义数据集类，用于处理短信垃圾邮件数据集
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256, no_padding=False):
        # 读取 CSV 文件，加载数据
        self.data = pd.read_csv(csv_file)
        # 设置最大长度，若未提供，则使用数据集中的最大文本长度
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)

        # 预先对文本进行分词并编码
        self.encoded_texts = [
            tokenizer.encode(text)[:self.max_length]  # 对文本进行编码，限制最大长度
            for text in self.data["Text"]
        ]

        if not no_padding:
            # 若没有禁用填充，使用 pad_token_id 对文本进行填充，确保所有序列的长度一致
            self.encoded_texts = [
                et + [pad_token_id] * (self.max_length - len(et))  # 填充序列
                for et in self.encoded_texts
            ]

    def __getitem__(self, index):
        # 根据索引返回一个编码后的文本和其对应的标签
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        # 返回数据集的大小
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        # 计算数据集中最长的文本的编码长度
        max_length = 0
        for text in self.data["Text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


# 下载并解压数据集
def download_and_unzip(url, zip_path, extract_to, new_file_path):
    # 如果文件已存在，则跳过下载和解压步骤
    if new_file_path.exists():
        print(f"{new_file_path} 已经存在，跳过下载和解压。")
        return

    # 下载文件
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())  # 将下载的内容写入本地文件

    # 解压文件
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)  # 解压到指定目录

    # 重命名文件为更具描述性的名称
    original_file = Path(extract_to) / "SMSSpamCollection"
    os.rename(original_file, new_file_path)  # 重命名文件
    print(f"文件已下载并保存为 {new_file_path}")


# 用于拆分数据集为训练集、验证集和测试集的函数
def random_split(df, train_frac, validation_frac):
    # 打乱整个 DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 计算数据拆分的结束位置
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # 拆分数据集
    train_df = df[:train_end]  # 训练集
    validation_df = df[train_end:validation_end]  # 验证集
    test_df = df[validation_end:]  # 测试集

    return train_df, validation_df, test_df


# 创建并保存 CSV 数据集
def create_dataset_csvs(new_file_path):
    df = pd.read_csv(new_file_path, sep="\t", header=None, names=["Label", "Text"])

    # 创建平衡的数据集，确保 spam 和 ham 数量相同
    n_spam = df[df["Label"] == "spam"].shape[0]  # 获取 spam 类别的样本数量
    ham_sampled = df[df["Label"] == "ham"].sample(n_spam, random_state=123)  # 从 ham 类别中随机采样，数量与 spam 相同
    balanced_df = pd.concat([ham_sampled, df[df["Label"] == "spam"]])  # 合并 spam 和 ham
    balanced_df = balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)  # 打乱数据顺序
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})  # 将标签 "ham" 映射为 0，"spam" 映射为 1

    # 将数据集拆分为训练集、验证集和测试集
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    # 保存数据集为 CSV 文件
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)
def instantiate_model(choose_model, load_weights):
    # 定义基本的模型配置
    BASE_CONFIG = {
        "vocab_size": 50257,     # 词汇表大小
        "context_length": 1024,  # 上下文长度
        "drop_rate": 0.0,        # Dropout 比率
        "qkv_bias": True         # 是否使用 Query-Key-Value 偏置
    }

    # 不同 GPT2 模型的配置
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},  # 小型模型
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},  # 中型模型
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},  # 大型模型
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},    # 超大模型
    }

    # 根据选择的模型更新配置
    BASE_CONFIG.update(model_configs[choose_model])

    # 如果不加载权重，则设置随机种子
    if not load_weights:
        torch.manual_seed(123)

    # 初始化 GPT 模型
    model = GPTModel(BASE_CONFIG, disable_causal_mask=args.disable_causal_mask)

    # 如果需要加载权重，则下载并加载相应的权重
    if load_weights:
        # 提取模型的大小（例如，124M、355M等）
        model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")
        # 下载并加载模型权重
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
        # 将下载的权重加载到模型中
        load_weights_into_gpt(model, params)

    # 设置模型为评估模式
    model.eval()
    return model  # 返回构建好的模型


def calc_loss_batch(input_batch, target_batch, model, device,
                    trainable_token_pos=-1, ignore_index=-100, average_embeddings=False):
    # 将输入批次和目标批次移至指定设备（例如GPU）
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    # 如果 trainable_token_pos 设置为 "flexible"，则选择最后一个非 padding 的位置
    if trainable_token_pos == "flexible":
        pad_token_id = 50256  # <|endoftext|> 是用于 padding 的特殊 token
        mask = input_batch != pad_token_id  # 创建一个掩码，标记非 padding 的位置
        last_token_pos = mask.sum(dim=1) - 1  # 获取每个序列的最后一个有效 token 的位置

        # 获取模型输出
        logits = model(input_batch)  # [batch_size, seq_len, num_classes] 形状的输出

        # 选择每个序列最后一个有效 token 的 logits
        batch_size = logits.size(0)
        selected_logits = logits[torch.arange(batch_size), last_token_pos]

        # 计算交叉熵损失
        loss = torch.nn.functional.cross_entropy(selected_logits, target_batch)
        return loss

    else:
        # 获取模型的输出
        model_output = model(input_batch)
        if average_embeddings:
            # 如果启用 average_embeddings，按序列维度（dim=1）平均输出
            logits = model_output.mean(dim=1)
        else:
            # 否则，选择指定位置的 token 的输出（例如，trainable_token_pos）
            logits = model_output[:, trainable_token_pos, :]

        # 计算交叉熵损失
        loss = torch.nn.functional.cross_entropy(logits, target_batch, ignore_index=ignore_index)
        return loss


def calc_loss_loader(data_loader, model, device,
                     num_batches=None, trainable_token_pos=-1,
                     ignore_index=-100, average_embeddings=False):
    # 初始化总损失为0
    total_loss = 0.
    # 如果数据加载器为空，返回 NaN
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)  # 默认处理所有批次
    else:
        # 如果指定了 num_batches，限制处理的批次数
        num_batches = min(num_batches, len(data_loader))
    # 遍历数据加载器中的每个批次
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            # 计算当前批次的损失
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
                average_embeddings=average_embeddings
            )
            total_loss += loss.item()  # 将当前批次的损失累加到总损失
        else:
            break
    # 返回平均损失
    return total_loss / num_batches


@torch.no_grad()  # 禁用梯度计算，提高效率
def calc_accuracy_loader(data_loader, model, device, num_batches=None,
                         trainable_token_pos=-1, average_embeddings=False):
    model.eval()  # 设置模型为评估模式
    correct_predictions, num_examples = 0, 0  # 初始化正确预测数量和总样本数

    if num_batches is None:
        num_batches = len(data_loader)  # 默认处理所有批次
    else:
        num_batches = min(num_batches, len(data_loader))  # 限制批次数

    # 如果 trainable_token_pos 设置为 "flexible"，则选择每个序列的最后一个非 padding 的 token
    if trainable_token_pos == "flexible":
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                # 找到每个序列的最后一个非 padding 的 token 位置
                pad_token_id = 50256  # <|endoftext|> 是用于 padding 的特殊 token
                mask = input_batch != pad_token_id
                last_token_pos = mask.sum(dim=1) - 1  # 获取最后一个有效 token 的位置

                with torch.no_grad():
                    logits = model(input_batch)  # 获取模型输出
                    # 选择每个序列的最后一个有效 token 的 logits
                    batch_size = logits.size(0)
                    selected_logits = logits[torch.arange(batch_size), last_token_pos]
                    predicted_labels = torch.argmax(selected_logits, dim=-1)  # 获取预测的标签

                num_examples += predicted_labels.shape[0]  # 更新样本数量
                correct_predictions += (predicted_labels == target_batch).sum().item()  # 统计正确预测的数量
            else:
                break

    else:
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                model_output = model(input_batch)
                if average_embeddings:
                    # 如果启用 average_embeddings，按序列维度（dim=1）平均输出
                    logits = model_output.mean(dim=1)
                else:
                    # 否则，选择指定位置的 token 的输出（例如，trainable_token_pos）
                    logits = model_output[:, trainable_token_pos, :]

                predicted_labels = torch.argmax(logits, dim=-1)  # 获取预测的标签

                num_examples += predicted_labels.shape[0]  # 更新样本数量
                correct_predictions += (predicted_labels == target_batch).sum().item()  # 统计正确预测的数量
            else:
                break

    # 返回正确率
    return correct_predictions / num_examples
def evaluate_model(model, train_loader, val_loader, device,
                   eval_iter, trainable_token_pos=-1,
                   ignore_index=-100, average_embeddings=False):
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 在不计算梯度的上下文中
        train_loss = calc_loss_loader(  # 计算训练集的损失
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
            average_embeddings=average_embeddings
        )
        val_loss = calc_loss_loader(  # 计算验证集的损失
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
            average_embeddings=average_embeddings
        )
    model.train()  # 将模型设置回训练模式
    return train_loss, val_loss  # 返回训练集和验证集的损失

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None, trainable_token_pos=-1,
                            accumulation_steps=1, ignore_index=-100, average_embeddings=False):
    # 初始化列表以跟踪损失和已看到的样本数
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # 主训练循环
    for epoch in range(num_epochs):  # 遍历每个训练周期
        model.train()  # 将模型设置为训练模式

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):  # 遍历训练数据加载器中的批次
            loss = calc_loss_batch(  # 计算批次的损失
                input_batch, target_batch, model, device,
                trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
                average_embeddings=average_embeddings
            )

            # 如果accumulation_steps > 1，则使用梯度累积
            loss /= accumulation_steps  # 将损失除以累积步数

            loss.backward()  # 计算损失梯度

            # 如果accumulation_steps > 1，则使用梯度累积
            is_update_step = ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader))  # 判断是否是更新步骤
            if is_update_step:  # 如果是更新步骤
                optimizer.step()  # 使用损失梯度更新模型权重
                optimizer.zero_grad()  # 重置之前批次迭代的损失梯度

            examples_seen += input_batch.shape[0]  # 更新已看到的样本数
            global_step += 1  # 更新全局步骤数

            # 可选的评估步骤
            if global_step % eval_freq == 0:  # 如果达到评估频率
                train_loss, val_loss = evaluate_model(  # 评估模型
                    model, train_loader, val_loader, device, eval_iter,
                    trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
                    average_embeddings=average_embeddings
                )
                train_losses.append(train_loss)  # 将训练损失添加到列表
                val_losses.append(val_loss)  # 将验证损失添加到列表
                print(f"Ep {epoch+1} (Step {global_step:06d}): "  # 打印训练和验证损失
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if max_steps is not None and global_step > max_steps:  # 如果达到最大步骤数
                break  # 退出循环

        # 在每个周期后计算准确率
        train_accuracy = calc_accuracy_loader(  # 计算训练集的准确率
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        val_accuracy = calc_accuracy_loader(  # 计算验证集的准确率
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")  # 打印训练准确率
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")  # 打印验证准确率
        train_accs.append(train_accuracy)  # 将训练准确率添加到列表
        val_accs.append(val_accuracy)  # 将验证准确率添加到列表

        if max_steps is not None and global_step > max_steps:  # 如果达到最大步骤数
            break  # 退出循环

    return train_losses, val_losses, train_accs, val_accs, examples_seen  # 返回训练和验证损失、准确率以及已看到的样本数

import argparse  # 导入argparse库，用于解析命令行参数

def replace_linear_with_lora(model, rank, alpha, alternative=False):  # 定义一个函数，用于将模型中的Linear层替换为LoRA层
    for name, module in model.named_children():  # 遍历模型的所有子模块
        if isinstance(module, torch.nn.Linear):  # 如果模块是Linear层
            # 替换Linear层为LinearWithLoRA或LinearWithLoRAMerged层
            if alternative:
                setattr(model, name, LinearWithLoRAMerged(module, rank, alpha))
            else:
                setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 对非Linear层的子模块递归应用相同的替换操作
            replace_linear_with_lora(module, rank, alpha)

if __name__ == "__main__":  # 如果这是主程序，则执行以下代码
    parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象

    parser.add_argument(  # 添加命令行参数
        "--model_size",
        type=str,
        default="gpt2-small (124M)",
        help=(
            "Which GPT model to use. Options: 'gpt2-small (124M)', 'gpt2-medium (355M)',"
             " 'gpt2-large (774M)', 'gpt2-xl (1558M)'."
        )
    )
    parser.add_argument(  # 添加命令行参数
        "--weights",
        type=str,
        default="pretrained",
        help=(
            "Whether to use 'pretrained' or 'random' weights."
        )
    )
    parser.add_argument(  # 添加命令行参数
        "--trainable_layers",
        type=str,
        default="last_block",
        help=(
            "Which layers to train. Options: 'all', 'last_block', 'last_two_blocks', 'last_layer', 'lora', 'lora_alternative'."
        )
    )
    parser.add_argument(  # 添加命令行参数
        "--trainable_token_pos",
        type=str,
        default="last",
        help=(
            "Which token position to train. Options: 'first', 'last', 'flexible'."
        )
    )
    parser.add_argument(  # 添加命令行参数
        "--average_embeddings",
        action='store_true',
        default=False,
        help=(
            "Average the output embeddings from all tokens instead of using"
             " only the embedding at the token position specified by `--trainable_token_pos`."
        )
    )
    parser.add_argument(  # 添加命令行参数
        "--context_length",
        type=str,
        default="longest_training_example",
        help=(
            "The context length of the data inputs."
             " Options: 'longest_training_example', 'model_context_length' or integer value."
        )
    )
    parser.add_argument(  # 添加命令行参数
        "--lora_rank",
        type=int,
        default=8,
        help=(
            "The LoRA rank when choosing `--trainable_layers lora`"
        )
    )
    parser.add_argument(  # 添加命令行参数
        "--lora_alpha",
        type=int,
        default=8,
        help=(
            "The LoRA alpha value when choosing `--trainable_layers lora`"
        )
    )
    parser.add_argument(  # 添加命令行参数
        "--no_padding",
        action='store_true',
        default=False,
        help=(
            "Disable padding, which means each example may have a different length."
             " This requires setting `--batch_size 1`."
        )
    )
    parser.add_argument(  # 添加命令行参数
        "--num_epochs",
        type=int,
        default=5,
        help=(
            "Number of training epochs."
        )
    )
    parser.add_argument(  # 添加命令行参数
        "--batch_size",
        type=int,
        default=8,
        help=(
            "The batch size used for training."
        )
    )
    parser.add_argument(  # 添加命令行参数
        "--accumulation_steps",
        type=int,
        default=1,
        help=(
            "Accumulation steps to allow for gradient accumulation."
             " See https://sebastianraschka.com/blog/2023/llm-grad-accumulation.html  for explanation."
             " For example, setting `batch_size=8` and `accumulation_steps=1` compute the exact same"
             " loss and weight updates as setting `batch_size=1` and `accumulation_steps=8`, however,"
             " the latter setting uses more iterations."
        )
    )
    parser.add_argument(  # 添加命令行参数
        "--disable_causal_mask",
        action='store_true',
        default=False,
        help=(
            "Disables the causal attention mask."
        )
    )
    parser.add_argument(  # 添加命令行参数
        "--ignore_index",
        type=int,
        default=-100,
        help=(
            "Sets the `ignore_index` in the cross-entropy loss."
        )
    )
    args = parser.parse_args()  # 解析命令行参数

    # 根据命令行参数设置可训练的token位置
    if args.trainable_token_pos == "first":
        args.trainable_token_pos = 0  # 如果设置为"first"，则将位置设置为0（序列的第一个token）
    elif args.trainable_token_pos == "last":
        args.trainable_token_pos = -1  # 如果设置为"last"，则将位置设置为-1（序列的最后一个token）
    # "flexible"设置选择padding token之前的最后一个token
    elif args.trainable_token_pos == "flexible":
        args.trainable_token_pos = "flexible"  # 设置为"flexible"，表示灵活选择最后一个非padding token的位置
    else:
        raise ValueError("Invalid --trainable_token_pos argument")  # 如果参数无效，抛出错误

    ###############################
    # Load model
    ###############################

    # 根据命令行参数决定是否加载预训练权重
    if args.weights == "pretrained":
        load_weights = True  # 如果选择"pretrained"，则加载预训练权重
    elif args.weights == "random":
        load_weights = False  # 如果选择"random"，则使用随机权重
    else:
        raise ValueError("Invalid --weights argument.")  # 如果参数无效，抛出错误

    model = instantiate_model(args.model_size, load_weights)  # 实例化模型
    for param in model.parameters():
        param.requires_grad = False  # 默认情况下，所有参数的梯度设置为False（不更新）

    # 根据模型大小设置输出层的输入特征数
    if args.model_size == "gpt2-small (124M)":
        in_features = 768
    elif args.model_size == "gpt2-medium (355M)":
        in_features = 1024
    elif args.model_size == "gpt2-large (774M)":
        in_features = 1280
    elif args.model_size == "gpt2-xl (1558M)":
        in_features = 1600
    else:
        raise ValueError("Invalid --model_size argument")  # 如果模型大小参数无效，抛出错误

    torch.manual_seed(123)  # 设置随机种子以确保结果可重复
    model.out_head = torch.nn.Linear(in_features=in_features, out_features=2)  # 设置模型的输出层

    # 根据命令行参数设置可训练的层
    if args.trainable_layers == "last_layer":
        pass  # 如果选择"last_layer"，则不进行任何操作（仅最后一个层可训练）
    elif args.trainable_layers == "last_block" or args.trainable_layers == "last_two_blocks":
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True  # 设置最后一个Transformer块的参数为可训练
        for param in model.final_norm.parameters():
            param.requires_grad = True  # 设置最后的归一化层参数为可训练
        if args.trainable_layers == "last_two_blocks":
            for param in model.trf_blocks[-2].parameters():
                param.requires_grad = True  # 如果选择"last_two_blocks"，则设置倒数第二个Transformer块的参数为可训练
    elif args.trainable_layers == "all":
        for param in model.parameters():
            param.requires_grad = True  # 如果选择"all"，则设置所有参数为可训练
    elif args.trainable_layers in ("lora", "lora_alternative"):
        if args.trainable_layers == "lora_alternative":
            alternative = True  # 如果选择"lora_alternative"，则使用替代的LoRA实现
        else:
            alternative = False  # 否则使用标准的LoRA实现
        replace_linear_with_lora(model, rank=args.lora_rank, alpha=args.lora_alpha,
                                 alternative=alternative)  # 替换Linear层为LoRA层
    else:
        raise ValueError("Invalid --trainable_layers argument.")  # 如果参数无效，抛出错误

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 确定设备（GPU或CPU）
    model.to(device)  # 将模型部署到指定设备

    ###############################
    # Instantiate dataloaders
    ###############################
    # 在这里，代码应该继续实例化数据加载器，但具体的代码未给出
    # 定义数据集的URL、zip文件路径、解压目录和新的文件路径
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extract_to = "sms_spam_collection"
    new_file_path = Path(extract_to) / "SMSSpamCollection.tsv"

    # 设置当前目录和CSV文件名列表
    base_path = Path(".")
    file_names = ["train.csv", "validation.csv", "test.csv"]
    # 检查所有CSV文件是否存在
    all_exist = all((base_path / file_name).exists() for file_name in file_names)

    # 如果CSV文件不存在，则下载数据集、解压并创建CSV文件
    if not all_exist:
        download_and_unzip(url, zip_path, extract_to, new_file_path)
        create_dataset_csvs(new_file_path)

    # 加载GPT-2分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 初始化训练数据集，如果需要则设置最大长度为None（不进行padding）
    train_dataset = None
    if args.no_padding:
        max_length = None
    else:
        # 根据上下文长度参数设置最大长度
        if args.context_length == "model_context_length":
            max_length = model.pos_emb.weight.shape[0]
        elif args.context_length == "longest_training_example":
            train_dataset = SpamDataset(base_path / "train.csv", max_length=None, tokenizer=tokenizer,
                                        no_padding=args.no_padding)
            max_length = train_dataset.max_length
        else:
            try:
                max_length = int(args.context_length)
            except ValueError:
                raise ValueError("Invalid --context_length argument")

    # 如果训练数据集未初始化，则根据最大长度初始化
    if train_dataset is None:
        train_dataset = SpamDataset(base_path / "train.csv", max_length=max_length, tokenizer=tokenizer,
                                    no_padding=args.no_padding)
    # 初始化验证和测试数据集
    val_dataset = SpamDataset(base_path / "validation.csv", max_length=max_length, tokenizer=tokenizer,
                              no_padding=args.no_padding)
    test_dataset = SpamDataset(base_path / "test.csv", max_length=max_length, tokenizer=tokenizer,
                               no_padding=args.no_padding)

    # 加载GPT-2分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 设置工作线程数为0
    num_workers = 0

    # 创建训练数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    # 创建验证数据加载器
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    # 创建测试数据加载器
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    # 确保数据集的最大长度不超过模型的上下文长度
    assert train_dataset.max_length <= model.pos_emb.weight.shape[0], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {model.pos_emb.weight.shape[0]}. Reinitialize data sets with "
        f"`max_length={model.pos_emb.weight.shape[0]}`"
    )

    ##############################
    # Train model
    ##############################

    # 记录开始时间，用于后续计算训练耗时
    start_time = time.time()
    torch.manual_seed(123)  # 设置随机种子以确保结果可重复
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)  # 使用AdamW优化器并设置学习率和权重衰减

    # 调用train_classifier_simple函数进行模型训练，并返回训练损失、验证损失、训练准确率、验证准确率以及处理的样本数
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs, eval_freq=50, eval_iter=5,
        max_steps=None, trainable_token_pos=args.trainable_token_pos,
        accumulation_steps=args.accumulation_steps, average_embeddings=args.average_embeddings
    )

    # 记录结束时间，并计算从开始到结束的总时间（分钟）
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")  # 打印训练完成所需的时间

    ##############################
    # Evaluate model
    ##############################

    # 使用calc_accuracy_loader函数计算训练集上的准确率
    train_accuracy = calc_accuracy_loader(
        train_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )
    # 使用calc_accuracy_loader函数计算验证集上的准确率
    val_accuracy = calc_accuracy_loader(
        val_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )
    # 使用calc_accuracy_loader函数计算测试集上的准确率
    test_accuracy = calc_accuracy_loader(
        test_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )

    # 打印训练集、验证集和测试集的准确率
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")