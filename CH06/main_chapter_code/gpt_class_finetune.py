# 版权所有 （c） Sebastian Raschka，Apache 许可证 2.0（参见 LICENSE.txt）。
# “从头开始构建大型语言模型” 的源代码
# - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch

# 这是一个摘要文件，包含第 6 章的主要内容。

import urllib.request  # 导入urllib.request模块，用于URL请求
import zipfile  # 导入zipfile模块，用于处理zip文件
import os  # 导入os模块，用于操作系统功能
from pathlib import Path  # 导入Path模块，用于路径操作
import time  # 导入time模块，用于时间相关功能

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot，用于绘图
import pandas as pd  # 导入pandas，用于数据处理
import tiktoken  # 导入tiktoken，用于分词
import torch  # 导入torch，PyTorch深度学习框架
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch的数据集和数据加载器模块

from gpt_download import download_and_load_gpt2  # 导入下载和加载GPT-2模型的函数
from previous_chapters import GPTModel, load_weights_into_gpt  # 导入GPT模型和权重加载函数

# 定义下载和解压垃圾邮件数据集的函数
def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path, test_mode=False):
    if data_file_path.exists():  # 如果数据文件已存在，则跳过下载和解压
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    if test_mode:  # 如果是测试模式，则尝试多次下载以处理CI连接问题
        max_retries = 5
        delay = 5  # 重试之间的延迟时间（秒）
        for attempt in range(max_retries):
            try:
                # 下载文件
                with urllib.request.urlopen(url, timeout=10) as response:
                    with open(zip_path, "wb") as out_file:
                        out_file.write(response.read())
                break  # 如果下载成功，则跳出循环
            except urllib.error.URLError as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)  # 等待后重试
                else:
                    print("Failed to download file after several attempts.")
                    return  # 如果所有重试都失败，则退出
    else:  # 非测试模式，正常下载文件
        # 下载文件
        with urllib.request.urlopen(url) as response:
            with open(zip_path, "wb") as out_file:
                out_file.write(response.read())

    # 解压文件
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # 添加.tsv文件扩展名
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")  # 打印文件下载和保存信息

# 定义创建平衡数据集的函数
def create_balanced_dataset(df):
    # 计算"spam"的实例数量
    num_spam = df[df["Label"] == "spam"].shape[0]

    # 随机抽样"ham"实例以匹配"spam"实例的数量
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # 将"ham"子集与"spam"合并
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df  # 返回平衡的数据集

# 定义随机分割数据集的函数
def random_split(df, train_frac, validation_frac):
    # 整个数据框架进行洗牌
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 计算分割索引
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # 分割数据框架
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df  # 返回训练、验证和测试数据集

# 定义PyTorch数据集类
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)  # 读取CSV文件

        # 预标记化文本
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:  # 如果没有指定最大长度
            self.max_length = self._longest_encoded_length()  # 获取最长编码长度
        else:
            self.max_length = max_length  # 设置最大长度
            # 截断序列如果它们超过最大长度
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # 将序列填充到最长序列
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]  # 获取编码文本
        label = self.data.iloc[index]["Label"]  # 获取标签
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )  # 返回编码文本和标签的张量

    def __len__(self):
        return len(self.data)  # 返回数据的长度

    def _longest_encoded_length(self):
        max_length = 0  # 初始化最大长度
        for encoded_text in self.encoded_texts:  # 遍历编码文本列表
            encoded_length = len(encoded_text)  # 获取编码长度
            if encoded_length > max_length:  # 如果当前编码长度大于最大长度
                max_length = encoded_length  # 更新最大长度
        return max_length  # 返回最大长度

# 定义计算准确率的函数
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()  # 将模型设置为评估模式
    correct_predictions, num_examples = 0, 0  # 初始化正确预测数和样本数

    if num_batches is None:  # 如果没有指定批次数量
        num_batches = len(data_loader)  # 使用数据加载器的全部批次
    else:
        num_batches = min(num_batches, len(data_loader))  # 如果指定的批次数量大于数据加载器的批次数量，则取较小值
    for i, (input_batch, target_batch) in enumerate(data_loader):  # 遍历数据加载器中的批次
        if i < num_batches:  # 如果当前批次小于指定的批次数量
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将输入和目标数据移动到指定的设备

            with torch.no_grad():  # 在不计算梯度的上下文中
                logits = model(input_batch)[:, -1, :]  # 获取最后一个输出标记的Logits
            predicted_labels = torch.argmax(logits, dim=-1)  # 获取预测标签

            num_examples += predicted_labels.shape[0]  # 更新样本数
            correct_predictions += (predicted_labels == target_batch).sum().item()  # 更新正确预测数
        else:
            break  # 如果已经处理了指定的批次数量，则退出循环
    return correct_predictions / num_examples  # 返回准确率
def calc_loss_batch(input_batch, target_batch, model, device):
    # 将输入数据和目标数据移动到指定的设备（CPU或GPU）
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 获取模型的输出（logits），只取最后一个token的输出
    logits = model(input_batch)[:, -1, :]
    # 使用交叉熵损失函数计算损失
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    # 如果数据加载器为空，返回NaN
    if len(data_loader) == 0:
        return float("nan")
    # 如果没有指定批次数，则使用数据加载器中的批次数
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 如果指定了批次数，取最小值（防止超出数据加载器的长度）
        num_batches = min(num_batches, len(data_loader))

    # 遍历数据加载器中的批次
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            # 计算当前批次的损失
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 累加总损失
            total_loss += loss.item()
        else:
            break
    # 返回平均损失
    return total_loss / num_batches
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # 将模型设置为评估模式（禁用dropout等）
    model.eval()
    with torch.no_grad():  # 禁用梯度计算
        # 计算训练集和验证集的损失
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    # 重新将模型设置为训练模式
    model.train()
    return train_loss, val_loss
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, tokenizer):
    # 初始化用于记录损失和准确率的列表
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    # 主训练循环
    for epoch in range(num_epochs):
        model.train()  # 将模型设置为训练模式

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 重置前一个批次的梯度
            # 计算当前批次的损失
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # 计算梯度
            optimizer.step()  # 使用梯度更新模型参数
            examples_seen += input_batch.shape[0]  # 更新处理的样本数
            global_step += 1  # 更新全局步骤数
            # 可选的评估步骤，每隔eval_freq步评估一次模型
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                # 将当前的训练和验证损失添加到列表中
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 计算每个epoch后的准确率
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        # 将训练和验证准确率添加到对应的列表中
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)


    return train_losses, val_losses, train_accs, val_accs, examples_seen
    # 返回记录的训练损失、验证损失、训练准确率、验证准确率和已处理的样本数量
def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 绘制训练和验证损失的图表
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")  # 设置x轴标签为“Epochs”
    ax1.set_ylabel(label.capitalize())  # 设置y轴标签
    ax1.legend()

    # 创建第二个x轴，用于显示已处理的样本数
    ax2 = ax1.twiny()  # 创建共享y轴的第二个x轴
    ax2.plot(examples_seen, train_values, alpha=0)  # 绘制一个不可见的图，用于对齐刻度
    ax2.set_xlabel("Examples seen")  # 设置第二个x轴的标签为“Examples seen”

    fig.tight_layout()  # 调整布局，使图表更加紧凑
    plt.savefig(f"{label}-plot.pdf")  # 保存为PDF文件
    # plt.show()  # 如果需要，可以取消注释以显示图表

if __name__ == "__main__":  # 如果这是主程序，则执行以下代码

    import argparse  # 导入argparse库，用于解析命令行参数

    parser = argparse.ArgumentParser(  # 创建ArgumentParser对象
        description="Finetune a GPT model for classification"  # 描述信息
    )
    parser.add_argument(  # 添加命令行参数
        "--test_mode",  # 参数名
        default=False,  # 默认值为False
        action="store_true",  # 如果指定了该参数，则将其值设为True
        help=("This flag runs the model in test mode for internal testing purposes. "
              "Otherwise, it runs the model as it is used in the chapter (recommended).")  # 帮助信息
    )
    args = parser.parse_args()  # 解析命令行参数

    ########################################
    # Download and prepare dataset
    ########################################

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"  # 数据集URL
    zip_path = "sms_spam_collection.zip"  # zip文件保存路径
    extracted_path = "sms_spam_collection"  # 解压后的文件夹名称
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"  # 数据文件保存路径

    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path, test_mode=args.test_mode)  # 下载并解压数据集
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])  # 读取数据文件
    balanced_df = create_balanced_dataset(df)  # 创建平衡数据集
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})  # 将标签映射为0和1

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)  # 随机分割数据集
    train_df.to_csv("train.csv", index=None)  # 保存训练数据集
    validation_df.to_csv("validation.csv", index=None)  # 保存验证数据集
    test_df.to_csv("test.csv", index=None)  # 保存测试数据集

    ########################################
    # Create data loaders
    ########################################
    tokenizer = tiktoken.get_encoding("gpt2")  # 获取GPT-2的编码器

    train_dataset = SpamDataset(  # 创建训练数据集
        csv_file="train.csv",
        max_length=None,
        tokenizer=tokenizer
    )

    val_dataset = SpamDataset(  # 创建验证数据集
        csv_file="validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    test_dataset = SpamDataset(  # 创建测试数据集
        csv_file="test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    num_workers = 0  # 数据加载器的工作线程数
    batch_size = 8  # 批次大小

    torch.manual_seed(123)  # 设置PyTorch的随机种子

    train_loader = DataLoader(  # 创建训练数据加载器
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(  # 创建验证数据加载器
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(  # 创建测试数据加载器
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    ########################################
    # Load pretrained model
    ########################################

    # 如果是测试模式，使用较小的GPT模型
    if args.test_mode:
        BASE_CONFIG = {
            "vocab_size": 50257,  # 词汇表大小
            "context_length": 120,  # 上下文长度
            "drop_rate": 0.0,  # Dropout比例
            "qkv_bias": False,  # 是否使用QKV偏置
            "emb_dim": 12,  # 嵌入层维度
            "n_layers": 1,  # 模型层数
            "n_heads": 2  # 注意力头数
        }
        model = GPTModel(BASE_CONFIG)  # 创建GPT模型
        model.eval()  # 设置为评估模式
        device = "cpu"  # 使用CPU进行推理

    # 如果不是测试模式，则加载实际的GPT模型
    else:
        CHOOSE_MODEL = "gpt2-small (124M)"  # 选择的模型是gpt2-small
        INPUT_PROMPT = "Every effort moves"  # 输入提示语

        BASE_CONFIG = {
            "vocab_size": 50257,  # 词汇表大小
            "context_length": 1024,  # 上下文长度
            "drop_rate": 0.0,  # Dropout比例
            "qkv_bias": True  # 是否使用QKV偏置
        }

        # 配置不同大小GPT模型的参数
        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])  # 更新配置，选择对应模型的参数

        # 检查训练数据集的最大长度是否超过模型的上下文长度
        assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
            f"Dataset length {train_dataset.max_length} exceeds model's context "
            f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
            f"`max_length={BASE_CONFIG['context_length']}`"
        )

        # 从模型名称中提取模型大小，进行下载和加载
        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

        model = GPTModel(BASE_CONFIG)  # 创建GPT模型
        load_weights_into_gpt(model, params)  # 加载预训练模型的权重
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备（GPU或CPU）

    ########################################
    # Modify and pretrained model
    ########################################

    # 设置模型的参数为不需要梯度计算
    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)  # 设置随机种子

    num_classes = 2  # 输出类别数（例如：二分类任务）
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)  # 添加分类头
    model.to(device)  # 将模型移到指定的设备（GPU或CPU）

    # 允许训练最后一层的参数
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    # 允许训练最终归一化层的参数
    for param in model.final_norm.parameters():
        param.requires_grad = True

    ########################################
    # Finetune modified model
    ########################################

    start_time = time.time()  # 记录训练开始时间
    torch.manual_seed(123)  # 设置随机种子

    # 使用AdamW优化器，设置学习率和权重衰减
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    num_epochs = 5  # 设置训练的epoch数
    # 调用训练函数，进行模型微调
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,  # 每50步评估一次，评估5个批次
        tokenizer=tokenizer
    )

    end_time = time.time()  # 记录训练结束时间
    execution_time_minutes = (end_time - start_time) / 60  # 计算训练的总时间（分钟）
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")  # 打印训练完成的时间

    ########################################
    # Plot results
    ########################################

    # 绘制损失曲线
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))  # 创建训练的epoch张量
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))  # 创建已处理样本数张量
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)  # 绘制损失图

    # 绘制准确率曲线
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))  # 创建训练的epoch张量
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))  # 创建已处理样本数张量
    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")  # 绘制准确率图
