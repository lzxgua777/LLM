# 版权声明：Sebastian Raschka 根据 Apache License 2.0（参见 LICENSE.txt）持有版权。
# "Build a Large Language Model From Scratch" 的来源
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch

import argparse
import os
from pathlib import Path
import time
import urllib
import zipfile

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 定义一个用于处理垃圾邮件数据集的类
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256, no_padding=False):
        self.data = pd.read_csv(csv_file)  # 读取CSV文件
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)  # 设置最大序列长度

        # 对文本进行预标记化
        self.encoded_texts = [
            tokenizer.encode(text)[:self.max_length]
            for text in self.data["Text"]
        ]

        if not no_padding:
            # 对序列进行填充，使其长度一致
            self.encoded_texts = [
                et + [pad_token_id] * (self.max_length - len(et))
                for et in self.encoded_texts
            ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)  # 返回编码后的文本和标签

    def __len__(self):
        return len(self.data)  # 返回数据集中的样本数量

    def _longest_encoded_length(self, tokenizer):
        max_length = 0
        for text in self.data["Text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length  # 返回所有文本中编码后最长的长度

# 定义一个函数用于下载并解压文件
def download_and_unzip(url, zip_path, extract_to, new_file_path):
    if new_file_path.exists():
        print(f"{new_file_path} already exists. Skipping download and extraction.")
        return

    # 下载文件
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # 解压文件
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    # 重命名文件以指示其格式
    original_file = Path(extract_to) / "SMSSpamCollection"
    os.rename(original_file, new_file_path)
    print(f"File downloaded and saved as {new_file_path}")

# 定义一个函数用于随机分割数据集
def random_split(df, train_frac, validation_frac):
    # 打乱整个DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 计算分割索引
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # 分割DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df  # 返回训练集、验证集和测试集

# 定义一个函数用于创建数据集CSV文件
def create_dataset_csvs(new_file_path):
    df = pd.read_csv(new_file_path, sep="\t", header=None, names=["Label", "Text"])  # 读取数据集文件

    # 创建平衡的数据集
    n_spam = df[df["Label"] == "spam"].shape[0]
    ham_sampled = df[df["Label"] == "ham"].sample(n_spam, random_state=123)
    balanced_df = pd.concat([ham_sampled, df[df["Label"] == "spam"]])
    balanced_df = balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})  # 将标签映射为0和1

    # 采样并保存CSV文件
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)  # 保存训练集、验证集和测试集的CSV文件
# 定义一个用于处理垃圾邮件数据集的类，继承自PyTorch的Dataset类
class SPAMDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256, use_attention_mask=False):
        self.data = pd.read_csv(csv_file)  # 读取CSV文件中的数据
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)  # 设置最大序列长度，如果没有指定，则自动计算
        self.pad_token_id = pad_token_id  # 填充 token 的ID
        self.use_attention_mask = use_attention_mask  # 是否使用注意力掩码

        # 预标记化文本，并在需要时创建注意力掩码
        self.encoded_texts = [
            tokenizer.encode(text, truncation=True, max_length=self.max_length)
            for text in self.data["Text"]
        ]
        self.encoded_texts = [
            et + [pad_token_id] * (self.max_length - len(et))
            for et in self.encoded_texts
        ]  # 对序列进行填充

        if self.use_attention_mask:
            self.attention_masks = [
                self._create_attention_mask(et)
                for et in self.encoded_texts
            ]
        else:
            self.attention_masks = None  # 如果不需要注意力掩码，则设置为None

    # 创建注意力掩码的函数
    def _create_attention_mask(self, encoded_text):
        return [1 if token_id != self.pad_token_id else 0 for token_id in encoded_text]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]  # 获取编码后的文本
        label = self.data.iloc[index]["Label"]  # 获取标签

        if self.use_attention_mask:
            attention_mask = self.attention_masks[index]  # 如果使用注意力掩码，则获取对应的掩码
        else:
            attention_mask = torch.ones(self.max_length, dtype=torch.long)  # 否则，创建一个全1的掩码

        return (
            torch.tensor(encoded, dtype=torch.long),  # 将编码后的文本转换为张量
            torch.tensor(attention_mask, dtype=torch.long),  # 将注意力掩码转换为张量
            torch.tensor(label, dtype=torch.long)  # 将标签转换为张量
        )

    def __len__(self):
        return len(self.data)  # 返回数据集中的样本数量

    def _longest_encoded_length(self, tokenizer):
        max_length = 0  # 初始化最大长度
        for text in self.data["Text"]:
            encoded_length = len(tokenizer.encode(text))  # 计算编码后的长度
            if encoded_length > max_length:
                max_length = encoded_length  # 更新最大长度
        return max_length  # 返回所有文本中编码后最长的长度


# 计算一批数据的损失函数
def calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device):
    attention_mask_batch = attention_mask_batch.to(device)  # 将注意力掩码移动到指定设备
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将输入和目标移动到指定设备
    # logits = model(input_batch)[:, -1, :]  # 获取最后一个输出token的logits（已注释）
    logits = model(input_batch, attention_mask=attention_mask_batch).logits  # 获取logits
    loss = torch.nn.functional.cross_entropy(logits, target_batch)  # 计算交叉熵损失
    return loss


# 计算数据加载器的损失函数
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.  # 初始化总损失
    if num_batches is None:
        num_batches = len(data_loader)  # 如果没有指定批次数量，则使用数据加载器中的所有批次
    else:
        num_batches = min(num_batches, len(data_loader))  # 如果指定的批次数量超过了数据加载器中的批次数量，则取较小值
    for i, (input_batch, attention_mask_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device)  # 计算每批的损失
            total_loss += loss.item()  # 累加损失
        else:
            break
    return total_loss / num_batches  # 返回平均损失


# 不计算梯度，提高效率
@torch.no_grad()
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()  # 将模型设置为评估模式
    correct_predictions, num_examples = 0, 0  # 初始化正确预测数和样本数

    if num_batches is None:
        num_batches = len(data_loader)  # 如果没有指定批次数量，则使用数据加载器中的所有批次
    else:
        num_batches = min(num_batches, len(data_loader))  # 如果指定的批次数量超过了数据加载器中的批次数量，则取较小值
    for i, (input_batch, attention_mask_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            attention_mask_batch = attention_mask_batch.to(device)  # 将注意力掩码移动到指定设备
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将输入和目标移动到指定设备
            # logits = model(input_batch)[:, -1, :]  # 获取最后一个输出token的logits（已注释）
            logits = model(input_batch, attention_mask=attention_mask_batch).logits  # 获取logits
            predicted_labels = torch.argmax(logits, dim=1)  # 获取预测的标签
            num_examples += predicted_labels.shape[0]  # 更新样本数
            correct_predictions += (predicted_labels == target_batch).sum().item()  # 更新正确预测数
        else:
            break
    return correct_predictions / num_examples  # 返回准确率
# 定义一个函数用于评估模型性能
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 关闭梯度计算
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)  # 计算训练集损失
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)  # 计算验证集损失
    model.train()  # 将模型设置回训练模式
    return train_loss, val_loss  # 返回训练集和验证集的损失


# 定义一个函数用于训练分类器
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None):
    # 初始化列表以跟踪损失和已处理的样本数
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1  # 初始化已处理样本数和全局步数

    # 主训练循环
    for epoch in range(num_epochs):  # 遍历每个epoch
        model.train()  # 将模型设置为训练模式

        for input_batch, attention_mask_batch, target_batch in train_loader:  # 遍历训练数据加载器中的批次
            optimizer.zero_grad()  # 清零梯度
            loss = calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 根据梯度更新模型权重
            examples_seen += input_batch.shape[0]  # 更新已处理样本数
            global_step += 1  # 更新全局步数

            # 可选的评估步骤
            if global_step % eval_freq == 0:  # 如果达到评估频率
                train_loss, val_loss = evaluate_model(  # 评估模型
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)  # 记录训练损失
                val_losses.append(val_loss)  # 记录验证损失
                print(f"Ep {epoch+1} (Step {global_step:06d}): "  # 打印当前epoch和步数
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if max_steps is not None and global_step > max_steps:  # 如果达到最大步数
                break  # 终止训练

        # 在每个epoch后计算准确率
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)  # 计算训练集准确率
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)  # 计算验证集准确率
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")  # 打印训练集准确率
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")  # 打印验证集准确率
        train_accs.append(train_accuracy)  # 记录训练集准确率
        val_accs.append(val_accuracy)  # 记录验证集准确率

        if max_steps is not None and global_step > max_steps:  # 如果达到最大步数
            break  # 终止训练

    return train_losses, val_losses, train_accs, val_accs, examples_seen  # 返回损失和准确率以及已处理样本数


# 如果是主程序，则执行以下代码
if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument(
        "--trainable_layers",
        type=str,
        default="all",
        help=(
            "Which layers to train. Options: 'all', 'last_block', 'last_layer'."
        )
    )
    parser.add_argument(
        "--use_attention_mask",
        type=str,
        default="true",
        help=(
            "Whether to use a attention mask for padding tokens. Options: 'true', 'false'."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert",
        help=(
            "Which model to train. Options: 'distilbert', 'bert', 'roberta'."
        )
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help=(
            "Number of epochs."
        )
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help=(
            "Learning rate."
        )
    )
    args = parser.parse_args()  # 解析参数

    ###############################
    # Load model
    ###############################

    torch.manual_seed(123)  # 设置随机种子以确保结果可复现
    if args.model == "distilbert":  # 如果选择的模型是distilbert
        model = AutoModelForSequenceClassification.from_pretrained(  # 从预训练模型加载
            "distilbert-base-uncased", num_labels=2)
        model.out_head = torch.nn.Linear(in_features=768, out_features=2)  # 添加输出层
        for param in model.parameters():  # 遍历模型所有参数
            param.requires_grad = False  # 设置为不更新梯度
        if args.trainable_layers == "last_layer":  # 如果只训练最后一层
            for param in model.out_head.parameters():  # 只更新输出层的梯度
                param.requires_grad = True
        elif args.trainable_layers == "last_block":  # 如果只训练最后一个块
            for param in model.pre_classifier.parameters():  # 更新预分类器的梯度
                param.requires_grad = True
            for param in model.distilbert.transformer.layer[-1].parameters():  # 更新最后一个transformer层的梯度
                param.requires_grad = True
        elif args.trainable_layers == "all":  # 如果训练所有层
            for param in model.parameters():  # 更新所有参数的梯度
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")  # 如果参数无效，抛出异常

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # 从预训练模型加载分词器
        # 如果选择的模型是BERT

    elif args.model == "bert":
        model = AutoModelForSequenceClassification.from_pretrained(  # 从预训练模型加载
                "bert-base-uncased", num_labels=2)  # 加载BERT基础模型，设置标签数为2
        model.classifier = torch.nn.Linear(in_features=768, out_features=2)  # 添加自定义分类器层
        for param in model.parameters():  # 遍历模型所有参数
            param.requires_grad = False  # 设置为不更新梯度
        if args.trainable_layers == "last_layer":  # 如果只训练最后一层
            for param in model.classifier.parameters():  # 只更新分类器层的梯度
                param.requires_grad = True
        elif args.trainable_layers == "last_block":  # 如果只训练最后一块
            for param in model.classifier.parameters():  # 更新分类器层的梯度
                param.requires_grad = True
            for param in model.bert.pooler.dense.parameters():  # 更新BERT池化层的梯度
                param.requires_grad = True
            for param in model.bert.encoder.layer[-1].parameters():  # 更新BERT最后一个编码器层的梯度
                param.requires_grad = True
        elif args.trainable_layers == "all":  # 如果训练所有层
            for param in model.parameters():  # 更新所有参数的梯度
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")  # 如果参数无效，抛出异常

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # 从预训练模型加载分词器

    # 如果选择的模型是RoBERTa
    elif args.model == "roberta":
        model = AutoModelForSequenceClassification.from_pretrained(  # 从预训练模型加载
            "FacebookAI/roberta-large", num_labels=2)  # 加载RoBERTa大型模型，设置标签数为2
        model.classifier.out_proj = torch.nn.Linear(in_features=1024, out_features=2)  # 添加自定义分类器层
        for param in model.parameters():  # 遍历模型所有参数
            param.requires_grad = False  # 设置为不更新梯度
        if args.trainable_layers == "last_layer":  # 如果只训练最后一层
            for param in model.classifier.parameters():  # 只更新分类器层的梯度
                param.requires_grad = True
        elif args.trainable_layers == "last_block":  # 如果只训练最后一块
            for param in model.classifier.parameters():  # 更新分类器层的梯度
                param.requires_grad = True
            for param in model.roberta.encoder.layer[-1].parameters():  # 更新RoBERTa最后一个编码器层的梯度
                param.requires_grad = True
        elif args.trainable_layers == "all":  # 如果训练所有层
            for param in model.parameters():  # 更新所有参数的梯度
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")  # 如果参数无效，抛出异常

        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")  # 从预训练模型加载分词器

    # 如果选择的模型不受支持
    else:
        raise ValueError("Selected --model {args.model} not supported.")  # 抛出异常提示模型不受支持

    # 设置设备，如果GPU可用则使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 将模型移动到指定设备
    model.eval()  # 将模型设置为评估模式

    ##############################
    # 实例化数据加载器
    ##############################

    # 由于网络原因，无法成功解析提供的链接。这可能是由于链接的问题或网络问题。请检查链接的合法性并适当重试。
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extract_to = "sms_spam_collection"
    new_file_path = Path(extract_to) / "SMSSpamCollection.tsv"

    base_path = Path(".")  # 设置基础路径为当前目录
    file_names = ["train.csv", "validation.csv", "test.csv"]  # 定义CSV文件名
    all_exist = all((base_path / file_name).exists() for file_name in file_names)  # 检查所有CSV文件是否存在

    if not all_exist:  # 如果文件不存在
        download_and_unzip(url, zip_path, extract_to, new_file_path)  # 下载并解压数据集
        create_dataset_csvs(new_file_path)  # 创建数据集CSV文件

    # 根据参数设置是否使用注意力掩码
    if args.use_attention_mask.lower() == "true":
        use_attention_mask = True
    elif args.use_attention_mask.lower() == "false":
        use_attention_mask = False
    else:
        raise ValueError("Invalid argument for `use_attention_mask`.")  # 如果参数无效，抛出异常

    # 实例化训练集、验证集和测试集的数据集对象
    train_dataset = SPAMDataset(
        base_path / "train.csv",
        max_length=256,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        use_attention_mask=use_attention_mask
    )
    val_dataset = SPAMDataset(
        base_path / "validation.csv",
        max_length=256,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        use_attention_mask=use_attention_mask
    )
    test_dataset = SPAMDataset(
        base_path / "test.csv",
        max_length=256,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        use_attention_mask=use_attention_mask
    )

    num_workers = 0  # 设置数据加载器的工作线程数
    batch_size = 8  # 设置批处理大小

    # 实例化训练集、验证集和测试集的数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    ##############################
    # 训练模型
    ##############################

    start_time = time.time()  # 记录开始时间
    torch.manual_seed(123)  # 设置随机种子以确保结果可复现
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)  # 实例化优化器

    # 调用训练函数并传入参数
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs, eval_freq=50, eval_iter=20,
        max_steps=None
    )

    end_time = time.time()  # 记录结束时间
    execution_time_minutes = (end_time - start_time) / 60  # 计算执行时间（分钟）
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")  # 打印训练完成所需的时间

    ##############################
    # 评估模型
    ##############################

    print("\nEvaluating on the full datasets ...\n")

    # 计算并打印训练集、验证集和测试集的准确率
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")