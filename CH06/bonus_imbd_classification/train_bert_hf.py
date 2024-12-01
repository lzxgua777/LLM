# 版权所有 (c) Sebastian Raschka，遵循 Apache 2.0 许可证（请参见 LICENSE.txt）。
# 来源： "从头开始构建大语言模型"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码地址： https://github.com/rasbt/LLMs-from-scratch

import argparse  # 导入 argparse 模块，用于命令行参数解析
from pathlib import Path  # 从 pathlib 模块导入 Path 类，用于路径操作
import time  # 导入 time 模块，用于时间相关的操作

import pandas as pd  # 导入 pandas 库，用于数据处理和操作
import torch  # 导入 PyTorch 库，用于深度学习和张量计算
from torch.utils.data import DataLoader  # 从 PyTorch 的 utils.data 模块导入 DataLoader 类，用于批量数据加载
from torch.utils.data import Dataset  # 从 PyTorch 的 utils.data 模块导入 Dataset 类，用于定义数据集

from transformers import AutoTokenizer, AutoModelForSequenceClassification  # 从 transformers 库导入 AutoTokenizer 和 AutoModelForSequenceClassification，分别用于自动加载分词器和预训练模型（用于序列分类任务）

class IMDBDataset(Dataset):
    # 初始化数据集，处理 IMDB 数据
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256, use_attention_mask=False):
        self.data = pd.read_csv(csv_file)  # 读取 CSV 文件
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)  # 如果 max_length 为 None，则计算最长编码长度
        self.pad_token_id = pad_token_id  # 填充 token 的 ID
        self.use_attention_mask = use_attention_mask  # 是否使用 attention mask

        # 对文本进行预处理并创建 attention masks（如果需要）
        self.encoded_texts = [
            tokenizer.encode(text, truncation=True, max_length=self.max_length)  # 对每个文本进行编码
            for text in self.data["text"]
        ]
        self.encoded_texts = [
            et + [pad_token_id] * (self.max_length - len(et))  # 填充每个编码文本直到 max_length 长度
            for et in self.encoded_texts
        ]

        if self.use_attention_mask:
            self.attention_masks = [
                self._create_attention_mask(et)  # 创建 attention mask
                for et in self.encoded_texts
            ]
        else:
            self.attention_masks = None  # 如果不需要 attention mask，设为 None

    def _create_attention_mask(self, encoded_text):
        # 根据 token 的 id 创建 attention mask
        return [1 if token_id != self.pad_token_id else 0 for token_id in encoded_text]

    def __getitem__(self, index):
        # 获取给定索引的数据
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["label"]  # 获取标签

        if self.use_attention_mask:
            attention_mask = self.attention_masks[index]
        else:
            attention_mask = torch.ones(self.max_length, dtype=torch.long)  # 如果不使用 attention mask，则返回全为 1 的张量

        return (
            torch.tensor(encoded, dtype=torch.long),  # 输入编码
            torch.tensor(attention_mask, dtype=torch.long),  # attention mask
            torch.tensor(label, dtype=torch.long)  # 标签
        )

    def __len__(self):
        return len(self.data)  # 返回数据集大小

    def _longest_encoded_length(self, tokenizer):
        # 计算数据集中最长的文本编码长度
        max_length = 0
        for text in self.data["text"]:
            encoded_length = len(tokenizer.encode(text))  # 获取文本编码后的长度
            if encoded_length > max_length:
                max_length = encoded_length  # 更新最大长度
        return max_length


def calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device):
    # 计算一个批次的损失
    attention_mask_batch = attention_mask_batch.to(device)  # 将 attention mask 移动到设备
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将输入和目标标签移动到设备
    # logits = model(input_batch)[:, -1, :]  # 获取最后一个 token 的 logits
    logits = model(input_batch, attention_mask=attention_mask_batch).logits  # 获取模型的 logits
    loss = torch.nn.functional.cross_entropy(logits, target_batch)  # 使用交叉熵损失函数计算损失
    return loss


# 计算数据加载器中所有批次的损失
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if num_batches is None:
        num_batches = len(data_loader)  # 如果没有指定批次数量，则使用所有批次
    else:
        # 如果指定的批次数量大于数据加载器中的批次数量，则使用数据加载器中的批次数量
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, attention_mask_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device)  # 计算每个批次的损失
            total_loss += loss.item()  # 累加损失
        else:
            break
    return total_loss / num_batches  # 返回平均损失


@torch.no_grad()  # 禁用梯度计算，提升效率
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()  # 设置模型为评估模式
    correct_predictions, num_examples = 0, 0  # 初始化正确预测数和样本总数

    if num_batches is None:
        num_batches = len(data_loader)  # 如果没有指定批次数量，则使用所有批次
    else:
        num_batches = min(num_batches, len(data_loader))  # 如果指定的批次数量大于数据加载器中的批次数量，则使用数据加载器中的批次数量
    for i, (input_batch, attention_mask_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            attention_mask_batch = attention_mask_batch.to(device)  # 将 attention mask 移动到设备
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将输入和目标标签移动到设备
            # logits = model(input_batch)[:, -1, :]  # 获取最后一个 token 的 logits
            logits = model(input_batch, attention_mask=attention_mask_batch).logits  # 获取模型的 logits
            predicted_labels = torch.argmax(logits, dim=1)  # 获取预测的标签
            num_examples += predicted_labels.shape[0]  # 更新样本数
            correct_predictions += (predicted_labels == target_batch).sum().item()  # 更新正确预测数
        else:
            break
    return correct_predictions / num_examples  # 返回准确率


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)  # 计算训练集损失
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)  # 计算验证集损失
    model.train()  # 设置模型为训练模式
    return train_loss, val_loss  # 返回训练集和验证集的损失


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None):
    # 初始化跟踪损失和样本数的列表
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1  # 初始化样本数和全局步骤

    # 主要的训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式

        for input_batch, attention_mask_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 重置之前批次的损失梯度
            loss = calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device)  # 计算损失
            loss.backward()  # 计算损失梯度
            optimizer.step()  # 使用损失梯度更新模型权重
            examples_seen += input_batch.shape[0]  # 记录已见样本数
            global_step += 1  # 更新全局步骤

            # 可选的评估步骤
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)  # 评估模型
                train_losses.append(train_loss)  # 跟踪训练损失
                val_losses.append(val_loss)  # 跟踪验证损失
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if max_steps is not None and global_step > max_steps:  # 如果设置了最大步骤数，且达到最大步骤数则提前终止
                break

        # 计算每个 epoch 后的准确率
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)  # 计算训练集准确率
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)  # 计算验证集准确率
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)  # 跟踪训练准确率
        val_accs.append(val_accuracy)  # 跟踪验证准确率

        if max_steps is not None and global_step > max_steps:  # 如果设置了最大步骤数，且达到最大步骤数则提前终止
            break

    return train_losses, val_losses, train_accs, val_accs, examples_seen  # 返回训练过程中的损失和准确率信息
if __name__ == "__main__":  # 当脚本作为主程序运行时执行以下代码块
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()

    # 定义训练时可选的参数 --trainable_layers，指定哪些层需要训练
    parser.add_argument(
        "--trainable_layers",  # 参数名称
        type=str,  # 参数类型为字符串
        default="all",  # 默认值为"all"，表示训练所有层
        help=(
            "Which layers to train. Options: 'all', 'last_block', 'last_layer'."
        )  # 参数说明
    )

    # 定义是否使用 attention mask 参数 --use_attention_mask
    parser.add_argument(
        "--use_attention_mask",  # 参数名称
        type=str,  # 参数类型为字符串
        default="true",  # 默认值为"true"
        help=(
            "Whether to use an attention mask for padding tokens. Options: 'true', 'false'."
        )  # 参数说明
    )

    # 定义选择训练的模型 --model
    parser.add_argument(
        "--model",  # 参数名称
        type=str,  # 参数类型为字符串
        default="distilbert",  # 默认使用模型为 "distilbert"
        help=(
            "Which model to train. Options: 'distilbert', 'bert', 'roberta'."
        )  # 参数说明
    )

    # 定义训练的轮次 --num_epochs
    parser.add_argument(
        "--num_epochs",  # 参数名称
        type=int,  # 参数类型为整数
        default=1,  # 默认训练 1 轮
        help=(
            "Number of epochs."
        )  # 参数说明
    )

    # 定义学习率 --learning_rate
    parser.add_argument(
        "--learning_rate",  # 参数名称
        type=float,  # 参数类型为浮动数
        default=5e-6,  # 默认学习率为 5e-6
        help=(
            "Learning rate."
        )  # 参数说明
    )

    # 解析命令行传入的参数
    args = parser.parse_args()

    ###############################
    # 加载模型
    ###############################

    # 设置 PyTorch 的随机种子，确保可重复的结果
    torch.manual_seed(123)

    # 根据传入的模型类型加载对应的预训练模型
    if args.model == "distilbert":  # 如果选择的是 DistilBERT 模型
        # 加载预训练的 DistilBERT 模型，适用于二分类任务
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        # 替换模型的输出层，确保输出为 2 类的分类
        model.out_head = torch.nn.Linear(in_features=768, out_features=2)

        # 默认不训练任何模型参数
        for param in model.parameters():
            param.requires_grad = False

        # 根据传入的 --trainable_layers 参数决定哪些层需要训练
        if args.trainable_layers == "last_layer":
            for param in model.out_head.parameters():  # 只训练输出层
                param.requires_grad = True
        elif args.trainable_layers == "last_block":
            for param in model.pre_classifier.parameters():  # 训练预分类器
                param.requires_grad = True
            for param in model.distilbert.transformer.layer[-1].parameters():  # 训练最后一个变换器块的参数
                param.requires_grad = True
        elif args.trainable_layers == "all":
            for param in model.parameters():  # 训练所有层
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")  # 如果参数不合法则抛出异常

        # 加载 DistilBERT 的分词器
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    elif args.model == "bert":  # 如果选择的是 BERT 模型
        # 加载预训练的 BERT 模型，适用于二分类任务
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        # 替换 BERT 模型的分类器层
        model.classifier = torch.nn.Linear(in_features=768, out_features=2)

        # 默认不训练任何模型参数
        for param in model.parameters():
            param.requires_grad = False

        # 根据传入的 --trainable_layers 参数决定哪些层需要训练
        if args.trainable_layers == "last_layer":
            for param in model.classifier.parameters():  # 只训练分类器
                param.requires_grad = True
        elif args.trainable_layers == "last_block":
            for param in model.classifier.parameters():  # 训练分类器
                param.requires_grad = True
            for param in model.bert.pooler.dense.parameters():  # 训练 BERT 的池化层
                param.requires_grad = True
            for param in model.bert.encoder.layer[-1].parameters():  # 训练 BERT 最后一个编码层的参数
                param.requires_grad = True
        elif args.trainable_layers == "all":
            for param in model.parameters():  # 训练所有层
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")  # 如果参数不合法则抛出异常

        # 加载 BERT 的分词器
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    elif args.model == "roberta":  # 如果选择的是 RoBERTa 模型
        # 继续编写 RoBERTa 模型的加载代码...
        # 加载 RoBERTa-large 模型，并设置为二分类任务
        model = AutoModelForSequenceClassification.from_pretrained(
            "FacebookAI/roberta-large", num_labels=2  # 加载 Facebook 的 RoBERTa-large 预训练模型，设置标签数为 2
        )

        # 修改分类器的输出层，确保其输出维度为 2
        model.classifier.out_proj = torch.nn.Linear(in_features=1024, out_features=2)

        # 默认情况下，冻结模型的所有参数（不进行训练）
        for param in model.parameters():
            param.requires_grad = False

        # 根据 --trainable_layers 参数来决定哪些层可训练
        if args.trainable_layers == "last_layer":  # 如果选择只训练最后一层
            for param in model.classifier.parameters():  # 只训练分类器的参数
                param.requires_grad = True
        elif args.trainable_layers == "last_block":  # 如果选择训练最后一个块
            for param in model.classifier.parameters():  # 训练分类器的参数
                param.requires_grad = True
            for param in model.roberta.encoder.layer[-1].parameters():  # 训练 RoBERTa 的最后一层编码器
                param.requires_grad = True
        elif args.trainable_layers == "all":  # 如果选择训练所有层
            for param in model.parameters():  # 训练所有模型参数
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")  # 如果传入无效的参数，抛出异常

        # 加载 RoBERTa-large 的分词器
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")

        # 如果选择的模型无效，抛出异常
    else:
        raise ValueError("Selected --model {args.model} not supported.")

    # 设置设备为 GPU（如果有的话），否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将模型移动到指定的设备（GPU 或 CPU）
    model.to(device)

    # 将模型设置为评估模式，这样模型就不会进行训练，且不会计算梯度
    model.eval()

    ###############################
    # 实例化数据加载器
    ###############################

    # 定义基础路径为当前目录
    base_path = Path(".")

    # 判断是否使用 attention mask 参数
    if args.use_attention_mask.lower() == "true":
        use_attention_mask = True  # 如果为 "true"，使用 attention mask
    elif args.use_attention_mask.lower() == "false":
        use_attention_mask = False  # 如果为 "false"，不使用 attention mask
    else:
        raise ValueError("Invalid argument for `use_attention_mask`.")  # 如果参数无效，抛出异常

    # 创建训练、验证、测试数据集实例
    train_dataset = IMDBDataset(
        base_path / "train.csv",  # 训练数据文件路径
        max_length=256,  # 最大序列长度
        tokenizer=tokenizer,  # 使用的分词器
        pad_token_id=tokenizer.pad_token_id,  # 填充 token 的 ID
        use_attention_mask=use_attention_mask  # 是否使用 attention mask
    )

    val_dataset = IMDBDataset(
        base_path / "validation.csv",  # 验证数据文件路径
        max_length=256,  # 最大序列长度
        tokenizer=tokenizer,  # 使用的分词器
        pad_token_id=tokenizer.pad_token_id,  # 填充 token 的 ID
        use_attention_mask=use_attention_mask  # 是否使用 attention mask
    )

    test_dataset = IMDBDataset(
        base_path / "test.csv",  # 测试数据文件路径
        max_length=256,  # 最大序列长度
        tokenizer=tokenizer,  # 使用的分词器
        pad_token_id=tokenizer.pad_token_id,  # 填充 token 的 ID
        use_attention_mask=use_attention_mask  # 是否使用 attention mask
    )

    # 定义数据加载器的参数
    num_workers = 0  # 使用的工作进程数（此处为 0，表示不使用多进程）
    batch_size = 8  # 每批次的样本数

    # 创建训练数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,  # 使用训练数据集
        batch_size=batch_size,  # 设置批次大小
        shuffle=True,  # 数据随机打乱
        num_workers=num_workers,  # 工作进程数
        drop_last=True,  # 丢弃最后一批次（如果样本数不足）
    )

    # 创建验证数据加载器
    val_loader = DataLoader(
        dataset=val_dataset,  # 使用验证数据集
        batch_size=batch_size,  # 设置批次大小
        num_workers=num_workers,  # 工作进程数
        drop_last=False,  # 不丢弃最后一批次
    )

    # 创建测试数据加载器
    test_loader = DataLoader(
        dataset=test_dataset,  # 使用测试数据集
        batch_size=batch_size,  # 设置批次大小
        num_workers=num_workers,  # 工作进程数
        drop_last=False,  # 不丢弃最后一批次
    )

    ###############################
    # 训练模型
    ###############################

    # 记录训练开始时间
    start_time = time.time()

    # 设置随机种子，确保实验的可重复性
    torch.manual_seed(123)

    # 设置优化器为 AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    # 调用训练函数进行训练，并记录训练和验证损失、准确率等指标
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,  # 训练模型，使用数据加载器、优化器和设备
        num_epochs=args.num_epochs,  # 训练的轮次
        eval_freq=50,  # 每 50 步评估一次
        eval_iter=20,  # 每次评估使用 20 个样本
        max_steps=None  # 训练的最大步骤数（默认为 None，表示没有最大限制）
    )

    # 记录训练结束时间
    end_time = time.time()

    # 计算训练所用的时间，并输出训练时长
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    ###############################
    # 评估模型
    ###############################

    print("\nEvaluating on the full datasets ...\n")

    # 计算训练集、验证集和测试集的准确率
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    # 输出训练集、验证集和测试集的准确率
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
