from importlib.metadata import version  # 导入版本检查模块

from IPython import get_ipython  # 导入IPython模块，用于获取当前的IPython实例

pkgs = ["matplotlib", "numpy", "tiktoken", "torch", "tensorflow", "pandas"]  # 定义需要检查版本的包列表
for p in pkgs:  # 遍历包列表并打印每个包的版本
    print(f"{p} version: {version(p)}")  # 打印包的版本号

# 定义一个装饰器，用于防止某些单元格被执行两次
from IPython.core.magic import register_line_cell_magic  # 导入IPython的魔法命令注册模块

executed_cells = set()  # 创建一个集合，用于存储已经执行过的单元格

@register_line_cell_magic  # 注册一个魔法命令
def run_once(line, cell):  # 定义魔法命令的函数
    if line not in executed_cells:  # 如果当前单元格没有被执行过
        get_ipython().run_cell(cell)  # 执行当前单元格
        executed_cells.add(line)  # 将当前单元格添加到已执行集合中
    else:
        print(f"Cell '{line}' has already been executed.")  # 提示当前单元格已执行过

import urllib.request  # 导入urllib请求模块
import zipfile  # 导入zip文件处理模块
import os  # 导入操作系统接口模块
from pathlib import Path  # 导入路径操作模块

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"  # 数据集的URL地址
zip_path = "sms_spam_collection.zip"  # zip文件的保存路径
extracted_path = "sms_spam_collection"  # 解压后的文件夹名称
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"  # 数据文件的保存路径

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):  # 定义下载和解压数据集的函数
    if data_file_path.exists():  # 如果数据文件已存在
        print(f"{data_file_path} already exists. Skipping download and extraction.")  # 提示跳过下载和解压
        return

    # 下载文件
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # 解压文件
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # 添加.tsv文件扩展名
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)  # 重命名文件
    print(f"File downloaded and saved as {data_file_path}")  # 提示文件下载和保存成功

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)  # 调用函数下载和解压数据集

import pandas as pd  # 导入pandas库

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])  # 读取数据文件
df  # 显示数据框架

print(df["Label"].value_counts())  # 打印标签的计数

# 创建平衡数据集的函数
def create_balanced_dataset(df):  # 定义创建平衡数据集的函数
    # 计算"spam"的实例数量
    num_spam = df[df["Label"] == "spam"].shape[0]

    # 随机抽样"ham"实例以匹配"spam"实例的数量
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # 将"ham"子集与"spam"合并
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df  # 返回平衡的数据集

balanced_df = create_balanced_dataset(df)  # 创建平衡的数据集
print(balanced_df["Label"].value_counts())  # 打印平衡数据集中标签的计数

# 对标签进行映射
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# 定义随机分割数据集的函数
def random_split(df, train_frac, validation_frac):  # 定义随机分割数据集的函数
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

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)  # 随机分割数据集
# 测试集大小被暗示为0.2作为剩余部分

train_df.to_csv("train.csv", index=None)  # 保存训练数据集
validation_df.to_csv("validation.csv", index=None)  # 保存验证数据集
test_df.to_csv("test.csv", index=None)  # 保存测试数据集

import tiktoken  # 导入Tiktoken库

tokenizer = tiktoken.get_encoding("gpt2")  # 获取GPT-2的编码器
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))  # 编码特殊标记

import torch  # 导入PyTorch库
from torch.utils.data import Dataset  # 导入PyTorch的数据集模块

# 定义Spam数据集类
class SpamDataset(Dataset):  # 继承自PyTorch的Dataset类
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):  # 初始化函数
        self.data = pd.read_csv(csv_file)  # 读取CSV文件

        # 预标记化文本
        self.encoded_texts = [  # 编码文本列表
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:  # 如果没有指定最大长度
            self.max_length = self._longest_encoded_length()  # 获取最长编码长度
        else:
            self.max_length = max_length  # 设置最大长度
            # 截断序列如果它们超过最大长度
            self.encoded_texts = [  # 截断编码文本列表
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]

        # 将序列填充到最长序列
        self.encoded_texts = [  # 填充编码文本列表
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):  # 获取项目的方法
        encoded = self.encoded_texts[index]  # 获取编码文本
        label = self.data.iloc[index]["Label"]  # 获取标签
        return (  # 返回编码文本和标签的张量
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):  # 获取长度的方法
        return len(self.data)  # 返回数据的长度

    def _longest_encoded_length(self):  # 获取最长编码长度的方法
        max_length = 0  # 初始化最大长度
        for encoded_text in self.encoded_texts:  # 遍历编码文本列表
            encoded_length = len(encoded_text)  # 获取编码长度
            if encoded_length > max_length:  # 如果当前编码长度大于最大长度
                max_length = encoded_length  # 更新最大长度
        return max_length  # 返回最大长度
# 加载训练集数据，创建SpamDataset对象
train_dataset = SpamDataset(
    csv_file="train.csv",  # 训练数据的CSV文件路径
    max_length=None,       # 未指定最大长度，允许自动调整
    tokenizer=tokenizer    # 指定的分词器
)

# 打印训练集数据集的最大长度
print(train_dataset.max_length)

# 使用训练集的最大长度加载验证集数据
val_dataset = SpamDataset(
    csv_file="validation.csv",   # 验证数据的CSV文件路径
    max_length=train_dataset.max_length,  # 使用训练集的最大长度
    tokenizer=tokenizer         # 指定的分词器
)

# 使用训练集的最大长度加载测试集数据
test_dataset = SpamDataset(
    csv_file="test.csv",        # 测试数据的CSV文件路径
    max_length=train_dataset.max_length,  # 使用训练集的最大长度
    tokenizer=tokenizer         # 指定的分词器
)

# 导入DataLoader模块，用于创建批量数据加载器
from torch.utils.data import DataLoader

# 设置DataLoader的参数
num_workers = 0  # 设置为0以禁用多线程数据加载
batch_size = 8   # 每个批次的数据量

# 设置随机种子，确保数据加载的随机性可复现
torch.manual_seed(123)

# 创建训练数据的DataLoader
train_loader = DataLoader(
    dataset=train_dataset,   # 指定训练数据集
    batch_size=batch_size,   # 批量大小
    shuffle=True,            # 是否对数据进行随机打乱
    num_workers=num_workers, # 工作线程数量
    drop_last=True,          # 是否丢弃最后一个不满batch_size的批次
)

# 创建验证数据的DataLoader
val_loader = DataLoader(
    dataset=val_dataset,     # 指定验证数据集
    batch_size=batch_size,   # 批量大小
    num_workers=num_workers, # 工作线程数量
    drop_last=False,         # 不丢弃最后一个不满batch_size的批次
)

# 创建测试数据的DataLoader
test_loader = DataLoader(
    dataset=test_dataset,    # 指定测试数据集
    batch_size=batch_size,   # 批量大小
    num_workers=num_workers, # 工作线程数量
    drop_last=False,         # 不丢弃最后一个不满batch_size的批次
)

# 打印训练加载器的基本信息
print("Train loader:")

# 遍历训练加载器，打印每个批次的数据维度信息
for input_batch, target_batch in train_loader:
    pass  # 仅遍历，不执行操作

# 打印输入数据批次的维度
print("Input batch dimensions:", input_batch.shape)

# 打印目标标签批次的维度
print("Label batch dimensions", target_batch.shape)

# 打印训练、验证和测试集的批次数量
print(f"{len(train_loader)} training batches")  # 打印训练数据的批次数量
print(f"{len(val_loader)} validation batches")  # 打印验证数据的批次数量
print(f"{len(test_loader)} test batches")       # 打印测试数据的批次数量

# 模型选择：GPT-2的一个具体版本
CHOOSE_MODEL = "gpt2-small (124M)"

# 输入提示词，用于文本生成任务
INPUT_PROMPT = "Every effort moves"

# 基础配置字典
BASE_CONFIG = {
    "vocab_size": 50257,     # GPT-2模型的词汇表大小
    "context_length": 1024,  # 模型的上下文长度
    "drop_rate": 0.0,        # Dropout的比例
    "qkv_bias": True         # Query-Key-Value的偏置设置
}

# 不同规模的GPT-2模型配置
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},  # 小模型
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}, # 中等模型
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},  # 大模型
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},   # 超大模型
}

# 更新基础配置为所选择模型的具体参数
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# 确保数据集的最大长度不超过模型的上下文长度
assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context "
    f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
    f"`max_length={BASE_CONFIG['context_length']}`"
)

# 导入下载和加载GPT-2权重的函数
from ch05.main_chapter_code.gpt_download import download_and_load_gpt2

# 导入GPT模型的定义
from ch04.main_chapter_code.ch04 import GPTModel

# 导入加载权重的函数
from ch05.main_chapter_code.gpt_generate import load_weights_into_gpt

# 提取模型的规模
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

# 下载和加载GPT-2的预训练参数
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

# 初始化GPT-2模型
model = GPTModel(BASE_CONFIG)

# 将下载的权重加载到模型中
load_weights_into_gpt(model, params)

# 将模型设置为评估模式
model.eval()

# 导入文本生成和编码解码的相关函数
from ch05.main_chapter_code.gpt_train import (
    generate_text_simple,   # 简单文本生成函数
    text_to_token_ids,      # 文本转换为token ID的函数
    token_ids_to_text       # token ID转换为文本的函数
)

# 输入的提示词，用于生成新文本
text_1 = "Every effort moves you"

# 使用generate_text_simple函数生成文本
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),  # 将输入文本转化为token ID
    max_new_tokens=15,                        # 生成最多15个新token
    context_size=BASE_CONFIG["context_length"]  # 上下文长度
)

# 打印生成的文本结果
print(token_ids_to_text(token_ids, tokenizer))  # 将生成的token ID转回文本并打印
# 生成问题文本，用于测试模型的生成能力
text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)

# 使用`generate_text_simple`生成新的文本序列
token_ids = generate_text_simple(
    model=model,  # 使用之前定义的GPT模型
    idx=text_to_token_ids(text_2, tokenizer),  # 将输入文本转换为token ID
    max_new_tokens=23,  # 设置生成的新token数量
    context_size=BASE_CONFIG["context_length"]  # 指定上下文长度
)

# 打印生成的文本
print(token_ids_to_text(token_ids, tokenizer))

# 打印模型信息
print(model)

# 冻结所有模型参数（不再更新权重）
for param in model.parameters():
    param.requires_grad = False

# 设置随机种子，确保结果可复现
torch.manual_seed(123)

# 定义分类任务输出的类别数量（2类：spam和非spam）
num_classes = 2

# 替换模型的输出头部为线性层，用于分类任务
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],  # 输入特征维度等于模型的嵌入维度
    out_features=num_classes  # 输出类别数
)

# 仅对模型最后一层的Transformer Block参数进行训练
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

# 允许训练模型的最终归一化层
for param in model.final_norm.parameters():
    param.requires_grad = True

# 输入示例文本，用于查看模型输出
inputs = tokenizer.encode("Do you have time")  # 将文本编码为token ID
inputs = torch.tensor(inputs).unsqueeze(0)  # 添加batch维度

# 打印输入数据和形状
print("Inputs:", inputs)
print("Inputs dimensions:", inputs.shape)  # 形状为 (batch_size, num_tokens)

# 禁用梯度计算以提升推理效率
with torch.no_grad():
    outputs = model(inputs)  # 模型前向传播，计算输出

# 打印模型的输出和形状
print("Outputs:\n", outputs)
print("Outputs dimensions:", outputs.shape)  # 形状为 (batch_size, num_tokens, num_classes)

# 提取最后一个token的输出
print("Last output token:", outputs[:, -1, :])

# 计算最后一个token的概率分布
probas = torch.softmax(outputs[:, -1, :], dim=-1)  # 对logits应用Softmax计算概率
label = torch.argmax(probas)  # 获取概率最大的类别索引
print("Class label:", label.item())  # 打印分类结果

# 直接基于logits输出计算类别
logits = outputs[:, -1, :]
label = torch.argmax(logits)
print("Class label:", label.item())

# 定义计算模型在数据加载器上的准确率
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()  # 设置模型为评估模式
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)  # 如果未指定，使用数据加载器的全部批次
    else:
        num_batches = min(num_batches, len(data_loader))  # 限制批次数量
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():  # 禁用梯度计算
                logits = model(input_batch)[:, -1, :]  # 提取最后一个token的logits
            predicted_labels = torch.argmax(logits, dim=-1)  # 预测标签

            num_examples += predicted_labels.shape[0]  # 累加样本总数
            correct_predictions += (predicted_labels == target_batch).sum().item()  # 累加正确预测数
        else:
            break
    return correct_predictions / num_examples  # 返回准确率

# 检查可用设备（GPU、MPS 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型移动到指定设备
model.to(device)

# 设置随机种子，保证结果可复现
torch.manual_seed(123)

# 计算训练、验证和测试集的准确率
train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

# 打印训练、验证和测试集的准确率
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# 计算单个批次的交叉熵损失
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # 提取最后一个token的logits
    loss = torch.nn.functional.cross_entropy(logits, target_batch)  # 计算交叉熵损失
    return loss

# 计算整个数据加载器的平均损失
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")  # 数据加载器为空返回NaN
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算单批次损失
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches  # 返回平均损失

# 禁用梯度，计算训练、验证和测试集的平均损失
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

# 打印平均损失
print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")

# 定义一个简单的分类模型训练函数
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # 初始化列表，用于跟踪损失和准确率
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1  # 跟踪已处理的样本数和全局步数

    # 主训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式

        for input_batch, target_batch in train_loader:  # 遍历训练数据加载器
            optimizer.zero_grad()  # 重置梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算当前批次损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 使用优化器更新模型参数
            examples_seen += input_batch.shape[0]  # 增加已处理样本数
            global_step += 1  # 更新全局步数

            # 可选的评估步骤
            if global_step % eval_freq == 0:  # 每隔`eval_freq`步进行评估
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 每个epoch后计算准确率
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen  # 返回训练过程中的记录

# 定义一个评估模型的函数
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)  # 计算训练损失
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)  # 计算验证损失
    model.train()  # 恢复为训练模式
    return train_loss, val_loss  # 返回损失

# 记录开始时间
import time
start_time = time.time()

# 设置随机种子以保证结果可复现
torch.manual_seed(123)

# 定义优化器，使用AdamW优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

# 设置训练轮数
num_epochs = 5

# 调用训练函数，记录损失和准确率
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
)

# 记录结束时间并计算总耗时
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# 导入Matplotlib用于绘图
import matplotlib.pyplot as plt

# 定义一个函数，用于绘制训练和验证的值
def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 绘制训练和验证的值（如损失或准确率）与epoch的关系
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")  # 设置x轴为epoch
    ax1.set_ylabel(label.capitalize())  # 设置y轴为对应的值
    ax1.legend()  # 显示图例

    # 创建第二个x轴，用于显示已处理的样本数
    ax2 = ax1.twiny()  # 创建一个共享y轴的第二x轴
    ax2.plot(examples_seen, train_values, alpha=0)  # 隐藏图形，仅用于对齐刻度
    ax2.set_xlabel("Examples seen")  # 设置x轴为已处理的样本数

    fig.tight_layout()  # 自动调整布局
    plt.savefig(f"{label}-plot.pdf")  # 保存图像为PDF文件
    plt.show()  # 显示图像

# 生成用于绘图的epoch和样本数张量
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

# 绘制损失图
plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

# 重新生成张量用于准确率绘图
epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

# 绘制准确率图
plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

# 计算训练、验证和测试集的准确率
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

# 打印准确率
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# 定义一个函数用于对文本进行分类
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()  # 设置模型为评估模式

    # 准备输入数据
    input_ids = tokenizer.encode(text)  # 将文本编码为token ID
    supported_context_length = model.pos_emb.weight.shape[0]  # 获取模型支持的上下文长度

    # 如果输入序列太长，则截断
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # 如果输入序列太短，则填充
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # 添加批次维度

    # 模型推理
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # 获取最后一个token的logits
    predicted_label = torch.argmax(logits, dim=-1).item()  # 获取预测的类别

    # 返回分类结果
    return "spam" if predicted_label == 1 else "not spam"

# 示例文本1：垃圾邮件
text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

# 对文本1进行分类
print(classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))

# 示例文本2：正常邮件
text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

# 对文本2进行分类
print(classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))

torch.save(model.state_dict(), "review_classifier.pth")  # 保存模型的权重到文件
# model.state_dict() 获取模型的参数字典，包含了模型中所有参数的键值对
# "review_classifier.pth" 是保存模型权重的文件名

model_state_dict = torch.load("review_classifier.pth", map_location=device, weights_only=True)  # 从文件加载模型权重
# torch.load() 用于加载之前保存的模型权重
# "review_classifier.pth" 是包含模型权重的文件名
# map_location=device 指定了权重将被加载到哪个设备上，例如GPU或CPU
# weights_only=True 表示只加载权重，不加载优化器状态等其他信息

model.load_state_dict(model_state_dict)  # 将加载的权重应用到模型
# model.load_state_dict() 用于将加载的参数字典应用到模型
# model_state_dict 是从文件中加载的参数字典








