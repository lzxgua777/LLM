from importlib.metadata import version   # 从importlib.metadata模块导入version函数，用于获取库的版本
import torch    # 导入PyTorch库

print("torch version:", version("torch"))   # 打印torch库的版本

# 打开文件"number-data.txt"并写入1到1000的数字，每个数字后面跟一个空格
with open("number-data.txt", "w", encoding="utf-8") as f:
    for number in range(1001):  # 循环从0到1000
        f.write(f"{number} ")   # 将数字写入文件，并在每个数字后添加一个空格

from torch.utils.data import Dataset, DataLoader # 从torch.utils.data模块导入Dataset和DataLoader类

# 定义GPTDatasetV1类，继承自Dataset
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        # 初始化方法，接收文本、分词器、最大长度和步长作为参数
        self.input_ids = []      # 初始化输入ID列表
        self.target_ids = []     # 初始化目标ID列表

        # 修改部分
        # token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        # 使用分词器对文本进行编码
        token_ids = [int(i) for i in txt.strip().split()]   # 将文本文件中的数字转换为整数列表

        # 使用滑动窗口将文本分割成重叠的序列，每个序列长度为max_length
        for i in range(0, len(token_ids) - max_length, stride):     # 步长为stride
            input_chunk = token_ids[i:i + max_length]               # 输入序列
            target_chunk = token_ids[i + 1: i + max_length + 1]     # 目标序列
            self.input_ids.append(torch.tensor(input_chunk))        # 将输入序列添加到列表
            self.target_ids.append(torch.tensor(target_chunk))      # 将目标序列添加到列表

    # 返回数据集中的样本数量
    def __len__(self):
        return len(self.input_ids)

    # 获取数据集中的单个样本
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
        # 返回输入ID和目标ID的张量

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 初始化分词器，这里被注释掉了，因为代码中没有使用分词器
    # tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = None    # 分词器设置为None，因为在这个函数中没有使用

    # 创建数据集实例
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,   # 批次大小
        shuffle=shuffle,         # 是否打乱数据
        drop_last=drop_last,     # 是否丢弃最后不完整的批次
        num_workers=num_workers  # 加载数据时使用的子进程数量
    )

    # 返回数据加载器
    return dataloader

# 读取文本文件
with open("number-data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 使用创建的数据加载器类创建一个数据加载器实例
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

# 迭代数据加载器
data_iter = iter(dataloader)
first_batch = next(data_iter)  # 获取第一个批次
print(first_batch)  # 打印第一个批次

second_batch = next(data_iter)  # 获取第二个批次
print(second_batch)  # 打印第二个批次

third_batch = next(data_iter)  # 获取第三个批次
print(third_batch)  # 打印第三个批次

# 重置迭代器并获取最后一个批次
for batch in dataloader:
    pass

last_batch = batch  # last_batch变量现在包含最后一个批次
print(last_batch)  # 打印最后一个批次

# 创建一个新的数据加载器实例，这次批次大小为2，步长为4
dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=False)

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 初始化分词器，这里被注释掉了，因为代码中没有使用分词器
    # tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = None  # 分词器设置为None，因为在这个函数中没有使用

    # 创建数据集实例
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # 批次大小
        shuffle=shuffle,  # 是否打乱数据
        drop_last=drop_last,  # 是否丢弃最后不完整的批次
        num_workers=num_workers  # 加载数据时使用的子进程数量
    )

    # 返回数据加载器
    return dataloader

# 读取文本文件
with open("number-data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 使用创建的数据加载器类创建一个数据加载器实例
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

# 迭代数据加载器
data_iter = iter(dataloader)
first_batch = next(data_iter)  # 获取第一个批次
print(first_batch)  # 打印第一个批次

second_batch = next(data_iter)  # 获取第二个批次
print(second_batch)  # 打印第二个批次

third_batch = next(data_iter)  # 获取第三个批次
print(third_batch)  # 打印第三个批次

# 重置迭代器并获取最后一个批次
for batch in dataloader:
    pass
last_batch = batch  # last_batch变量现在包含最后一个批次
print(last_batch)  # 打印最后一个批次

# 创建一个新的数据加载器实例，这次批次大小为2，步长为4
dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=False)

# 迭代新的数据加载器，但不打印输出
for inputs, targets in dataloader:
    pass


# 打印最后一个批次的输入和目标
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

torch.manual_seed(123)  # 设置随机种子以确保结果的可重复性
# 创建一个新的数据加载器实例，这次打乱数据
dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=True)
# 迭代新的数据加载器，但不打印输出
for inputs, targets in dataloader:
    pass
# 打印最后一个批次的输入和目标
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)













