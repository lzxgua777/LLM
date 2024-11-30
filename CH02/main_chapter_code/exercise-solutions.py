from importlib.metadata import version              # 从importlib.metadata模块导入version函数

print("torch version:", version("torch"))           # 打印torch和tiktoken库的版本
print("tiktoken version:", version("tiktoken"))

# Exercise 2.1


import tiktoken                                     # 导入tiktoken库

tokenizer = tiktoken.get_encoding("gpt2")           # 获取gpt2的tokenizer
integers = tokenizer.encode("Akwirw ier")           # 对字符串"Akwirw ier"进行编码
print(integers)                                     # 打印编码后的整数列表


for i in integers:                                  # 遍历编码后的整数列表，并打印每个整数及其对应的解码字符串
    print(f"{i} -> {tokenizer.decode([i])}")

# 分别打印字符串"Ak", "w", "ir", "w", " ", "ier"的编码
print(tokenizer.encode("Ak"))
print(tokenizer.encode("w"))
print(tokenizer.encode("ir"))
print(tokenizer.encode("w"))
print(tokenizer.encode(" "))
print(tokenizer.encode("ier"))
# 打印整数列表[33901, 86, 343, 86, 220, 959]的解码字符串
print(tokenizer.decode([33901, 86, 343, 86, 220, 959]))

# Exercise 2.2

# 导入tiktoken和torch库，以及torch.utils.data模块中的Dataset和DataLoader类
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# 定义GPTDatasetV1类，继承自Dataset

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []             # 存储输入ID的列表
        self.target_ids = []            # 存储目标ID的列表

        # 对整个文本进行编码
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分割成重叠的序列，每个序列长度为max_length
        for i in range(0, len(token_ids) - max_length, stride):
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

# 定义create_dataloader函数，用于创建数据加载器
def create_dataloader(txt, batch_size=4, max_length=256, stride=128):
    # 初始化tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size)
    # 返回数据加载器
    return dataloader

# 打开文件"the-verdict.txt"并读取内容
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")       # 获取gpt2的tokenizer
encoded_text = tokenizer.encode(raw_text)       # 对原始文本进行编码
# 设置词汇表大小、输出维度和最大长度
vocab_size = 50257
output_dim = 256
max_len = 4
context_length = max_len
# 创建token嵌入层和位置嵌入层
token_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# 使用max_length=2和stride=2创建数据加载器
dataloader = create_dataloader(raw_text, batch_size=4, max_length=2, stride=2)

# 遍历数据加载器中的批次，并在第一个批次后跳出循环
for batch in dataloader:
    x, y = batch
    break

print(x)

# 使用max_length=8和stride=2创建数据加载器
dataloader = create_dataloader(raw_text, batch_size=4, max_length=8, stride=2)

# 遍历数据加载器中的批次，并在第一个批次后跳出循环
for batch in dataloader:
    x, y = batch
    break

print(x)












