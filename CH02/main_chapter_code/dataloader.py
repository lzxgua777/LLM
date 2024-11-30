# NBVAL_SKIP
from importlib.metadata import version              # 从importlib.metadata模块导入version函数

# 打印torch和tiktoken库的版本
print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))


# 导入tiktoken和torch库
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader    # 从torch.utils.data模块导入Dataset和DataLoader类


class GPTDatasetV1(Dataset):                        # 定义一个名为GPTDatasetV1的类，它继承自Datase
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []                         # 存储输入ID的列表
        self.target_ids = []                        # 存储目标ID的列表

        # 使用tokenizer对整个文本进行编码
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分割成长度为max_length的重叠序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]           # 输入序列
            target_chunk = token_ids[i + 1: i + max_length + 1] # 目标序列
            self.input_ids.append(torch.tensor(input_chunk))    # 将输入序列添加到列表
            self.target_ids.append(torch.tensor(target_chunk))  # 将目标序列添加到列表
    # 返回数据集中的样本数量
    def __len__(self):
        return len(self.input_ids)
    # 获取数据集中的单个样本
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# 定义一个函数，用于创建数据加载器
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 初始化tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


# 打开文件"the-verdict.txt"并读取内容
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")   # 获取gpt2的tokenizer
encoded_text = tokenizer.encode(raw_text)   # 对原始文本进行编码

# 设置词汇表大小、输出维度和上下文长度
vocab_size = 50257
output_dim = 256
context_length = 1024

# 创建token嵌入层和位置嵌入层
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# 设置最大长度
max_length = 4
# 创建数据加载器
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length)


# 遍历数据加载器中的批次
for batch in dataloader:
    x, y = batch

    # 获取token嵌入和位置嵌入
    token_embeddings = token_embedding_layer(x)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))

    # 将token嵌入和位置嵌入相加得到输入嵌入
    input_embeddings = token_embeddings + pos_embeddings

    # 跳出循环
    break


# 打印输入嵌入的形状
print(input_embeddings.shape)









