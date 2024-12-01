def format_input(entry):
    instruction_text = (
        f"<|user|>\n{entry['instruction']}"  # 格式化 instruction 部分，加上一个标记 <|user|>，以区分用户输入
    )

    input_text = f"\n{entry['input']}" if entry["input"] else ""  # 如果有输入，则将其添加到文本中

    return instruction_text + input_text  # 返回格式化后的 instruction 和 input

sample_data = [
    {'instruction': 'Identify the correct spelling of the following word.', 'input': 'Ocassion', 'output': "The correct spelling is 'Occasion.'"},
    {'instruction': "What is an antonym of 'complicated'?", 'input': '', 'output': "An antonym of 'complicated' is 'simple'."}
]

print(format_input(sample_data[0]))  # 打印第一个样本的格式化输入
print()
print(format_input(sample_data[1]))  # 打印第二个样本的格式化输入

import tiktoken
from torch.utils.data import Dataset

class InstructionDataset(Dataset):  # 定义一个继承自 Dataset 的类，用于加载数据
    def __init__(self, data, tokenizer):
        self.data = data  # 保存数据

        # 预先对文本进行标记化
        self.encoded_texts = []  # 存储编码后的文本
        for entry in data:

            ###################################################################
            # 新增: 使用 `format_input_phi` 函数并调整响应文本模板
            instruction_plus_input = format_input(entry)  # 格式化 instruction 和 input
            response_text = f"\n<|assistant|>:\n{entry['output']}"  # 格式化响应部分
            ###################################################################
            full_text = instruction_plus_input + response_text  # 合并 instruction、input 和 response
            self.encoded_texts.append(
                tokenizer.encode(full_text)  # 使用 tokenizer 对合并后的文本进行编码
            )

    def __getitem__(self, index):  # 获取指定索引的数据
        return self.encoded_texts[index]

    def __len__(self):  # 获取数据集的长度
        return len(self.data)


tokenizer = tiktoken.get_encoding("gpt2")  # 获取 gpt2 模型的 tokenizer

# 这是从原书第七章中复制的 `format_input` 函数
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"  # 格式化 instruction 部分，增加任务描述
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""  # 如果有输入，则格式化 input 部分

    return instruction_text + input_text  # 返回格式化后的 instruction 和 input


import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):  # 定义一个继承自 Dataset 的类
    def __init__(self, data, tokenizer):
        self.data = data  # 保存数据

        ##########################################################################################
        # 新增: 用于保存 instruction 长度的列表
        self.instruction_lengths = []  # 保存 instruction 部分的长度
        ##########################################################################################

        self.encoded_texts = []  # 存储编码后的文本

        for entry in data:
            instruction_plus_input = format_input(entry)  # 格式化 instruction 和 input
            response_text = f"\n\n### Response:\n{entry['output']}"  # 格式化响应部分
            full_text = instruction_plus_input + response_text  # 合并为完整文本

            self.encoded_texts.append(
                tokenizer.encode(full_text)  # 使用 tokenizer 对完整文本进行编码
            )

            ##########################################################################################
            # 新增: 收集 instruction 部分的长度
            instruction_length = len(tokenizer.encode(instruction_plus_input))  # 获取 instruction 部分的 token 长度
            self.instruction_lengths.append(instruction_length)  # 将长度添加到列表中
            ##########################################################################################

    def __getitem__(self, index):  # 获取指定索引的数据
        # 新增: 返回 instruction 长度和编码后的文本
        return self.instruction_lengths[index], self.encoded_texts[index]

    def __len__(self):  # 获取数据集的长度
        return len(self.data)


import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")  # 获取 gpt2 模型的 tokenizer


def custom_collate_fn(
        batch,
        pad_token_id=50256,
        ignore_index=-100,
        allowed_max_length=None,
        device="cpu"
):
    # 查找批次中最长的序列
    batch_max_length = max(len(item) + 1 for instruction_length, item in batch)  # 新增: 现在批次是一个元组，包含 instruction_length 和 item

    # 填充并准备输入和目标
    inputs_lst, targets_lst = [], []  # 存储输入和目标的列表

    for instruction_length, item in batch:  # 新增: 批次是一个元组，解包成 instruction_length 和 item
        new_item = item.copy()  # 复制 item
        # 添加一个 <|endoftext|> token
        new_item += [pad_token_id]
        # 填充序列至最大长度
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # 截取最后一个 token，用于 inputs
        targets = torch.tensor(padded[1:])  # 将目标左移一个位置，用于 targets

        # 将除第一个外所有 padding token 替换为 ignore_index
        mask = targets == pad_token_id  # 创建一个 mask，标记 padding token
        indices = torch.nonzero(mask).squeeze()  # 获取 padding token 的索引
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index  # 将除第一个外的 padding token 设置为 ignore_index

        ##########################################################################################
        # 新增: 将输入和 instruction 部分的 tokens 在目标中设置为 -100（忽略）
        targets[:instruction_length - 1] = -100  # 忽略掉 input 和 instruction 部分的目标
        ##########################################################################################

        # 可选地对序列进行截断，限制最大长度
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)  # 将处理后的输入添加到列表
        targets_lst.append(targets)  # 将处理后的目标添加到列表

    # 将输入和目标的列表转换为张量并转移到目标设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor  # 返回输入和目标的张量
sample_data = [
    {'instruction': "What is an antonym of 'complicated'?", 'input': '', 'output': "An antonym of 'complicated' is 'simple'."},  # 第一个样本：问题没有输入，只是一个简单的问答
    {'instruction': 'Sort the following list in alphabetical order.', 'input': 'Zebra, Elephant, Crocodile', 'output': 'Crocodile, Elephant, Zebra'},  # 第二个样本：输入一个混乱的字母顺序的列表，要求排序
    {'instruction': 'Arrange the given numbers in descending order.', 'input': '5, 12, 8, 3, 15', 'output': '15, 12, 8, 5, 3.'}  # 第三个样本：输入一组数字，要求按降序排列
]

from torch.utils.data import DataLoader  # 从 PyTorch 导入 DataLoader

train_dataset = InstructionDataset(sample_data, tokenizer)  # 创建一个训练数据集对象，使用之前定义的 InstructionDataset 类
train_loader = DataLoader(
    train_dataset,  # 使用创建的数据集
    batch_size=len(sample_data),  # 批次大小设置为数据集大小，即一次性加载所有数据
    collate_fn=custom_collate_fn,  # 使用自定义的 collate 函数来处理批次数据
    num_workers=0  # 设置数据加载的工作线程数
)

print("Train loader:")  # 输出训练数据加载器的批次信息
for inputs, targets in train_loader:  # 遍历数据加载器中的批次
    print(inputs.shape, targets.shape)  # 打印输入和目标张量的形状

print("Inputs:\n", inputs[1])  # 打印第二个输入样本的内容（index 1）

print("\n\nTargets:\n", targets[1])  # 打印第二个目标样本的内容（index 1）

print(tokenizer.decode(list(inputs[1])))  # 使用 tokenizer 解码输入张量，输出对应的文本

non_masked_targets = targets[1][targets[1] != -100]  # 过滤掉被遮蔽的部分（targets 中值为 -100 的部分）

print(tokenizer.decode(list(non_masked_targets)))  # 解码并输出过滤后的目标文本
