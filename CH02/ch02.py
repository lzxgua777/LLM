
from importlib.metadata import version  # 从importlib.metadata模块中导入version函数，用于获取已安装Python包的版本信息

import os
import urllib.request

print("torch version:", version("torch"))  # 获取torch的版本
print("tiktoken version:", version("tiktoken"))  # 获取tiktoken的版本

'加载我们要处理的原始文本'

#if not os.path.exists("the-verdict.txt"):      # 如果"the_verdict.txt"文件在当前目录下不存在，则执行下面的代码块
 #   url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
  #  file_path = "the-verdict.txt"              # 指定下载文件的保存路径和文件名
   # urllib.request.urlretrieve(url, file_path) # 从url指定的网址下载文件，并将其保存到file_path指定的路径
# 快速检查文件内容或进行初步的内容审查
with open("the-verdict.txt", "r", encoding="utf-8") as f:  # 用with语句打开名为"the-verdict.txt"的文件
    raw_text = f.read()  # 调用文件对象f的read()方法，读取文件内容并存储在raw_text中
    # "r"参数表示以只读模式打开文件
    # encoding="utf-8"指定文件的编码格式为UTF-8，这样可以正确读取包含特殊字符的文本文件
    # as f将打开的文件对象赋值给变量f

print("Total number of character:", len(raw_text))  # 打印出文件中字符的总数
print(raw_text[:99])  # 打印出文件开头的99个字符

'''
目标是将文本分词并嵌入到LLM中
让我们基于一些简单的示例文本开发一个简单的分词器，然后将其应用于上面的文本
下面的正则表达式将在空格上拆分
'''
import re  # 导入正则表达式模块re

text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)  # 使用re.split函数和正则表达式(\s)来分割字符串。(\s)匹配任何空白字符

print(result)

result = re.split(r'([,.]|\s)', text)  # ([,.]|\s)正则表达式匹配逗号'，'、句号'.'或任何空白字符

print(result)

'正如我们所看到的，这会创建空字符串，让我们删除它们'
result = [item for item in result if item.strip()]  # strip()方法用于移除字符串两端的空白字符
print(result)

'这看起来不错，但我们也应该处理其他类型的标点符号，如句号、问号等'
text = "Hello, world. Is this-- a test?"

result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

'这相当不错，我们现在准备将这种标记化应用于原始文本'
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])  # 打印preprocessed列表中的前30个元素
print(len(preprocessed))  # 打印preprocessed列表的长度，即元素的总数

all_words = sorted(set(preprocessed))
# 先将preprocessed列表转换成一个集合（set），集合是无序的、不包含重复元素的数据结构
# 再将集合转换成一个有序的列表（list），排序依据是自然顺序（对于字符串来说，一般是字典序）
vocab_size = len(all_words)  # 计算all_words列表的长度，即词汇表中不同单词的数量
print(vocab_size)  # 打印出词汇表的大小

vocab = {token: integer for integer, token in enumerate(all_words)}
# enumerate函数用于给列表、元组或字符串加上索引，会生成一个包含单词索引（从0开始）和单词本身的元组
# 遍历生成的元组，并创建一个新词典vocab，其中token（单词）作为键，integer（索引）作为值

for i, item in enumerate(vocab.items()):  # 打印词汇表的前50个条目
    print(item)
    if i >= 50:
        break

'''
SimpleTokenizerV1类提供了一个简单的文本解码和编码功能
'''


class SimpleTokenizerV1:
    def __init__(self, vocab):  # 构造函数
        self.str_to_int = vocab  # self.str_to_int存储了传入的词汇表，用于将单词转换为对应的整数索引
        self.int_to_str = {i: s for s, i in vocab.items()}  # self.int_to_str是上面的反向字典，用于将整数索引转换回单词

    def encode(self, text):  # encode方法用于将文本编码为整数序列
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)  # 分割文本

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()  # 去除空白，过滤掉空字符串
        ]
        ids = [self.str_to_int[s] for s in preprocessed]  # 将每个非空字符串元素转换为其对应的整数索引
        return ids  # 返回包含整数索引的列表

    def decode(self, ids):  # decode方法用于将整数序列解码回文本
        text = " ".join([self.int_to_str[i] for i in ids])  # 将单词列表连接成一个字符串，单词之间用空格分隔
        # 替换指定标点符号前的空格
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)  # r'\1'表示用第一个捕获组（即标点符号）替换匹配文本
        return text  # 返回解码后的文本


tokenizer = SimpleTokenizerV1(vocab)  # 创建一个实例，将vocab词汇表传递给构造函数__init__

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)  # 调用encode方法，将text字符串编码为一系列整数ID
print(ids)  # 打印出这个整数序列
print(tokenizer.decode(ids))  # 我们可以将整数解码回文本
print(tokenizer.decode(tokenizer.encode(text)))

#tokenizer = SimpleTokenizerV1(vocab)

#text = "Hello, do you like tea. Is this-- a test?"

#tokenizer.encode(text)
'''
上述操作会产生一个错误，因为词汇表中没有包含"Hello"这个词
为了应对这种情况，我们可以在词汇表中添加想"<|unk|>"这样的特殊符号来表示未知单词
既然我们已经在扩展词汇表了，让我们添加另一个称为"<|endoftext|>"的标记
它在GPT-2的训练中用于表示一个文本的结尾
'''
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])  # extend方法用于在列表末尾追加指定元素
vocab = {token: integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))

for i, item in enumerate(list(vocab.items())[-5:]):  # 遍历vocab中的最后5个条目
    print(item)
'''
我们还需要调整分词器
'''


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [  # 对于每个非空字符串
            item if item in self.str_to_int  # 检查是否在词汇表self.str_to_int中
            else "<|unk|>" for item in preprocessed  # 如果不在，则替换为<|unk|>
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


tokenizer = SimpleTokenizerV2(vocab)  # 初始化分词器

text1 = "Hello,do you like tea?"  # 定义了两个字符串变量
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))  # 将text1和text2连接
# 中间用特定分隔符" <|endoftext|> "分隔
print(text)  # 打印连接后的文本

print(tokenizer.encode(text))  # 将文本转换为一系列token ID
print(tokenizer.decode(tokenizer.encode(text)))  # 将token ID转换为原始文本

import importlib
import tiktoken  # 这是一个处理tokenizer的工具库

print("tiktoken version:", importlib.metadata.version("tiktoken"))  # 获取tiktoken的版本

tokenizer = tiktoken.get_encoding("gpt2")  # 使用函数获取名为"gpt2"的tokenizer
# 这里的gpt2指的是GPT-2模型的tokenizer，它能够处理GPT-2模型的编码和解码
text = (  # 定义了一个字符串变量，包含要处理的文本
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"  # 其中包含了特殊标记<|endoftext|>用于分隔两个句子
    "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})  # 使用tokenizer对text进行编码，生成一系列tokwn IDs
# allowed_special参数指定了特殊标记<|endoftext|>被允许在编码过程中使用
print(integers)

strings = tokenizer.decode(integers)  # 使用tokenizer对编码后的整数序列进行解码

print(strings)

with open("the-verdict.txt", "r", encoding="utf-8") as f:  # with语句确保文件在操作完成后会被正确关闭
    raw_text = f.read()  # 读取文件的全部内容，并将其存储在变量raw_text中

enc_text = tokenizer.encode(raw_text)  # 对raw_text编码，并将这些ID存储在enc_text中
print(len(enc_text))  # 打印编码后的token IDs序列的长度

enc_sample = enc_text[50:]  # 从enc_text中提取从第51个token开始到末尾的部分，作为编码样本

context_size = 4  # 定义上下文大小为4

x = enc_sample[:context_size]  # 从enc_sample中提取前4个token IDs作为x
y = enc_sample[1:context_size + 1]  # 从enc_sample中提取第2个到第5个token IDs作为y

print(f"x: {x}")
print(f"y:      {y}")
"""
对于每个文本块，我们想要输入和目标
由于我们希望模型预测下一个单词，因此目标是将输入向右移动一个位置
一个接一个，预测如下：
"""
for i in range(1, context_size + 1):  # 从1迭代到context_size（包含）
    context = enc_sample[:i]  # 对于每次迭代，context被设置为enc_sample的前i个token IDs
    desired = enc_sample[i]  # desired被设置为enc_sample中的第i个token ID
    # 这是当前上下文之后的下一个token。

    print(context, "---->", desired)

for i in range(1, context_size + 1):  # 这个循环与第一个循环类似
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
    # 这里使用tokenizer.decode方法将token IDs转换回原始文本字符串
    # desired被放在一个列表中，因为tokenizer.decode方法期望一个token ID列表作为输入
"""
在介绍注意力机制之后，我们将在后面的章节中讨论下一个词的预测。
目前，我们实现一个简单的数据加载器，该加载器对输入数据集进行迭代，并返回一个移位的输入和目标
"""
import torch  # 导入Pytorch库

print("PyTorch version:", torch.__version__)  # 获取Pytorch的版本

"创建dataset和dataloader，从输入文本数据集提取块"
from torch.utils.data import Dataset, DataLoader  # 从Pytorch库中导入了Dataset和DataLoader类

"""
Dataset是一个抽象类，用于定义数据集的结构和操作方法
DataLoader用于创建可迭代的数据加载器，它能够批量加载数据并可选地打乱数据顺序
定义了一个名为GPTDatasetV1的类，它继承自Dataset类
"""


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):  # 构造函数
        """

        :param txt: 要处理的文本
        :param tokenizer: 用于将文本编码为token ID的分词器
        :param max_length: 每个输入序列的最大长度
        :param stride: 滑动窗口的步长
        """
        self.input_ids = []  # 初始化两个空列表
        self.target_ids = []  # 用于存储输入和目标序列的token ID

        # 使用提供的分词器将整个文本编码为token ID序列
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分割成重叠的序列，每个序列的长度为 max_length。步长由 stride 参数控制
        for i in range(0, len(token_ids) - max_length, stride):
            # 对于每个窗口，创建一个输入序列 input_chunk 和一个目标序列 target_chunk。目标序列是输入序列的下一个token
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            # 将输入和目标序列的token ID转换为PyTorch张量，并将它们添加到相应的列表中
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):  # 返回数据集中的序列数量
        return len(self.input_ids)

    def __getitem__(self, idx):  # 接受一个索引idx并返回对应的输入和目标序列
        return self.input_ids[idx], self.target_ids[idx]


"""
定义了一个名为create_dataloader_v1的函数，它用于创建并返回一个DataLoader对象
这个对象可以用于批量加载和迭代数据集中的数据
"""


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    """

    :param txt: 要处理的文本数据
    :param batch_size: 每个批次中的样本数量，默认为4
    :param max_length: 每个输入序列的最大长度，默认为256
    :param stride: 滑动窗口的步长，默认为128
    :param shuffle: 是否在每个epoch开始时打乱数据，默认为True
    :param drop_last: 如果数据集不能被批次大小整除，是否丢弃最后不完整的批次，默认为True
    :param num_workers: 加载数据时使用的子进程数量，默认为0，意味着数据将在主进程中加载
    :return:
    """

    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建GPTDatasetV1的实例
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    # 返回创建的DataLoader对象
    return dataloader


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)  # 将dataloader对象转换为一个迭代器data_iter，因为DataLoader对象需要被迭代以获取批次数据
first_batch = next(data_iter)  # 使用next函数从迭代器data_iter中获取第一批数据，并将其存储在变量first_batch中
print(first_batch)  # 打印第一批数据的内容

second_batch = next(data_iter)
print(second_batch)  # 打印第二批数据的内容

# 这行代码再次调用 create_dataloader_v1 函数，但这次使用了不同的参数
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)  # 将新创建的 dataloader 对象转换为迭代器 data_iter
inputs, targets = next(data_iter)  # 获取第一批数据，由于batch_size=8，inputs和targets将包含8个序列的token ID张量
print("Inputs:\n", inputs)  # 分别打印输入序列和目标序列的内容
print("\nTargets:\n", targets)

input_ids = torch.tensor([2, 3, 5, 1])  # 创建一个包含token ID的PyTorch张量。这些ID将被用于查找嵌入层中的向量表示

vocab_size = 6  # 设置词汇表大小
output_dim = 3  # 设置输出维度

# 设置PyTorch的随机种子为123。这确保了每次运行代码时，嵌入层的初始权重都是相同的，这有助于结果的可重复性
torch.manual_seed(123)
# 创建一个Embedding层，它将词汇表中的每个词映射到一个固定维度的向量空间
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer.weight)  # 打印嵌入层的权重矩阵

print(embedding_layer(torch.tensor([3])))  # 要将具有 ID 3 的token转换为三维向量
"""
请注意，以上是嵌入层权重矩阵的第4行
为了嵌入上述所有四个input_ids值，我们这样做
"""
print(embedding_layer(input_ids))

vocab_size = 50257  # 设置词汇表大小
output_dim = 256  # 设置输出维度

# 创建一个Embeddind层，它将词汇表中的每个词映射到一个固定维度的向量空间
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4  # 设置序列的最大长度为4

# 调用 create_dataloader_v1 函数来创建一个 DataLoader 对象，用于批量加载和迭代数据集中的数据
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)  # 打印输入序列的token ID和它们的形状
print("\nInputs shape:\n", inputs.shape)  # nputs是一个三维张量，分别为批次大小、序列长度、特征维度

token_embeddings = token_embedding_layer(inputs)  # 使用token_embedding_layer将输入的token IDs转换为嵌入向量
print(token_embeddings.shape)  # 打印token嵌入的形状

context_length = max_length  # 设置 context_length 为 max_length，即序列的最大长度
"""
使用 pos_embedding_layer 将位置索引（从0到 max_length-1）转换为位置嵌入向量
orch.arange(max_length) 生成一个从0开始到 max_length-1 的整数序列
"""
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings  # 将 token 嵌入和位置嵌入相加，得到最终的输入嵌入
print(input_embeddings.shape)  # 打印最终输入嵌入的形状
