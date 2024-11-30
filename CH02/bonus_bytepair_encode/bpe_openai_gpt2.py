# 来源：https://github.com/openai/gpt-2/blob/master/src/encoder.py
# 许可证：
# 修改后的MIT许可证

# Software Copyright (c) 2019 OpenAI

# 我们不声称拥有您使用GPT-2创造的内容的所有权，所以它是您的，您可以随意使用。
# 我们只要求您负责任地使用GPT-2，并清楚地表明您的内容是使用GPT-2创建的。

# 特此免费授予任何获得此软件和相关文档文件（“软件”）的个人，无限制地处理软件的权利，

# 包括但不限于使用、复制、修改、合并、出版、发行、再授权和/或出售软件副本的权利，
# 并允许向其提供软件的人员这样做，前提是满足以下条件：

# 以上版权声明和此权限声明必须包含在软件的所有副本或实质部分中。
# 以上版权声明和此权限声明不需要包含在软件创建的内容中。


# 软件按“原样”提供，不提供任何形式的明示或暗示担保，
# 包括但不限于对适销性、特定用途的适用性和非侵权性的担保。
# 在任何情况下，即使在合同行为、侵权行为或其他行为中，
# 作者或版权持有人也不对任何索赔、损害或其他责任负责，无论是直接的、间接的还是附带的，
# 因软件或软件的使用或其他交易而引起的或与之相关的。


import os                        # 导入os模块，提供操作系统相关功能
import json                      # 导入json模块，用于处理JSON数据
import regex as re               # 导入regex模块，并重命名为re，用于正则表达式操作
import requests                  # 导入requests模块，用于发起网络请求
from tqdm import tqdm            # 从tqdm模块导入tqdm，用于显示进度条
from functools import lru_cache  # 从functools模块导入lru_cache，用于缓存函数的结果


@lru_cache()                     # 使用lru_cache装饰器，缓存函数的结果，避免重复计算
def bytes_to_unicode():
    """
    返回utf-8字节的列表和对应的Unicode字符串列表。
    可逆的bpe代码在Unicode字符串上工作。
    这意味着如果您想避免UNKs（未知字符），您的词汇表中需要大量的Unicode字符。
    当您处理的是大约100亿个token的数据集时，您最终需要大约5000个字符来获得不错的覆盖率。
    这占您正常的32K bpe词汇表的一个相当大的比例。
    为了避免这种情况，我们希望建立utf-8字节和Unicode字符串之间的查找表。
    并避免映射到bpe代码无法处理的空白字符/控制字符。
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    # 创建一个包含ASCII和一些扩展拉丁字符的字节列表
    cs = bs[:]                      # 创建一个bs的副本
    n = 0                           # 初始化一个计数器
    for b in range(2 ** 8):         # 遍历0到255的所有字节值
        if b not in bs:             # 如果字节值不在bs中
            bs.append(b)            # 添加到bs列表
            cs.append(2 ** 8 + n)   # 添加到cs列表
            n += 1                  # 计数器加1
    cs = [chr(n) for n in cs]       # 将cs列表中的数字转换为对应的Unicode字符
    return dict(zip(bs, cs))        # 返回一个字典，将bs中的字节映射到cs中的Unicode字符


def get_pairs(word):
    """
     返回一个单词中的符号对集合。
    单词被表示为符号的元组（符号是可变长度的字符串）。
    """
    pairs = set()                     # 创建一个空集合，用于存储符号对
    prev_char = word[0]               # 获取单词的第一个字符
    for char in word[1:]:             # 遍历单词的其余字符
        pairs.add((prev_char, char))  # 将前一个字符和当前字符组成的对添加到集合中
        prev_char = char              # 更新前一个字符为当前字符
    return pairs                      # 返回符号对集合


class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder                                              # 编码器字典
        self.decoder = {v: k for k, v in self.encoder.items()}              # 解码器字典，编码器的逆字典
        self.errors = errors                                                # 解码时错误的处理方式，默认为替换
        self.byte_encoder = bytes_to_unicode()                              # UTF-8字节到Unicode字符串的编码器
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}    # Unicode字符串到UTF-8字节的解码器
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))      # BPE合并的排名字典，用于确定合并的顺序
        self.cache = {}                                                     # 缓存字典，用于存储已处理的BPE结果

        # 编译一个正则表达式，用于匹配单词中的各种模式，包括缩写、字母、数字和非字母数字字符
        # 同时匹配一个或多个空白字符，以及单独的空白字符
        # 应该添加re.IGNORECASE以使BPE合并可以适用于缩写的首字母大写版本
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:             # 如果token已经在缓存中，则直接返回缓存的结果
            return self.cache[token]
        word = tuple(token)                 # 将token转换为元组
        pairs = get_pairs(word)             # 获取单词中的所有符号对


        if not pairs:                       # 如果没有符号对，说明单词已经不能再被合并，直接返回原token
            return token
        # 循环直到无法再合并
        while True:                         # 找到排名最小的符号对，即优先合并的符号对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            # 如果这个符号对没有在bpe_ranks中，则跳出循环
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # 遍历单词中的每个符号
            while i < len(word):
                try:
                    # 尝试找到第一个符号在单词中的索引
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    # 如果找不到，将剩余的符号添加到new_word中，并跳出循环
                    new_word.extend(word[i:])
                    break
                # 如果当前符号和下一个符号组成了要合并的符号对
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            # 如果单词中只有一个符号了，说明无法再合并，跳出循环
            if len(word) == 1:
                break
            else:
                # 重新计算符号对
                pairs = get_pairs(word)
        # 将合并后的单词转换为字符串，并用空格连接
        word = ' '.join(word)
        self.cache[token] = word     # 将结果缓存起来
        return word

    def encode(self, text):     # 初始化一个空列表，用于存储BPE编码后的token
        bpe_tokens = []         # 使用正则表达式pat找到文本中的所有token
        for token in re.findall(self.pat, text):    # 将token编码为utf-8字节，然后使用byte_encoder转换为Unicode字符串
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # 对每个token应用BPE算法，然后将结果分割为单词
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
            # 返回BPE编码后的token列表
        return bpe_tokens

    def decode(self, tokens):   # 将token列表转换为字符串，使用decoder进行解码
        text = ''.join([self.decoder[token] for token in tokens])
        # 将解码后的字符串转换为字节数组，然后使用byte_decoder转换回Unicode字符串
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        # 返回解码后的文本
        return text


def get_encoder(model_name, models_dir): # 打开模型目录中指定模型的encoder.json文件，并加载为JSON对象
    with open(os.path.join(models_dir, model_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    # 打开模型目录中指定模型的vocab.bpe文件，并读取内容
    with open(os.path.join(models_dir, model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    # 处理vocab.bpe文件内容，提取BPE合并操作
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    # 创建并返回Encoder对象
    return Encoder(encoder=encoder, bpe_merges=bpe_merges)


def download_vocab():
    # 此代码段修改自其他代码
    subdir = 'gpt2_model'           # 设置子目录名称为'gpt2_model'
    if not os.path.exists(subdir):  # 检查子目录是否存在
        os.makedirs(subdir)         # 如果不存在，则创建子目录
    subdir = subdir.replace('\\', '/')  # needed for Windows
    # 将路径中的反斜杠替换为斜杠，适用于Windows系统

    for filename in ['encoder.json', 'vocab.bpe']:  # 遍历文件名列表
        r = requests.get("https://openaipublic.blob.core.windows.net/gpt-2/models/117M/" + filename, stream=True)
        # 发起网络请求，下载文件
        with open(os.path.join(subdir, filename), 'wb') as f: # 打开文件以写入二进制数据
            file_size = int(r.headers["content-length"])      # 获取文件大小
            chunk_size = 1000       # 设置每次写入的块大小为1000字节
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 使用tqdm显示进度条
                # 1k为chunk_size，因为以太网数据包大小约为1500字节
                for chunk in r.iter_content(chunk_size=chunk_size): # 遍历下载的文件内容
                    f.write(chunk)   # 将文件内容写入文件
                    pbar.update(chunk_size) # 更新进度条