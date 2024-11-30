
# Using BPE from tiktoken

from importlib.metadata import version          # 导入version函数来获取tiktoken库的版本

print("tiktoken version:", version("tiktoken")) # 打印tiktoken库的版本

import tiktoken   # 导入tiktoken库

tik_tokenizer = tiktoken.get_encoding("gpt2")   # 使用tiktoken库获取GPT-2模型的编码器

text = "Hello, world. Is this-- a test?"        # 定义要编码的文本

integers = tik_tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# 使用tik_tokenizer对文本进行编码，允许的特殊符号包括"<|endoftext|>"

print(integers)                             # 打印编码后的整数序列

strings = tik_tokenizer.decode(integers)    # 使用tik_tokenizer对编码后的整数序列进行解码

print(strings)                              # 打印解码后的文本
print(tik_tokenizer.n_vocab)                # 打印词汇表大小

# Using the original BPE implementation used in GPT-2

from bpe_openai_gpt2 import get_encoder, download_vocab


download_vocab()    # 下载词汇表
# 获取原始的GPT-2模型编码器
orig_tokenizer = get_encoder(model_name="gpt2_model", models_dir=".")

integers = orig_tokenizer.encode(text)      # 使用原始编码器对文本进行编码

print(integers)                             # 打印编码后的整数序列

strings = orig_tokenizer.decode(integers)   # 使用原始编码器对编码后的整数序列进行解码

print(strings)                              # 打印解码后的文本

# Using the BPE via Hugging Face transformers
import transformers
# 使用Hugging Face transformers库中的BP

print(transformers.__version__)             # 打印transformers库的版本
from transformers import GPT2Tokenizer

hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2") # 从预训练模型中加载GPT-2分词器
print(hf_tokenizer(strings)["input_ids"])           # 使用Hugging Face分词器对之前解码的字符串进行编码，并打印输入ID
# A quick performance benchmark

# 性能测试，读取文本文件
with open('main-chapter-code/the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

import timeit

%timeit orig_tokenizer.encode(raw_text)  # 使用IPython的timeit魔法命令来测试原始编码器的编码性能
%timeit tik_tokenizer.encode(raw_text)   # 使用IPython的timeit魔法命令来测试tiktoken编码器的编码性能

%timeit hf_tokenizer(raw_text)["input_ids"]  # 使用IPython的timeit魔法命令来测试Hugging Face分词器的编码性能
# 使用IPython的timeit魔法命令来测试Hugging Face分词器在最大长度限制下的编码性能
%timeit hf_tokenizer(raw_text, max_length=5145, truncation=True)["input_ids"]

