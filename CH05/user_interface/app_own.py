# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from pathlib import Path  # 导入Path模块，用于处理文件路径
import sys  # 导入sys模块，用于操作系统相关的功能

import tiktoken  # 导入tiktoken，用于与GPT-2模型交互的分词器
import torch  # 导入PyTorch，用于深度学习相关操作
import chainlit  # 导入Chainlit，用于构建聊天对话界面

from previous_chapters import (
    generate,  # 导入生成文本的函数
    GPTModel,  # 导入自定义GPT模型
    text_to_token_ids,  # 导入将文本转换为token ID的函数
    token_ids_to_text,  # 导入将token ID转换回文本的函数
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU，若有则使用GPU，否则使用CPU


def get_model_and_tokenizer():
    """
    加载一个已在第5章训练的GPT-2模型及其预训练权重。
    这需要你首先运行第5章的代码，生成必要的model.pth文件。
    """

    # GPT模型配置，包含词汇大小、上下文长度、嵌入维度等
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # 词汇表大小
        "context_length": 256,  # 上下文长度（此处设置为256，原为1024）
        "emb_dim": 768,  # 嵌入维度
        "n_heads": 12,  # 注意力头的数量
        "n_layers": 12,  # 层数
        "drop_rate": 0.1,  # Dropout比率
        "qkv_bias": False  # 是否使用Query-Key-Value偏置
    }

    # 使用GPT-2的分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 模型路径
    model_path = Path("..") / "01_main-chapter-code" / "model.pth"

    # 检查模型路径是否存在，如果不存在，提示用户运行第5章的代码生成模型文件
    if not model_path.exists():
        print(
            f"Could not find the {model_path} file. Please run the chapter 5 code (ch05.ipynb) to generate the model.pth file.")
        sys.exit()  # 如果模型文件不存在，程序终止

    # 加载模型权重
    checkpoint = torch.load(model_path, weights_only=True)

    # 初始化模型，并加载权重
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint)
    model.to(device)  # 将模型移至GPU或CPU

    return tokenizer, model, GPT_CONFIG_124M  # 返回分词器、模型及模型配置


# 获取所需的分词器、模型及模型配置
tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    Chainlit的主函数，用于处理用户消息并生成模型的响应。
    """
    # 使用generate函数生成模型输出，generate函数内部已经包含了`torch.no_grad()`，以避免梯度计算
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(message.content, tokenizer).to(device),  # 将用户的文本转化为token ID
        max_new_tokens=50,  # 生成50个新的token
        context_size=model_config["context_length"],  # 上下文大小
        top_k=1,  # 采样时选择概率最大的token
        temperature=0.0  # 温度设置为0，确保生成的是确定性的文本
    )

    # 将生成的token ID转换为文本
    text = token_ids_to_text(token_ids, tokenizer)

    # 使用Chainlit返回生成的文本作为响应
    await chainlit.Message(
        content=f"{text}",  # 返回的模型生成文本
    ).send()  # 发送消息
