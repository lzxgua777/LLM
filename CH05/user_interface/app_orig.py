# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import tiktoken  # 导入 tiktoken，用于与 GPT-2 模型交互的分词器
import torch  # 导入 PyTorch，用于处理深度学习任务
import chainlit  # 导入 chainlit，用于构建对话接口

from ch04.main_chapter_code.ch04 import (
    download_and_load_gpt2,  # 导入下载和加载 GPT-2 权重的函数
    generate,  # 导入生成文本的函数
    GPTModel,  # 导入自定义 GPT 模型
    load_weights_into_gpt,  # 导入将权重加载到 GPT 模型的函数
    text_to_token_ids,  # 导入将文本转换为 token ID 的函数
    token_ids_to_text,  # 导入将 token ID 转回文本的函数
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的 GPU，如果有则使用 GPU 否则使用 CPU


def get_model_and_tokenizer():
    """
    加载 GPT-2 模型及其分词器。
    如果模型不存在，代码会自动下载并加载预训练权重。
    """

    CHOOSE_MODEL = "gpt2-small (124M)"  # 可以选择其他模型，例如 gpt2-small, gpt2-medium, gpt2-large 等

    BASE_CONFIG = {
        "vocab_size": 50257,     # 词汇表大小
        "context_length": 1024,  # 上下文长度
        "drop_rate": 0.0,        # Dropout 比例
        "qkv_bias": True         # 是否使用 Query-Key-Value 偏置
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},  # 124M 参数配置
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},  # 355M 参数配置
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},  # 774M 参数配置
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},  # 1558M 参数配置
    }

    # 获取选择的模型的大小（例如 124M）
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    # 根据选择的模型更新配置
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # 下载并加载 GPT-2 模型的权重
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    # 初始化 GPT 模型，并加载权重
    gpt = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(gpt, params)  # 将下载的权重加载到模型中
    gpt.to(device)  # 将模型移动到设备（GPU 或 CPU）
    gpt.eval()  # 将模型设置为评估模式，禁用 Dropout

    tokenizer = tiktoken.get_encoding("gpt2")  # 使用 GPT-2 的分词器

    return tokenizer, gpt, BASE_CONFIG  # 返回分词器、模型和基础配置


# 获取所需的分词器和模型文件
tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    Chainlit 主函数，用于响应用户消息。
    """
    # 使用 generate 函数生成模型输出。该函数已内置禁用梯度计算（torch.no_grad()）。
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(message.content, tokenizer).to(device),  # 将用户输入的文本转换为 token IDs
        max_new_tokens=50,  # 生成 50 个新的 token
        context_size=model_config["context_length"],  # 上下文大小设置为模型的最大上下文长度
        top_k=1,  # 采样时选择概率最高的 token
        temperature=0.0  # 温度设置为 0.0 以确保输出是确定性的
    )

    # 将生成的 token IDs 转换为文本
    text = token_ids_to_text(token_ids, tokenizer)

    # 使用 Chainlit 返回生成的文本作为响应
    await chainlit.Message(
        content=f"{text}",  # 生成的模型输出文本
    ).send()  # 发送响应消息
