# 版权声明：Sebastian Raschka 根据 Apache License 2.0（参见 LICENSE.txt）持有版权。
# "Build a Large Language Model From Scratch" 的来源
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch

from pathlib import Path
import sys

import tiktoken
import torch
import chainlit

from ch06.main_chapter_code.ch06 import (
    classify_review,
    GPTModel
)

# 设置设备，如果GPU可用则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义一个函数，用于加载在第6章中微调过的GPT-2模型
def get_model_and_tokenizer():
    """
    加载第6章生成的微调后的GPT-2模型的代码。
    这需要您先运行第6章的代码，以生成必要的model.pth文件。
    """

    # 定义GPT模型配置参数
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # 词汇表大小
        "context_length": 1024,  # 上下文长度
        "emb_dim": 768,          # 嵌入维度
        "n_heads": 12,           # 注意力头数
        "n_layers": 12,          # 层数
        "drop_rate": 0.1,        # Dropout比率
        "qkv_bias": True         # Query-key-value偏置
    }

    # 加载GPT-2模型的分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 设置模型文件路径
    model_path = Path("..") / "01_main-chapter-code" / "review_classifier.pth"
    if not model_path.exists():
        print(
            f"找不到{model_path}文件。请运行第6章的代码（ch06.ipynb）以生成review_classifier.pth文件。"
        )
        sys.exit()

    # 实例化模型
    model = GPTModel(GPT_CONFIG_124M)

    # 将模型转换为分类器，如第6章6.5节所述
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=GPT_CONFIG_124M["emb_dim"], out_features=num_classes)

    # 然后加载模型权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # 返回分词器和模型
    return tokenizer, model

# 获取chainlit函数所需的分词器和模型文件
tokenizer, model = get_model_and_tokenizer()

# 定义Chainlit的主函数
@chainlit.on_message
async def main(message: chainlit.Message):
    """
    Chainlit的主函数。
    """
    user_input = message.content  # 获取用户输入

    # 使用classify_review函数对用户输入的评论进行分类
    label = classify_review(user_input, model, tokenizer, device, max_length=120)

    # 将模型的响应返回到界面
    await chainlit.Message(
        content=f"{label}",
    ).send()
