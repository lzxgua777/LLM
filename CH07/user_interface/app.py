from pathlib import Path  # 导入Path模块用于文件路径操作
import sys  # 导入系统操作模块

import tiktoken  # 导入tokenizer处理库
import torch  # 导入PyTorch深度学习库
import chainlit  # 导入Chainlit框架

from ch05.main_chapter_code.ch05 import (  # 从第五章的代码中导入相关函数
    generate,
    GPTModel,
    text_to_token_ids,
    token_ids_to_text,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为GPU（如果可用），否则为CPU


def get_model_and_tokenizer():  # 定义函数加载GPT-2模型和tokenizer
    """
    加载第七章生成的微调过的GPT-2模型和tokenizer。
    需要先运行第七章的代码，生成必要的gpt2-medium355M-sft.pth文件。
    """
    GPT_CONFIG_355M = {  # 定义GPT-2模型的配置参数
        "vocab_size": 50257,  # 词汇表大小
        "context_length": 1024,  # 缩短的上下文长度（原：1024）
        "emb_dim": 1024,  # 嵌入维度
        "n_heads": 16,  # 注意力头数
        "n_layers": 24,  # 层数
        "drop_rate": 0.0,  # 丢弃率
        "qkv_bias": True  # Query-key-value偏置
    }

    tokenizer = tiktoken.get_encoding("gpt2")  # 使用tiktoken库获取GPT-2的tokenizer
    model_path = Path("..") / "01_main-chapter-code" / "gpt2-medium355M-sft.pth"  # 设置模型文件的路径
    if not model_path.exists():  # 检查文件是否存在
        print(f"找不到{model_path}文件。请运行第七章的代码（ch07.ipynb）来生成gpt2-medium355M-sft.pt文件。")
        sys.exit()  # 如果文件不存在，则退出程序

    checkpoint = torch.load(model_path, weights_only=True)  # 加载模型权重
    model = GPTModel(GPT_CONFIG_355M)  # 创建GPTModel实例
    model.load_state_dict(checkpoint)  # 加载权重
    model.to(device)  # 将模型移动到指定设备

    return tokenizer, model, GPT_CONFIG_355M  # 返回tokenizer、模型和模型配置


def extract_response(response_text, input_text):  # 定义函数从响应文本中提取实际的回复内容
    return response_text[len(input_text):].replace("### Response:", "").strip()


# 获取Chainlit函数所需的tokenizer和模型文件
tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message  # 定义Chainlit消息处理函数
async def main(message: chainlit.Message):
    """
    Chainlit的主函数。
    """
    torch.manual_seed(123)  # 设置PyTorch的随机种子以确保结果的可重复性

    prompt = f"""以下是描述任务的指令。写一个适当完成请求的回应。

    ### 指令：
    {message.content}
    """  # 构造一个包含用户消息内容的提示

    token_ids = generate(  # 使用generate函数生成token IDs
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),  # 将用户文本转换为token IDs并提供给generate函数
        max_new_tokens=35,  # 最大新token数
        context_size=model_config["context_length"],  # 上下文大小
        eos_id=50256  # 终止符ID
    )

    text = token_ids_to_text(token_ids, tokenizer)  # 将token IDs转换回文本
    response = extract_response(text, prompt)  # 提取响应内容

    await chainlit.Message(  # 将模型的响应发送回界面
        content=f"{response}",  # 返回模型响应
    ).send()  # 发送响应