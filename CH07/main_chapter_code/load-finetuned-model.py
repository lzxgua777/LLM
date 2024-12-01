from importlib.metadata import version  # 导入模块以检查软件包版本

# 定义要检查的库列表
pkgs = [
    "tiktoken",    # 分词器库
    "torch",       # 深度学习库
]
# 遍历库并打印其版本
for p in pkgs:
    print(f"{p} version: {version(p)}")

from pathlib import Path  # 导入路径模块

# 定义微调后的模型文件路径
finetuned_model_path = Path("gpt2-medium355M-sft.pth")
if not finetuned_model_path.exists():  # 如果文件不存在
    print(
        f"Could not find '{finetuned_model_path}'.\n"
        "Run the `ch07.ipynb` notebook to finetune and save the finetuned model."
        # 提示用户运行指定的 notebook 以微调并保存模型
    )

from ch05.main_chapter_code.ch05 import GPTModel  # 导入 GPT 模型类

# 定义基本配置
BASE_CONFIG = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "drop_rate": 0.0,        # Dropout 比例
    "qkv_bias": True         # QKV 偏置
}

# 定义不同 GPT 模型的配置
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# 选择要使用的模型
CHOOSE_MODEL = "gpt2-medium (355M)"

# 更新基本配置为所选模型的配置
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# 从模型名称中提取模型大小
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
# 初始化 GPT 模型
model = GPTModel(BASE_CONFIG)

import torch  # 导入 PyTorch 库

# 加载微调后的模型参数
model.load_state_dict(torch.load(
    "gpt2-medium355M-sft.pth",  # 文件路径
    map_location=torch.device("cpu"),  # 将模型加载到 CPU
    weights_only=True  # 仅加载权重
))
model.eval()  # 设置模型为评估模式

import tiktoken  # 导入分词器库

# 初始化分词器
tokenizer = tiktoken.get_encoding("gpt2")

# 定义生成的提示文本
prompt = """Below is an instruction that describes a task. Write a response 
that appropriately completes the request.

### Instruction:
Convert the active sentence to passive: 'The chef cooks the meal every day.'
"""

from ch05.main_chapter_code.ch05 import (
    generate,  # 导入生成文本函数
    text_to_token_ids,  # 导入文本转 ID 函数
    token_ids_to_text  # 导入 ID 转文本函数
)

# 定义一个函数，用于从响应中提取最终结果
def extract_response(response_text, input_text):
    return response_text[len(input_text):].replace("### Response:", "").strip()

torch.manual_seed(123)  # 设置随机种子，确保生成结果一致性

# 调用生成函数，生成模型响应
token_ids = generate(
    model=model,  # 模型
    idx=text_to_token_ids(prompt, tokenizer),  # 将提示文本转为 ID
    max_new_tokens=35,  # 最大生成令牌数
    context_size=BASE_CONFIG["context_length"],  # 上下文长度
    eos_id=50256  # 结束标志 ID
)

# 将生成的 ID 转换为文本
response = token_ids_to_text(token_ids, tokenizer)
# 提取最终的模型响应文本
response = extract_response(response, prompt)
# 打印模型生成的结果
print(response)

