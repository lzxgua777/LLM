from importlib.metadata import version  # 导入版本检查模块

pkgs = [
    "tiktoken",    # 分词器库
    "torch",       # 深度学习库
]
for p in pkgs:  # 遍历包列表并打印每个包的版本
    print(f"{p} version: {version(p)}")  # 打印包的版本号

from pathlib import Path  # 导入路径操作模块

finetuned_model_path = Path("review_classifier.pth")  # 定义微调模型文件的路径
if not finetuned_model_path.exists():  # 如果模型文件不存在
    print(f"Could not find '{finetuned_model_path}'.\n"  # 打印错误信息
    "Run the `ch06.ipynb` notebook to finetune and save the finetuned model.")  # 提示用户运行Jupyter笔记本来微调和保存模型

from ch04.main_chapter_code.ch04 import GPTModel  # 从指定模块导入GPTModel类

BASE_CONFIG = {  # 定义基础配置
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "drop_rate": 0.0,        # Dropout率
    "qkv_bias": True         # Query-key-value偏置
}

model_configs = {  # 定义不同规模的GPT模型配置
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-small (124M)"  # 选择要使用的模型配置

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])  # 更新基础配置为所选模型配置

# Initialize base model  # 初始化基础模型
model = GPTModel(BASE_CONFIG)

import torch  # 导入PyTorch库

# Convert model to classifier as in section 6.5 in ch06.ipynb  # 将模型转换为分类器
num_classes = 2  # 类别数量
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

# Then load pretrained weights  # 然后加载预训练权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 确定设备
model.load_state_dict(torch.load("review_classifier.pth", map_location=device, weights_only=True))  # 加载模型权重
model.to(device)  # 将模型部署到设备
model.eval();  # 将模型设置为评估模式

import tiktoken  # 导入Tiktoken库

tokenizer = tiktoken.get_encoding("gpt2")  # 获取GPT-2的编码器

# This function was implemented in ch06.ipynb  # 这个函数在ch06.ipynb中实现
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):  # 定义分类函数
    model.eval()  # 将模型设置为评估模式

    # Prepare inputs to the model  # 准备模型输入
    input_ids = tokenizer.encode(text)  # 使用分词器编码文本
    supported_context_length = model.pos_emb.weight.shape[0]  # 获取支持的上下文长度

    # Truncate sequences if they too long  # 如果序列太长则截断
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad sequences to the longest sequence  # 将序列填充到最长序列
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # 添加批次维度

    # Model inference  # 模型推理
    with torch.no_grad():  # 在不计算梯度的上下文中
        logits = model(input_tensor.to(device))[:, -1, :]  # 获取最后一个输出标记的Logits
    predicted_label = torch.argmax(logits, dim=-1).item()  # 获取预测标签

    # Return the classified result  # 返回分类结果
    return "spam" if predicted_label == 1 else "not spam"

text_1 = (  # 定义要分类的文本1
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

print(classify_review(  # 打印分类结果
    text_1, model, tokenizer, device, max_length=120
))

text_2 = (  # 定义要分类的文本2
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print(classify_review(  # 打印分类结果
    text_2, model, tokenizer, device, max_length=120
))



