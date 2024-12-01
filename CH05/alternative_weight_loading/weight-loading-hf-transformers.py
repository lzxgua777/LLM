# 导入用于获取包版本的模块
from importlib.metadata import version

# 定义要检查版本的包列表
pkgs = ["numpy", "torch", "transformers"]
for p in pkgs:
    print(f"{p} version: {version(p)}")

# 从transformers库导入GPT2Model类
from transformers import GPT2Model
# 定义允许的模型名称和对应的预训练模型名称
model_names = {
    "gpt2-small (124M)": "openai-community/gpt2",
    "gpt2-medium (355M)": "openai-community/gpt2-medium",
    "gpt2-large (774M)": "openai-community/gpt2-large",
    "gpt2-xl (1558M)": "openai-community/gpt2-xl"
}
# 选择要使用的模型
CHOOSE_MODEL = "gpt2-small (124M)"
# 从预训练模型加载GPT2模型，设置缓存目录
gpt_hf = GPT2Model.from_pretrained(model_names[CHOOSE_MODEL], cache_dir="checkpoints")
gpt_hf.eval()
# 定义基础配置
BASE_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "drop_rate": 0.0,       # Dropout rate
    "qkv_bias": True        # Query-key-value bias
}
# 定义不同模型的配置
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# 更新基础配置，加入选定模型的配置
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
# 定义一个函数，用于检查两个张量的形状是否匹配，并分配权重
def assign_check(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach())

# 导入numpy库
import numpy as np

# 定义一个函数，用于加载权重
# 以下代码块将预训练模型的权重分配给自定义GPT模型的相应层
# 定义一个函数，用于将预训练模型的权重加载到自定义的GPT模型中
def load_weights(gpt, gpt_hf):
    # 获取预训练模型的权重字典
    d = gpt_hf.state_dict()

    # 将预训练模型的位置编码权重分配给自定义模型的位置编码层
    gpt.pos_emb.weight = assign_check(gpt.pos_emb.weight, d["wpe.weight"])
    # 将预训练模型的词汇编码权重分配给自定义模型的词汇编码层
    gpt.tok_emb.weight = assign_check(gpt.tok_emb.weight, d["wte.weight"])

    # 遍历模型的每一层
    for b in range(BASE_CONFIG["n_layers"]):
        # 将预训练模型的注意力层权重按照查询（Q）、键（K）和值（V）拆分
        q_w, k_w, v_w = np.split(d[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
        # 将查询（Q）权重分配给自定义模型的对应层
        gpt.trf_blocks[b].att.W_query.weight = assign_check(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        # 将键（K）权重分配给自定义模型的对应层
        gpt.trf_blocks[b].att.W_key.weight = assign_check(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        # 将值（V）权重分配给自定义模型的对应层
        gpt.trf_blocks[b].att.W_value.weight = assign_check(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 将预训练模型的注意力层偏置按照查询（Q）、键（K）和值（V）拆分
        q_b, k_b, v_b = np.split(d[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
        # 将查询（Q）偏置分配给自定义模型的对应层
        gpt.trf_blocks[b].att.W_query.bias = assign_check(gpt.trf_blocks[b].att.W_query.bias, q_b)
        # 将键（K）偏置分配给自定义模型的对应层
        gpt.trf_blocks[b].att.W_key.bias = assign_check(gpt.trf_blocks[b].att.W_key.bias, k_b)
        # 将值（V）偏置分配给自定义模型的对应层
        gpt.trf_blocks[b].att.W_value.bias = assign_check(gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 将预训练模型的输出层权重分配给自定义模型的对应层
        gpt.trf_blocks[b].att.out_proj.weight = assign_check(gpt.trf_blocks[b].att.out_proj.weight,
                                                             d[f"h.{b}.attn.c_proj.weight"].T)
        # 将预训练模型的输出层偏置分配给自定义模型的对应层
        gpt.trf_blocks[b].att.out_proj.bias = assign_check(gpt.trf_blocks[b].att.out_proj.bias,
                                                           d[f"h.{b}.attn.c_proj.bias"])

        # 将预训练模型的前馈网络第一层权重分配给自定义模型的对应层
        gpt.trf_blocks[b].ff.layers[0].weight = assign_check(gpt.trf_blocks[b].ff.layers[0].weight,
                                                             d[f"h.{b}.mlp.c_fc.weight"].T)
        # 将预训练模型的前馈网络第一层偏置分配给自定义模型的对应层
        gpt.trf_blocks[b].ff.layers[0].bias = assign_check(gpt.trf_blocks[b].ff.layers[0].bias,
                                                           d[f"h.{b}.mlp.c_fc.bias"])
        # 将预训练模型的前馈网络输出层权重分配给自定义模型的对应层
        gpt.trf_blocks[b].ff.layers[2].weight = assign_check(gpt.trf_blocks[b].ff.layers[2].weight,
                                                             d[f"h.{b}.mlp.c_proj.weight"].T)
        # 将预训练模型的前馈网络输出层偏置分配给自定义模型的对应层
        gpt.trf_blocks[b].ff.layers[2].bias = assign_check(gpt.trf_blocks[b].ff.layers[2].bias,
                                                           d[f"h.{b}.mlp.c_proj.bias"])

        # 将预训练模型的第一层归一化权重分配给自定义模型的对应层
        gpt.trf_blocks[b].norm1.scale = assign_check(gpt.trf_blocks[b].norm1.scale, d[f"h.{b}.ln_1.weight"])
        # 将预训练模型的第一层归一化偏置分配给自定义模型的对应层
        gpt.trf_blocks[b].norm1.shift = assign_check(gpt.trf_blocks[b].norm1.shift, d[f"h.{b}.ln_1.bias"])
        # 将预训练模型的第二层归一化权重分配给自定义模型的对应层
        gpt.trf_blocks[b].norm2.scale = assign_check(gpt.trf_blocks[b].norm2.scale, d[f"h.{b}.ln_2.weight"])
        # 将预训练模型的第二层归一化偏置分配给自定义模型的对应层
        gpt.trf_blocks[b].norm2.shift = assign_check(gpt.trf_blocks[b].norm2.shift, d[f"h.{b}.ln_2.bias"])

        # 将预训练模型的最终归一化权重分配给自定义模型
        gpt.final_norm.scale = assign_check(gpt.final_norm.scale, d[f"ln_f.weight"])
        # 将预训练模型的最终归一化偏置分配给自定义模型
        gpt.final_norm.shift = assign_check(gpt.final_norm.shift, d[f"ln_f.bias"])
        # 将预训练模型的输出层权重分配给自定义模型的输出层
        gpt.out_head.weight = assign_check(gpt.out_head.weight, d["wte.weight"])
# 导入PyTorch库和之前章节定义的GPTModel类
import torch
from previous_chapters import GPTModel

# 根据基础配置创建一个GPT模型实例
gpt = GPTModel(BASE_CONFIG)

# 检测是否有可用的CUDA设备（GPU），如果有则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将预训练模型的权重加载到自定义的GPT模型中
load_weights(gpt, gpt_hf)

# 导入tiktoken库和之前章节定义的文本生成相关的函数
import tiktoken
from previous_chapters import generate, text_to_token_ids, token_ids_to_text

# 设置PyTorch的随机种子，以确保结果的可复现性
torch.manual_seed(123)

# 获取GPT-2模型的tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# 使用GPT模型生成文本
# 将输入文本转换为token IDs，然后生成最多30个新token的文本
token_ids = generate(
    model=gpt.to(device),  # 将模型发送到GPU或CPU
    idx=text_to_token_ids("Every effort moves", tokenizer).to(device),  # 将输入文本转换为token IDs并发送到GPU或CPU
    max_new_tokens=30,  # 生成的最大新token数量
    context_size=BASE_CONFIG["context_length"],  # 上下文长度，从基础配置中获取
    top_k=1,  # 采样时考虑的token数量
    temperature=1.0  # 控制生成文本的随机性
)

# 将生成的token IDs转换回文本，并打印输出
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))