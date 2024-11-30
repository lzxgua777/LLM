# 版权所有 （c） Sebastian Raschka，Apache 许可证 2.0（参见 LICENSE.txt）。
# “从头开始构建大型语言模型” 的源代码
# - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch
# 引入标准库和第三方模块
import json  # 用于处理 JSON 格式数据
import numpy as np  # 用于数值计算
import os  # 用于文件和路径操作
import urllib.request  # 用于从 URL 下载文件
# import requests  # 可选，另一种用于网络请求的模块
import tensorflow as tf  # 用于处理 TensorFlow 模型检查点
import tiktoken  # 用于 GPT 的分词器
import torch  # PyTorch 库，用于深度学习
from tqdm import tqdm  # 用于显示进度条

# 引入本地模块
from ch04.main_chapter_code.ch04 import GPTModel  # 本地 GPT 模型类

# 文本转换为 Token ID
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)  # 使用分词器对文本编码
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加批次维度
    return encoded_tensor

# Token ID 转换为文本
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # 移除批次维度
    return tokenizer.decode(flat.tolist())  # 使用分词器解码
def download_and_load_gpt2(model_size, models_dir):
    # 验证模型大小是否有效
    allowed_sizes = ("124M", "355M", "774M", "1558M")  # 可用模型大小
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # 定义模型路径
    model_dir = os.path.join(models_dir, model_size)  # 模型存储目录
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"  # 模型基础 URL
    filenames = [  # 模型文件名列表
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 下载所需文件
    os.makedirs(model_dir, exist_ok=True)  # 确保目录存在
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)  # 文件完整 URL
        file_path = os.path.join(model_dir, filename)  # 本地文件路径
        download_file(file_url, file_path)  # 调用下载函数

    # 加载设置和参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)  # 获取最新的检查点路径
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))  # 加载超参数
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)  # 加载模型参数

    return settings, params  # 返回超参数和模型参数


"""
def download_file(url, destination):
    # 发送 GET 请求以流式方式下载文件
    response = requests.get(url, stream=True)

    # 从 header 中获取总文件大小，如果不存在，则默认为 0
    file_size = int(response.headers.get("content-length", 0))

    # 检查文件是否存在且大小相同
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # 定义读取文件的块大小
    block_size = 1024  # 1 Kilobyte

    # 使用总文件大小初始化进度条
    progress_bar_description = url.split("/")[-1]  # Extract filename from URL
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # 以二进制写入模式打开目标文件
        with open(destination, "wb") as file:
            # 以 chunk 为单位迭代文件数据
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # 更新进度条
                file.write(chunk)  # 将 chunk 写入文件
"""

def download_file(url, destination):
    # 发送 GET 请求下载文件
    with urllib.request.urlopen(url) as response:
        # 从响应头中获取文件大小（默认为 0）
        file_size = int(response.headers.get("Content-Length", 0))

        # 检查文件是否已存在且大小相同
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        # 定义读取块大小
        block_size = 1024  # 1KB

        # 初始化带进度条的下载器
        progress_bar_description = os.path.basename(url)  # 从 URL 提取文件名
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            # 打开目标文件以写入
            with open(destination, "wb") as file:
                # 循环读取块数据并写入文件
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    file.write(chunk)
                    progress_bar.update(len(chunk))  # 更新进度条

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # 初始化参数字典，每层对应一个空字典
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 遍历检查点中的变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载变量并移除单例维度
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 处理变量名以提取有意义的部分
        variable_name_parts = name.split("/")[1:]  # 跳过 'model/' 前缀

        # 确定变量所属的目标字典
        target_dict = params
        if variable_name_parts[0].startswith("h"):  # 如果变量名以 "h" 开头，表示属于某层
            layer_number = int(variable_name_parts[0][1:])  # 提取层号
            target_dict = params["blocks"][layer_number]

        # 递归访问或创建嵌套字典
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})  # 创建嵌套结构

        # 将变量数组赋值给最终的键
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params  # 返回加载的参数字典


# 定义一个赋值函数，用于将参数赋值给模型中的权重
def assign(left, right):
    if left.shape != right.shape:  # 如果左边和右边的形状不匹配，抛出错误
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))  # 将右边的值转化为 Tensor 并返回作为参数


# 加载 GPT 模型的权重参数
def load_weights_into_gpt(gpt, params):
    # 加载位置嵌入层权重
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    # 加载词嵌入层权重
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    # 遍历每一层
    for b in range(len(params["blocks"])):
        # 分割注意力层的权重矩阵为 Q, K, V（Query, Key, Value）
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)

        # 分配 Query, Key, Value 权重
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 分割注意力层的偏置项
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)

        # 分配偏置项
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 加载注意力输出层的权重和偏置
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # 加载前馈层的权重和偏置
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])

        # 加载前馈层第二层的权重和偏置
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # 加载 LayerNorm 层的权重和偏置
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    # 加载最终的 LayerNorm 层和输出头的权重
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


# 生成函数，基于输入生成新文本
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # 每次生成一个新的 token
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # 获取最后 `context_size` 个 token

        # 使用模型进行推理
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # 只关注最后一个时间步的 logits

        # 如果设置了 top_k，应用 top-k 采样
        if top_k is not None:
            # 取前 k 个最大值
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # 如果设置了 temperature，对 logits 进行缩放
        if temperature > 0.0:
            logits = logits / temperature  # 通过 temperature 缩放 logits

            # 应用 softmax 转换为概率
            probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)

            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # 否则，直接选择 logits 最大的索引作为下一个 token
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # 如果生成了 eos_id（结束符），则提前停止生成
        if idx_next == eos_id:
            break

        # 将新生成的 token 拼接到原序列上
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx  # 返回生成的 token 序列


# 主函数，执行模型加载、生成等操作
def main(gpt_config, input_prompt, model_size):
    # 设置设备（使用 GPU 如果有的话）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 下载并加载 GPT 模型的参数
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    # 初始化 GPT 模型
    gpt = GPTModel(gpt_config)
    load_weights_into_gpt(gpt, params)  # 将下载的权重加载到 GPT 模型中
    gpt.to(device)  # 将模型移动到设备上
    gpt.eval()  # 设置为评估模式

    # 获取 GPT-2 分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)  # 设置随机种子，保证结果的可重复性

    # 生成新的 token 序列
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(input_prompt, tokenizer).to(device),
        max_new_tokens=25,  # 生成 25 个新 token
        context_size=gpt_config["context_length"],  # 上下文长度
        top_k=50,  # top-k 采样限制
        temperature=1.0  # 采样温度
    )

    # 输出生成的文本
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


# 如果是主程序执行
if __name__ == "__main__":
    torch.manual_seed(123)  # 设置全局随机种子

    CHOOSE_MODEL = "gpt2-small (124M)"  # 选择模型的大小
    INPUT_PROMPT = "Every effort moves you"  # 输入的提示文本

    # 设置基础模型配置
    BASE_CONFIG = {
        "vocab_size": 50257,  # 词汇表大小
        "context_length": 1024,  # 上下文长度
        "drop_rate": 0.0,  # Dropout 概率
        "qkv_bias": True  # 是否使用查询-键-值偏置
    }

    # 模型尺寸对应的配置
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    main(BASE_CONFIG, INPUT_PROMPT, model_size)