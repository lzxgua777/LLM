# 版权声明：Sebastian Raschka 根据 Apache License 2.0（参见 LICENSE.txt）持有版权。
# "Build a Large Language Model From Scratch" 的来源
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch

import os  # 导入os模块，用于操作文件和目录
import urllib.request  # 导入urllib.request模块，用于URL请求

# import requests  # 导入requests模块，用于发送HTTP请求（本段代码未被使用）
import json  # 导入json模块，用于处理JSON数据
import numpy as np  # 导入numpy模块，用于数值计算
import tensorflow as tf  # 导入tensorflow模块，用于机器学习模型的构建和训练
from tqdm import tqdm  # 导入tqdm模块，用于显示进度条


# 定义一个函数，用于下载并加载GPT-2模型
def download_and_load_gpt2(model_size, models_dir):
    # 验证模型大小是否合法
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")  # 如果模型大小不合法，抛出异常

    # 定义模型目录路径
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"  # 定义模型的基本URL
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]  # 定义需要下载的文件列表

    # 下载文件
    os.makedirs(model_dir, exist_ok=True)  # 如果模型目录不存在，则创建
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)  # 构造每个文件的URL
        file_path = os.path.join(model_dir, filename)  # 构造每个文件的本地路径
        download_file(file_url, file_path)  # 调用download_file函数下载文件

    # 加载模型设置和参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)  # 获取最新的TensorFlow检查点路径
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))  # 加载模型设置
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)  # 加载模型参数

    return settings, params  # 返回模型设置和参数


# 定义一个函数，用于下载文件
def download_file(url, destination):
    # 发送GET请求以下载文件
    try:
        with urllib.request.urlopen(url) as response:
            # 从响应头中获取文件总大小，如果不存在则默认为0
            file_size = int(response.headers.get("Content-Length", 0))

            # 检查文件是否存在并且大小相同
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"文件已存在并且是最新的：{destination}")
                    return

            # 定义读取文件的块大小
            block_size = 1024  # 1 千字节

            # 使用总文件大小初始化进度条
            progress_bar_description = os.path.basename(url)  # 从URL中提取文件名
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                # 以二进制写模式打开目标文件
                with open(destination, "wb") as file:
                    # 按块读取文件并写入目标文件
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))  # 更新进度条
    except urllib.error.HTTPError:
        s = (
            f"指定的URL ({url}) 是错误的，无法建立互联网连接，"
            "\n或请求的文件暂时不可用。\n请访问以下网站寻求帮助：https://github.com/rasbt/LLMs-from-scratch/discussions/273")
        print(s)  # 打印错误信息


# 使用requests模块的替代方法（本段代码被注释掉了，未被使用）
"""
def download_file(url, destination):
    # 以流模式发送GET请求以下载文件
    response = requests.get(url, stream=True)

    # 从响应头中获取文件总大小，如果不存在则默认为0
    file_size = int(response.headers.get("content-length", 0))

    # 检查文件是否存在并且大小相同
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"文件已存在并且是最新的：{destination}")
            return

    # 定义读取文件的块大小
    block_size = 1024  # 1 千字节

    # 使用总文件大小初始化进度条
    progress_bar_description = url.split("/")[-1]  # 从URL中提取文件名
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # 以二进制写模式打开目标文件
        with open(destination, "wb") as file:
            # 按块迭代文件数据
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # 更新进度条
                file.write(chunk)  # 将块写入文件
"""


# 定义一个函数，从TensorFlow检查点加载GPT-2模型参数
def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # 使用空块为每个层初始化参数字典
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 遍历检查点中的每个变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载变量并移除单例维度
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 处理变量名以提取相关部分
        variable_name_parts = name.split("/")[1:]  # 跳过 'model/' 前缀

        # 确定变量的目标字典
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # 递归访问或创建嵌套字典
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # 将变量数组分配给最后一个键
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params  # 返回参数