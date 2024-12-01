# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import os  # 导入操作系统模块，用于处理文件路径和文件操作
import urllib.request  # 导入urllib模块，用于下载文件

# import requests  # 如果需要，可以替代urllib用于HTTP请求

import json  # 导入json模块，用于解析和生成JSON数据
import numpy as np  # 导入numpy模块，用于处理数组和矩阵操作
import tensorflow as tf  # 导入TensorFlow库，主要用于机器学习和深度学习任务
from tqdm import tqdm  # 导入tqdm库，用于显示进度条


# 下载并加载GPT-2模型
def download_and_load_gpt2(model_size, models_dir):
    # 验证模型大小是否有效
    allowed_sizes = ("124M", "355M", "774M", "1558M")  # 定义支持的模型大小
    if model_size not in allowed_sizes:
        # 如果模型大小无效，抛出错误
        raise ValueError(f"Model size not in {allowed_sizes}")

    # 定义模型路径
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"  # 模型文件存储的基本URL
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",  # 需要下载的文件
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 创建模型目录，如果不存在的话
    os.makedirs(model_dir, exist_ok=True)

    # 下载模型文件
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)  # 拼接出文件的URL
        file_path = os.path.join(model_dir, filename)  # 拼接出文件保存的路径
        download_file(file_url, file_path)  # 下载文件

    # 加载设置和模型参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)  # 查找最新的TensorFlow检查点文件
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))  # 读取模型设置文件
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)  # 从检查点加载模型参数

    return settings, params  # 返回设置和模型参数


# 下载文件的函数
def download_file(url, destination):
    # 发送GET请求来下载文件
    try:
        with urllib.request.urlopen(url) as response:
            # 从响应头中获取文件的总大小，若没有则默认为0
            file_size = int(response.headers.get("Content-Length", 0))

            # 检查文件是否已经存在且大小匹配
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return  # 文件已经下载完成，无需重新下载

            # 定义每次读取的块大小（1KB）
            block_size = 1024  # 1 Kilobyte

            # 初始化进度条
            progress_bar_description = os.path.basename(url)  # 提取URL中的文件名作为进度条的描述
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                # 打开目标文件准备写入
                with open(destination, "wb") as file:
                    # 以块的形式读取文件并写入目标文件
                    while True:
                        chunk = response.read(block_size)  # 读取一个块的数据
                        if not chunk:
                            break  # 如果没有更多数据，退出循环
                        file.write(chunk)  # 将数据写入文件
                        progress_bar.update(len(chunk))  # 更新进度条
    except urllib.error.HTTPError:
        # 处理可能出现的HTTP错误
        s = (
            f"The specified URL ({url}) is incorrect, the internet connection cannot be established,"
            "\nor the requested file is temporarily unavailable.\nPlease visit the following website"
            " for help: https://github.com/rasbt/LLMs-from-scratch/discussions/273")
        print(s)


# 加载GPT-2模型参数的函数
def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # 初始化一个字典用于存储每个层的参数
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}  # 创建一个空字典用于存储所有层的参数

    # 遍历检查点文件中的每个变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载变量并去除单一维度
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 处理变量名，提取有用的部分
        variable_name_parts = name.split("/")[1:]  # 跳过 'model/' 前缀

        # 确定该变量要放入的目标字典
        target_dict = params
        if variable_name_parts[0].startswith("h"):  # 如果是Transformer层中的参数
            layer_number = int(variable_name_parts[0][1:])  # 提取层的编号
            target_dict = params["blocks"][layer_number]  # 定位到相应的层

        # 递归访问或创建目标字典中的嵌套字典
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})  # 如果不存在该键，创建一个空字典

        # 将变量数组赋值给最后一个键
        last_key = variable_name_parts[-1]  # 获取变量名的最后一部分作为字典的键
        target_dict[last_key] = variable_array  # 将变量数组存入字典

    return params  # 返回包含所有层参数的字典
