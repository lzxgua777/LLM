# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import os  # 导入os模块，用于操作文件和目录
import urllib.request  # 导入urllib.request模块，用于URL请求

# import requests  # 注释掉了requests库的导入，因为下面的代码使用的是urllib
import json  # 导入json模块，用于处理JSON数据
import numpy as np  # 导入NumPy库，用于数学运算
import tensorflow as tf  # 导入TensorFlow库，用于加载模型
from tqdm import tqdm  # 导入tqdm库，用于显示进度条


def download_and_load_gpt2(model_size, models_dir):  # 定义下载和加载GPT-2模型的函数
    # Validate model size  # 验证模型大小是否合法
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths  # 定义模型的存储路径
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"  # 定义模型的下载URL
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]  # 定义需要下载的文件列表

    # Download files  # 下载文件
    os.makedirs(model_dir, exist_ok=True)  # 如果模型目录不存在，则创建
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)  # 构造文件的下载URL
        file_path = os.path.join(model_dir, filename)  # 构造文件的存储路径
        download_file(file_url, file_path)  # 调用download_file函数下载文件

    # Load settings and params  # 加载模型的设置和参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)  # 获取最新的TensorFlow检查点路径
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))  # 加载hparams.json文件
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)  # 从TensorFlow检查点加载参数

    return settings, params  # 返回模型的设置和参数


def download_file(url, destination):  # 定义下载文件的函数
    # Send a GET request to download the file  # 发送GET请求下载文件
    try:
        with urllib.request.urlopen(url) as response:  # 打开URL
            # Get the total file size from headers, defaulting to 0 if not present  # 获取文件总大小
            file_size = int(response.headers.get("Content-Length", 0))

            # Check if file exists and has the same size  # 检查文件是否存在且大小相同
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return

            # Define the block size for reading the file  # 定义读取文件的块大小
            block_size = 1024  # 1 Kilobyte

            # Initialize the progress bar with total file size  # 使用总文件大小初始化进度条
            progress_bar_description = os.path.basename(url)  # 从URL中提取文件名
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:  # 创建进度条
                # Open the destination file in binary write mode  # 以二进制写模式打开目标文件
                with open(destination, "wb") as file:
                    # Read the file in chunks and write to destination  # 按块读取文件并写入目标文件
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))  # 更新进度条
    except urllib.error.HTTPError:  # 捕获HTTP错误
        s = (
            f"The specified URL ({url}) is incorrect, the internet connection cannot be established,"
            "\nor the requested file is temporarily unavailable.\nPlease visit the following website"
            " for help: https://github.com/rasbt/LLMs-from-scratch/discussions/273")  # 提供错误信息和帮助链接
        print(s)  # 打印错误信息

# Alternative way using `requests`
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
        # 以二进制写入模式打开目标文件 #
        with open(destination, "wb") as file:
            # 以块的形式迭代文件数据
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # Update progress bar
                file.write(chunk)  # Write the chunk to the file
"""

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # 初始化参数字典，为每一层创建一个空字典
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 遍历检查点中的每个变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载变量并移除单维度（singleton dimensions）
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 处理变量名称，提取相关部分
        variable_name_parts = name.split("/")[1:]  # 跳过 'model/' 前缀

        # 确定变量应该放置在哪个目标字典中
        target_dict = params
        if variable_name_parts[0].startswith("h"):  # 如果变量属于某一层（如h0, h1, h2, ...）
            layer_number = int(variable_name_parts[0][1:])  # 提取层号
            target_dict = params["blocks"][layer_number]  # 选择该层对应的字典

        # 递归地访问或创建嵌套字典
        for key in variable_name_parts[1:-1]:  # 遍历变量名称的中间部分（除了最后一个部分）
            target_dict = target_dict.setdefault(key, {})  # 如果没有该键，则创建一个空字典

        # 将变量数组赋值给最后一个键
        last_key = variable_name_parts[-1]  # 获取变量名称的最后部分
        target_dict[last_key] = variable_array  # 将变量数组存储到对应的键中

    return params  # 返回包含加载参数的字典
