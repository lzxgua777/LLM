# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import os
import urllib.request

# import requests  # 如果需要替代方法，可以使用 requests（目前被注释掉）
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm  # 用于显示进度条

def download_and_load_gpt2(model_size, models_dir):
    # 验证模型大小是否在允许的范围内
    allowed_sizes = ("124M", "355M", "774M", "1558M")  # GPT-2 允许的模型大小
    if model_size not in allowed_sizes:  # 如果用户选择的模型大小不在范围内
        raise ValueError(f"Model size not in {allowed_sizes}")  # 抛出错误

    # 定义模型的路径
    model_dir = os.path.join(models_dir, model_size)  # 模型存储的目录路径
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"  # 模型的远程存储 URL
    filenames = [  # 需要下载的文件列表
        "checkpoint", "encoder.json", "hparams.json",  # 配置文件
        "model.ckpt.data-00000-of-00001", "model.ckpt.index", "model.ckpt.meta",  # 模型检查点
        "vocab.bpe"  # 词汇文件
    ]

    # 下载所需文件
    os.makedirs(model_dir, exist_ok=True)  # 如果目录不存在，则创建目录
    for filename in filenames:  # 遍历文件列表
        file_url = os.path.join(base_url, model_size, filename)  # 构建文件的远程 URL
        file_path = os.path.join(model_dir, filename)  # 文件的本地存储路径
        download_file(file_url, file_path)  # 下载文件到本地

    # 加载模型设置和参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)  # 获取 TensorFlow 检查点的最新路径
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))  # 加载模型超参数
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)  # 从检查点加载参数

    return settings, params  # 返回超参数和模型参数
def download_file(url, destination):
    # 通过 URL 下载文件
    try:
        with urllib.request.urlopen(url) as response:  # 打开 URL
            # 从响应头中获取文件大小，如果没有则默认 0
            file_size = int(response.headers.get("Content-Length", 0))

            # 如果文件已存在且大小匹配，则跳过下载
            if os.path.exists(destination):  # 检查文件是否已存在
                file_size_local = os.path.getsize(destination)  # 获取本地文件大小
                if file_size == file_size_local:  # 如果本地文件大小与远程文件相同
                    print(f"File already exists and is up-to-date: {destination}")  # 打印提示
                    return

            # 定义读取文件的块大小
            block_size = 1024  # 1 KB

            # 初始化进度条并设置总大小
            progress_bar_description = os.path.basename(url)  # 从 URL 中提取文件名
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                # 以二进制写模式打开目标文件
                with open(destination, "wb") as file:
                    # 分块读取文件并写入本地文件
                    while True:
                        chunk = response.read(block_size)  # 读取一个块
                        if not chunk:  # 如果没有数据，结束读取
                            break
                        file.write(chunk)  # 将数据写入文件
                        progress_bar.update(len(chunk))  # 更新进度条
    except urllib.error.HTTPError:  # 捕获 HTTP 错误
        s = (
            f"The specified URL ({url}) is incorrect, the internet connection cannot be established,"
            "\nor the requested file is temporarily unavailable.\nPlease visit the following website"
            " for help: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(s)  # 打印错误信息



# Alternative way using `requests`
"""
def download_file(url, destination):
    # 发送 GET 请求以流式下载文件
    response = requests.get(url, stream=True)
    # 从响应头中获取文件的总大小，如果没有提供，则默认为 0
    file_size = int(response.headers.get("content-length", 0))
    # 检查本地文件是否存在以及大小是否匹配
    if os.path.exists(destination):  # 如果目标文件已存在
        file_size_local = os.path.getsize(destination)  # 获取本地文件的大小
        if file_size == file_size_local:  # 如果本地文件大小和远程文件大小相同
            print(f"File already exists and is up-to-date: {destination}")  # 文件已存在且是最新
            return  # 退出函数，避免重复下载
    # 定义读取文件的块大小
    block_size = 1024  # 1 千字节（1 KB）
    # 初始化带有总大小的进度条
    progress_bar_description = url.split("/")[-1]  # 从 URL 中提取文件名作为进度条描述
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # 以二进制写模式打开目标文件
        with open(destination, "wb") as file:
            # 遍历下载的数据块并写入本地文件
            for chunk in response.iter_content(block_size):  # 以块为单位迭代内容
                progress_bar.update(len(chunk))  # 更新进度条
                file.write(chunk)  # 将当前块写入本地文件

"""
def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # 初始化参数字典，其中 "blocks" 包含每一层的空字典
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}  # 初始化每一层的参数字典

    # 遍历检查点中的每个变量
    for name, _ in tf.train.list_variables(ckpt_path):  # 获取检查点中的所有变量
        # 加载变量并移除单维度
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 处理变量名，提取相关部分
        variable_name_parts = name.split("/")[1:]  # 跳过 "model/" 前缀

        # 确定目标字典（决定将变量存储在哪个层）
        target_dict = params
        if variable_name_parts[0].startswith("h"):  # 如果变量名以 "h" 开头，表示属于某一层
            layer_number = int(variable_name_parts[0][1:])  # 提取层号
            target_dict = params["blocks"][layer_number]  # 目标字典为该层的参数字典

        # 递归地访问或创建嵌套的字典
        for key in variable_name_parts[1:-1]:  # 遍历中间的键（层级）
            target_dict = target_dict.setdefault(key, {})  # 如果键不存在，则创建一个新的字典

        # 将变量数组赋值给最后一个键
        last_key = variable_name_parts[-1]  # 提取最后一个键
        target_dict[last_key] = variable_array  # 将变量赋值给目标字典的键

    return params  # 返回参数字典
