# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# 版权所有，作者为 Sebastian Raschka，依据 Apache License 2.0 (详见 LICENSE.txt)。
# 来源: "Build a Large Language Model From Scratch" (从零开始构建大型语言模型)
# - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码地址: https://github.com/rasbt/LLMs-from-scratch
#
# 一个基于第七章代码的最小指令微调文件

import json  # 导入 JSON 模块，用于处理 JSON 数据
import psutil  # 导入 psutil 模块，用于检查系统进程
from tqdm import tqdm  # 导入 tqdm 模块，用于显示进度条
import urllib.request  # 导入 urllib.request 模块，用于发送 HTTP 请求


# 定义函数，用于查询模型生成的响应
def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    # 创建请求数据字典
    data = {
        "model": model,  # 模型名称
        "messages": [
            {"role": "user", "content": prompt}  # 用户输入的提示
        ],
        "options": {     # 确保生成确定性响应的设置
            "seed": 123,             # 随机种子，确保响应一致性
            "temperature": 0,        # 生成温度，0 表示完全确定性
            "num_ctx": 2048          # 最大上下文长度
        }
    }

    # 将字典转为 JSON 格式字符串并编码为字节
    payload = json.dumps(data).encode("utf-8")

    # 创建 HTTP POST 请求并设置所需的头部信息
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")  # 设置请求内容类型为 JSON

    # 发送请求并捕获响应
    response_data = ""  # 初始化响应数据为空字符串
    with urllib.request.urlopen(request) as response:  # 打开请求连接
        # 循环逐行读取响应内容并解码为字符串
        while True:
            line = response.readline().decode("utf-8")  # 解码每一行
            if not line:  # 如果没有更多内容，则退出循环
                break
            response_json = json.loads(line)  # 将响应行解析为 JSON 对象
            response_data += response_json["message"]["content"]  # 累加生成的内容

    return response_data  # 返回最终生成的响应数据


# 定义函数，用于检查指定进程是否正在运行
def check_if_running(process_name):
    running = False  # 初始化运行状态为 False
    for proc in psutil.process_iter(["name"]):  # 遍历所有正在运行的进程
        if process_name in proc.info["name"]:  # 如果找到匹配的进程名称
            running = True  # 设置运行状态为 True
            break  # 退出循环
    return running  # 返回运行状态


# 定义函数，用于格式化输入数据
def format_input(entry):
    # 构造指令部分文本
    instruction_text = (
        f"Below is an instruction that describes a task. "  # 提供的任务描述
        f"Write a response that appropriately completes the request."  # 要求生成适当的响应
        f"\n\n### Instruction:\n{entry['instruction']}"  # 插入任务指令
    )

    # 如果存在输入内容，追加到格式化文本中
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text  # 返回完整格式化文本


# 主函数
def main(file_path):
    # 检查 Ollama 服务是否正在运行
    ollama_running = check_if_running("ollama")

    if not ollama_running:  # 如果 Ollama 未运行，抛出异常
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
    print("Ollama running:", check_if_running("ollama"))  # 打印 Ollama 的运行状态

    # 打开并读取测试数据文件
    with open(file_path, "r") as file:
        test_data = json.load(file)  # 加载 JSON 文件内容为 Python 对象

    model = "llama3"  # 设置模型名称
    scores = generate_model_scores(test_data, "model_response", model)  # 计算模型响应评分
    print(f"Number of scores: {len(scores)} of {len(test_data)}")  # 打印评分数量
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")  # 打印平均分


# 定义函数，用于为数据集中的模型响应生成评分
def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []  # 初始化空列表以存储评分
    for entry in tqdm(json_data, desc="Scoring entries"):  # 显示进度条
        if entry[json_key] == "":  # 如果模型响应为空
            scores.append(0)  # 将评分设置为 0
        else:
            # 构造评分提示文本
            prompt = (
                f"Given the input `{format_input(entry)}` "  # 输入提示
                f"and correct output `{entry['output']}`, "  # 数据集中正确的响应
                f"score the model response `{entry[json_key]}`"  # 模型生成的响应
                f" on a scale from 0 to 100, where 100 is the best score. "  # 评分范围
                f"Respond with the integer number only."  # 要求仅返回整数评分
            )
            score = query_model(prompt, model)  # 发送提示并获取评分
            try:
                scores.append(int(score))  # 将评分转换为整数并添加到列表中
            except ValueError:  # 如果转换失败
                print(f"Could not convert score: {score}")  # 打印错误消息
                continue  # 跳过该条目

    return scores  # 返回评分列表


# 主入口
if __name__ == "__main__":

    import argparse  # 导入 argparse 模块，用于解析命令行参数

    # 定义命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Evaluate model responses with ollama"  # 描述工具功能
    )
    parser.add_argument(
        "--file_path",  # 参数名称
        required=True,  # 参数为必填
        help=(  # 参数帮助信息
            "The path to the test dataset `.json` file with the"
            " `'output'` and `'model_response'` keys"  # 测试数据需要包含 'output' 和 'model_response' 键
        )
    )
    args = parser.parse_args()  # 解析命令行参数

    main(file_path=args.file_path)  # 调用主函数并传入文件路径参数
