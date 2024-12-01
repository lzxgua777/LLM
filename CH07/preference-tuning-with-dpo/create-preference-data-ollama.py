# 导入模块
from importlib.metadata import version

# 包名称列表
pkgs = ["tqdm",    # 进度条库
        ]

# 循环输出包的版本
for p in pkgs:
    print(f"{p} version: {version(p)}")

# 导入用于HTTP请求的库
import urllib.request
import json

# 定义一个查询模型的函数
def query_model(prompt, model="llama3.1:70b", url="http://localhost:11434/api/chat"):
    # 创建数据载荷字典
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "options": {
            "seed": 123,
            "temperature": 0,
        }
    }

    # 将字典转换为JSON格式字符串并编码为字节
    payload = json.dumps(data).encode("utf-8")

    # 创建一个请求对象，设置方法为POST并添加必要的头信息
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # 发送请求并捕获响应
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # 逐行读取并解码响应
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data

# 查询模型，输入问题
result = query_model("What do Llamas eat?")
print(result)

# 导入Path用于处理文件路径
from pathlib import Path

# 定义JSON文件路径
json_file = Path("..", "01_main-chapter-code", "instruction-data.json")

# 打开并加载JSON文件数据
with open(json_file, "r") as file:
    json_data = json.load(file)

# 打印数据条目数量
print("Number of entries:", len(json_data))

# 打印第一个条目的内容
json_data[0]

# 定义一个格式化输入函数
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. Write a response that "
        f"appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # 如果有输入，添加输入部分
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    instruction_text + input_text

    return instruction_text + input_text

# 随机选择礼貌程度并修改输出
import random

for entry in json_data[:5]:
    politeness = random.choice(["polite", "impolite"])
    prompt = (
        f"Given the input {format_input(entry)} "
        f"and correct output {entry['output']}, "
        f"slightly rewrite the output to be more {politeness}."
        "Keep the modification minimal."
        "Only return return the generated response and nothing else."
    )
    # 打印数据集响应和修改后的响应
    print("\nDataset response:")
    print(">>", entry['output'])
    print(f"\n{politeness} response:")
    print(">>", query_model(prompt))

# 导入随机库和进度条库
import random
from tqdm import tqdm

# 定义生成模型响应的函数
def generate_model_responses(json_data):
    for i, entry in enumerate(tqdm(json_data, desc="Writing entries")):
        # 随机选择礼貌程度
        politeness = random.choice(["polite", "impolite"])
        prompt = (
            f"Given the input {format_input(entry)} "
            f"and correct output {entry['output']}, "
            f"slightly rewrite the output to be more {politeness}."
            "Keep the modification minimal."
            "Only return return the generated response and nothing else."
        )
        # 获取模型响应
        response = query_model(prompt)

        # 根据礼貌程度将响应分配到选项中
        if politeness == "polite":
            json_data[i]["chosen"] = response
            json_data[i]["rejected"] = entry["output"]
        else:
            json_data[i]["rejected"] = response
            json_data[i]["chosen"] = entry["output"]

# 生成模型响应并保存结果
generate_model_responses(json_data)

# 将结果写入新的JSON文件
with open("instruction-data-with-preference.json", "w") as file:
    json.dump(json_data, file, indent=4)

