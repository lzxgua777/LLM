from importlib.metadata import version

# 定义一个包含要检查版本的包名的列表
pkgs = [
    "tqdm",    # 进度条库
]

# 遍历包名列表，打印每个包的版本号
for p in pkgs:
    print(f"{p} version: {version(p)}")

import urllib.request
import json

# 定义一个函数，用于向模型发送查询
def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat", role="user"):
    # 创建一个字典，包含模型参数和要发送的消息
    data = {
        "model": model,
        "seed": 123,        # 用于得到确定性响应
        "temperature": 1.,   # 用于得到确定性响应
        "top_p": 1,
        "messages": [
            {"role": role, "content": prompt}
        ]
    }

    # 将字典转换为JSON格式的字符串，并编码为字节
    payload = json.dumps(data).encode("utf-8")

    # 创建一个请求对象，设置方法为POST，并添加必要的头部信息
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # 发送请求并捕获响应
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # 读取和解码响应
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    # 返回响应数据
    return response_data


# 使用query_model函数查询“Llamas吃什么？”
result = query_model("What do Llamas eat?")
print(result)


# 定义一个函数，用于从文本中提取指令
def extract_instruction(text):
    for content in text.split("\n"):
        if content:
            return content.strip()

# 定义一个查询字符串
query = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"

# 使用query_model函数查询，并提取指令
result = query_model(query, role="assistant")
instruction = extract_instruction(result)
print(instruction)

# 使用提取的指令再次查询模型
response = query_model(instruction, role="user")
print(response)

from tqdm import tqdm

# 设置数据集大小和初始化数据集列表
dataset_size = 5
dataset = []

# 使用tqdm进度条遍历数据集大小次数
for i in tqdm(range(dataset_size)):
    # 查询模型并提取指令
    result = query_model(query, role="assistant")
    instruction = extract_instruction(result)
    # 使用提取的指令查询模型
    response = query_model(instruction, role="user")
    # 将指令和响应作为条目添加到数据集中
    entry = {
        "instruction": instruction,
        "output": response
    }
    dataset.append(entry)

# 将数据集写入JSON文件
with open("instruction-data-llama3-7b.json", "w") as file:
    json.dump(dataset, file, indent=4)

