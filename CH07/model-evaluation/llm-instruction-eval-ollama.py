# 导入 version 方法用于获取已安装包的版本
from importlib.metadata import version

# 要检查版本的包列表
pkgs = ["tqdm",    # 进度条
        ]

# 循环打印每个包的版本
for p in pkgs:
    print(f"{p} version: {version(p)}")  # 打印包名和版本号

# 导入用于网络请求的urllib模块和json模块
import urllib.request
import json

# 定义一个函数，用于查询模型响应
def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    # 创建数据负载，作为字典传递
    data = {
        "model": model,  # 模型名称
        "messages": [
            {
                "role": "user",  # 用户角色
                "content": prompt  # 用户输入的内容
            }
        ],
        "options": {  # 设置一些选项，确保响应具有确定性
            "seed": 123,  # 设置随机种子
            "temperature": 0,  # 设置温度为0，确保结果不随机
            "num_ctx": 2048  # 上下文窗口大小
        }
    }

    # 将字典转换为JSON格式的字符串，并编码为字节
    payload = json.dumps(data).encode("utf-8")

    # 创建请求对象，设置方法为POST，并添加必要的头部
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")  # 设置请求内容类型为JSON

    # 发送请求并捕获响应数据
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # 读取并解码响应
        while True:
            line = response.readline().decode("utf-8")  # 按行读取响应
            if not line:  # 如果没有更多内容，跳出循环
                break
            response_json = json.loads(line)  # 解析每行的JSON数据
            response_data += response_json["message"]["content"]  # 获取响应内容并追加

    return response_data  # 返回拼接后的响应内容

# 调用query_model函数并打印结果
result = query_model("What do Llamas eat?")  # 查询关于“Llamas吃什么”的问题
print(result)  # 打印模型返回的结果

# 加载一个包含多个示例的JSON文件
json_file = "eval-example-data.json"

# 打开并读取JSON文件
with open(json_file, "r") as file:
    json_data = json.load(file)  # 解析JSON文件内容

# 打印JSON文件中的条目数
print("Number of entries:", len(json_data))

# 查看JSON文件中的第一个条目
json_data[0]

# 定义一个函数来格式化输入数据
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. Write a response that "
        f"appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"  # 任务说明
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""  # 如果有输入，显示输入内容
    instruction_text + input_text  # 拼接说明和输入内容

    return instruction_text + input_text  # 返回最终格式化后的文本

# 对前五条数据执行评分操作
for entry in json_data[:5]:
    # 生成评分的提示文本
    prompt = (f"Given the input `{format_input(entry)}` "
              f"and correct output `{entry['output']}`, "
              f"score the model response `{entry['model 1 response']}`"
              f" on a scale from 0 to 100, where 100 is the best score. "
              )
    print("\nDataset response:")  # 输出数据集的正确响应
    print(">>", entry['output'])
    print("\nModel response:")  # 输出模型的响应
    print(">>", entry["model 1 response"])
    print("\nScore:")  # 输出评分
    print(">>", query_model(prompt))  # 调用模型评分函数并打印结果
    print("\n-------------------------")

# 导入进度条工具
from tqdm import tqdm

# 定义一个函数，用于为所有条目生成模型评分
def generate_model_scores(json_data, json_key):
    scores = []  # 存储评分的列表
    for entry in tqdm(json_data, desc="Scoring entries"):  # 使用进度条遍历数据
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt)  # 调用模型评分函数
        try:
            scores.append(int(score))  # 将评分转换为整数并添加到列表
        except ValueError:  # 如果评分转换失败，跳过
            continue

    return scores  # 返回所有的评分

# 导入路径处理模块
from pathlib import Path

# 为“model 1 response”和“model 2 response”分别生成评分
for model in ("model 1 response", "model 2 response"):
    scores = generate_model_scores(json_data, model)  # 生成评分
    print(f"\n{model}")  # 输出模型名称
    print(f"Number of scores: {len(scores)} of {len(json_data)}")  # 输出评分数量
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")  # 输出平均评分

    # 可选：保存评分到文件
    save_path = Path("scores") / f"llama3-8b-{model.replace(' ', '-')}.json"  # 生成文件保存路径
    with open(save_path, "w") as file:
        json.dump(scores, file)  # 将评分保存为JSON格式文件
