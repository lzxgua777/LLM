# 导入 version 方法用于获取已安装包的版本
from importlib.metadata import version

# 要检查版本的包列表
pkgs = ["openai",  # OpenAI API
        "tqdm",    # 进度条
       ]

# 循环打印每个包的版本
for p in pkgs:
    print(f"{p} version: {version(p)}")  # 打印包名和版本号

# 导入json模块，用于处理JSON数据
import json
# 从OpenAI模块导入OpenAI类
from openai import OpenAI


# 从JSON文件中加载API密钥
# 确保替换 "sk-..." 为你实际的API密钥，密钥可以从 https://platform.openai.com/api-keys 获取
with open("config.json", "r") as config_file:
    config = json.load(config_file)  # 读取配置文件内容
    api_key = config["OPENAI_API_KEY"]  # 获取API密钥

# 创建OpenAI客户端对象，并传入API密钥
client = OpenAI(api_key=api_key)

# 定义一个函数，用于与ChatGPT模型进行交互并获取响应
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def run_chatgpt(prompt, client, model="gpt-4-turbo"):
    response = client.chat.completions.create(
        model=model,  # 使用指定的模型
        messages=[{"role": "user", "content": prompt}],  # 用户输入的消息
        temperature=0.0,  # 控制生成结果的随机性
    )
    return response.choices[0].message.content  # 返回生成的内容

# 准备输入数据
sentence = "I ate breakfast"  # 示例句子
prompt = f"Convert the following sentence to passive voice: '{sentence}'"  # 请求将句子转换为被动语态
run_chatgpt(prompt, client)  # 调用函数获取结果

# 载入一个包含多个示例的JSON文件
json_file = "instruction-examples.json"

# 打开并读取JSON文件
with open(json_file, "r") as file:
    json_data = json.load(file)  # 解析JSON文件内容

# 打印JSON文件中的条目数
print("Number of entries:", len(json_data))
# 输出前5条记录进行查看
for entry in json_data[:5]:
    text = entry["output"]  # 获取每个条目的输出内容
    prompt = f"Without adding any response or explanation, convert the following text to passive voice: {text}"  # 生成转换为被动语态的请求

    print("\nInput:")  # 输出输入文本
    print(">>", text)
    print("\nOutput:")  # 输出转换后的文本
    print(">>", run_chatgpt(prompt, client))  # 调用函数转换并打印结果
    print("\n-------------------------")

# 导入tqdm模块，用于显示进度条
from tqdm import tqdm  # 进度条工具

# 使用进度条处理所有条目
for i, entry in tqdm(enumerate(json_data[:5]), total=len(json_data[:5])):
    text = entry["output"]  # 获取每个条目的输出内容
    prompt = f"Without adding any response or explanation, convert the following text to passive voice: {text}"  # 生成转换为被动语态的请求
    json_data[i]["output_2"] = run_chatgpt(prompt, client)  # 获取并存储转换结果

# 查看处理后的第一个条目
json_data[0]

# 对所有条目执行转换，并更新输出
for i, entry in tqdm(enumerate(json_data), total=len(json_data)):
    text = entry["output"]  # 获取每个条目的输出内容
    prompt = f"Without adding any response or explanation, convert the following text to passive voice: {text}"  # 生成转换为被动语态的请求
    json_data[i]["output_2"] = run_chatgpt(prompt, client)  # 获取并存储转换结果

# 生成修改后的JSON文件名
new_json_file = json_file.replace(".json", "-modified.json")

# 将修改后的数据保存到新的JSON文件中
with open(new_json_file, "w") as file:
    json.dump(json_data, file, indent=4)  # 使用"indent"参数进行格式化输出

