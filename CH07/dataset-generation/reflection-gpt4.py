from importlib.metadata import version  # 导入 version 函数，用于检查安装的包的版本

# 定义需要检查版本的包列表
pkgs = [
    "openai",  # OpenAI API
    "tqdm",    # 进度条库
]

# 遍历包列表，打印每个包的版本
for p in pkgs:
    print(f"{p} version: {version(p)}")

import json  # 导入 JSON 模块，用于处理 JSON 格式数据
from openai import OpenAI  # 从 openai 包中导入 OpenAI 客户端

# 从配置文件中加载 API 密钥
with open("config.json", "r") as config_file:
    config = json.load(config_file)  # 加载配置文件内容
    api_key = config["OPENAI_API_KEY"]  # 提取 API 密钥

client = OpenAI(api_key=api_key)  # 使用 API 密钥实例化 OpenAI 客户端


# 定义一个函数，用于调用 ChatGPT API
def run_chatgpt(prompt, client, model="gpt-4o-mini", system_prompt=None):
    messages = []  # 初始化消息列表

    # 如果提供了 system_prompt，则将其添加到消息列表
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 添加用户的输入 prompt 到消息列表
    messages.append({"role": "user", "content": prompt})

    # 调用 OpenAI API
    response = client.chat.completions.create(
        model=model,  # 使用的模型
        messages=messages,  # 消息列表
        temperature=0.0,  # 设置生成结果的随机性
        seed=123,  # 随机种子，保证可重复性
    )

    # 返回模型的响应内容
    return response.choices[0].message.content


prompt = f"Respond with 'hello world' if you got this message."  # 定义测试 prompt
run_chatgpt(prompt, client)  # 调用 run_chatgpt 函数

from pathlib import Path  # 导入 Path 模块，用于处理文件路径

# 定义 JSON 文件路径
json_file = Path("..") / "01_main-chapter-code" / "instruction-data.json"

# 打开 JSON 文件并加载内容
with open(json_file, "r") as file:
    json_data = json.load(file)  # 加载 JSON 数据

print("Number of entries:", len(json_data))  # 打印 JSON 数据的条目数量

from pprint import pp as pprint  # 导入 pprint 并重命名为 pprint，用于格式化输出

pprint(json_data[0])  # 打印 JSON 数据中的第一个条目


# 定义一个函数，用于根据输入生成新的系统提示和用户提示
def instr_prompt_no_input(ins, outp):
    sys_prompt = "You are a helpful, precise but picky assistant for checking the quality of a given instruction."  # 定义系统提示
    prompt_template = (
        "[Instruction]\n{ins}\n\n[The Start of Answer]\n{outp}\n\n[The End of Answer]\n\n[System]\n{criteria}\n\n"
    )  # 定义用户提示的模板
    criteria = (
        "We would like you to answer several questions related to the quality of a given instruction. \n"
        + "1. Why this instruction is not good? First analyse the instruction based on Complexity of the Topic, Level of Detail Required, Knowledge Required, Ambiguity of the Instruction and Logical Reasoning or Problem-Solving Involved. \n"
        + "Then analyse why this answer is not good for the given instruction? Analyse based on the Helpfulness, Relevance, Accuracy and Level of Details. \n"
        + "Finally analyse why this bad instruction lead to a bad answer. "
        + "2. Based on the reason you provided, generate a new and complete instruction which is complex and difficult to answer directly. "
        + "Make sure the new instruction is relevent but independent to the original instruction, which can be answered without knowing the original instruction, put the new instruction in the format of [New Instruction] your instruction [End]"
        + "3. Answer the newly generated instruction as detailed as possible, in the format of [New Answer] your answer [End] \n"
    )  # 定义评估的标准
    prompt = prompt_template.format(ins=ins, outp=outp, criteria=criteria)  # 格式化提示
    return sys_prompt, prompt  # 返回系统提示和用户提示

pprint(json_data[2])  # 打印 JSON 数据中的第三个条目
entry = json_data[2]  # 获取第三个条目

# 调用 instr_prompt_no_input 函数生成系统提示和用户提示
system_prompt, prompt = instr_prompt_no_input(ins=entry["instruction"], outp=entry["output"])
output = run_chatgpt(prompt=prompt, client=client, system_prompt=system_prompt)  # 调用 ChatGPT API
print(output)  # 打印模型的响应

import re  # 导入正则表达式模块

# 定义一个函数，用于从响应文本中提取新指令
def extract_ins(text, no_input=True):
    if '[New Instruction]' in text:
        pattern = r'(\[New Instruction\])(.*?)(\[End\]|\[New Answer\]|New Answer:)'  # 匹配新指令的正则表达式
    else:
        pattern = r'(New Instruction:)(.*?)(\[End\]|\[New Answer\]|New Answer:)'  # 匹配未标记的新指令
    segments = re.findall(pattern, text, re.DOTALL)  # 搜索所有符合条件的片段
    if len(segments) == 0:
        seg_ins = ''  # 如果未匹配到，返回空字符串
    else:
        seg_ins = segments[0][1].strip()  # 提取第一个匹配片段并去掉首尾空格
    if seg_ins.endswith("\n\n3."):
        seg_ins = seg_ins[:-4]  # 去除尾部的多余内容
    return seg_ins  # 返回提取的指令

# 定义一个函数，用于从响应文本中提取新答案
def extract_oup(text, no_input=True):
    if '[New Answer]' in text:
        pattern = r'(\[New Answer\])(.*?)(\[End\]|$)'  # 匹配新答案的正则表达式
    else:
        pattern = r'(New Answer:)(.*?)(\[End\]|$)'  # 匹配未标记的新答案
    segments = re.findall(pattern, text, re.DOTALL)  # 搜索所有符合条件的片段
    if len(segments) == 0:
        seg_oup = ''  # 如果未匹配到，返回空字符串
    else:
        seg_oup = segments[0][1].strip()  # 提取第一个匹配片段并去掉首尾空格
    return seg_oup  # 返回提取的答案

# 定义一个函数，从响应文本中提取新指令和答案
def extract_instruction(text):
    if text == '':
        return []  # 如果输入为空，返回空列表
    seg_ins = extract_ins(text, no_input=True)  # 提取新指令
    seg_oup = extract_oup(text, no_input=True)  # 提取新答案
    return [seg_ins, seg_oup]  # 返回提取结果的列表

new_instr, new_outp = extract_instruction(output)  # 从模型输出中提取新指令和答案

print(new_instr)  # 打印新指令
print(new_outp)  # 打印新答案
def res_gen_prompt_no_input(ins, outp):
    """
    定义函数：生成系统提示和用户提示，用于在没有输入情况下检查答案的质量。
    """
    sys_prompt = "You are a helpful, precise but picky assistant for checking the quality of the answer to a given instruction."  # 系统提示
    prompt_template = "[Instruction]\n{ins}\n\n[The Start of Answer]\n{outp}\n\n[The End of Answer]\n\n[System]\n{criteria}\n\n"  # 用户提示模板
    criteria = (
        "We would like you to answer several questions related to the quality of the answer to the given instruction. \n"
        + "1. Why this answer is not good for the given instruction? Analyse based on the Helpfulness, Relevance, Accuracy and Level of Details. \n"
        + "2. Based on the reason you provided, generate a better answer, new and complete, as detailed as possible, in the format of [Better Answer] your answer [End] \n"
    )  # 定义评估标准
    prompt = prompt_template.format(ins=ins, outp=outp, criteria=criteria)  # 格式化提示
    return sys_prompt, prompt  # 返回系统提示和用户提示


def res_gen_prompt_input(ins, inp, outp):
    """
    定义函数：生成系统提示和用户提示，用于在有输入情况下检查答案的质量。
    """
    sys_prompt = "You are a helpful and precise assistant for checking the quality of the answer to a given instruction and its input."  # 系统提示
    prompt_template = "[Instruction]\n{ins}\n\n[The Start of Input]\n{inp}\n\n[The End of Input]\n\n[The Start of Answer]\n{outp}\n\n[The End of Answer]\n\n[System]\n{criteria}\n\n"  # 用户提示模板
    criteria = (
        "We would like you to answer several questions related to the quality of the answer to the given instruction and corresponding input. \n"
        + "1. Why this answer is not good for the given instruction and corresponding input? Analyse based on the Helpfulness, Relevance, Accuracy and Level of Details. \n"
        + "2. Based on the reason you provided, generate a better answer, new and complete, as detailed as possible, in the format of [Better Answer] your answer [End] \n"
    )  # 定义评估标准
    prompt = prompt_template.format(ins=ins, inp=inp, outp=outp, criteria=criteria)  # 格式化提示
    return sys_prompt, prompt  # 返回系统提示和用户提示


entry = json_data[2]  # 获取 JSON 数据中的第三个条目

# 使用函数生成提示
system_prompt, prompt = res_gen_prompt_no_input(ins=entry["instruction"], outp=entry["output"])
output = run_chatgpt(prompt=prompt, client=client, system_prompt=system_prompt)  # 调用 ChatGPT API 生成响应
print(output)  # 打印生成的响应


def extract_response(text):
    """
    定义函数：从文本中提取更好的答案。
    """
    if text.count('[Better Answer]') >= 2:
        pattern = r'\[(Better Answer)\](.*?)(\[End\]|\[Better Answer\]|$)'  # 匹配多个“更好答案”的正则表达式
        segments = re.findall(pattern, text, re.DOTALL)
    else:
        pattern = r'\[(Better Answer)\](.*?)(\[End\]|End|$)'  # 匹配单个“更好答案”的正则表达式
        segments = re.findall(pattern, text, re.DOTALL)
    return [segment[1].strip() for segment in segments]  # 返回提取的答案


response = extract_response(output)[0]  # 提取第一个“更好答案”
print(response)  # 打印提取的答案


data_to_process = json_data[:3]  # 获取 JSON 数据的前三个条目

from tqdm import tqdm  # 导入 tqdm 模块，用于显示进度条


def reflect_instructions(json_data, client):
    """
    定义函数：重新生成指令。
    """
    new_json_data = []  # 初始化新数据列表

    for entry in tqdm(json_data):  # 遍历 JSON 数据并显示进度条

        if not entry["input"]:  # 如果没有输入
            # 生成新的系统提示和用户提示
            system_prompt, prompt = instr_prompt_no_input(ins=entry["instruction"], outp=entry["output"])
            output = run_chatgpt(prompt=prompt, client=client, system_prompt=system_prompt)  # 调用 ChatGPT API
            new_instr, new_outp = extract_instruction(output)  # 提取新指令和答案
            new_entry = {"instruction": new_instr, "input": "", "output": new_outp}  # 创建新的条目
            new_json_data.append(new_entry)  # 添加到新数据列表
        else:
            new_json_data.append(entry)  # 如果有输入，保留原始条目

    return new_json_data  # 返回新数据列表


data_to_process = json_data[:3]  # 获取 JSON 数据的前三个条目
new_json_data = reflect_instructions(data_to_process, client)  # 调用函数重新生成指令

# 打印新数据的前三个条目
for i in new_json_data[:3]:
    pprint(i)
    print("\n\n")

# 将新数据保存到文件
with open("instruction-reflected.json", "w") as file:
    json.dump(new_json_data, file, indent=4)


data_to_process = json_data[:3]  # 再次获取 JSON 数据的前三个条目


def reflect_responses(json_data, client):
    """
    定义函数：重新生成答案。
    """
    new_json_data = []  # 初始化新数据列表

    for entry in tqdm(json_data):  # 遍历 JSON 数据并显示进度条

        if not entry["input"]:  # 如果没有输入
            system_prompt, prompt = res_gen_prompt_no_input(ins=entry["instruction"], outp=entry["output"])  # 生成提示
            output = run_chatgpt(prompt=prompt, client=client, system_prompt=system_prompt)  # 调用 ChatGPT API
            new_response = extract_response(output)  # 提取新答案

            if not len(new_response):  # 如果未提取到新答案，使用原始答案
                new_response = entry["output"]

            new_entry = {"instruction": entry["instruction"], "input": "", "output": new_response[0]}  # 创建新条目
            new_json_data.append(new_entry)  # 添加到新数据列表

        else:  # 如果有输入
            system_prompt, prompt = res_gen_prompt_input(ins=entry["instruction"], inp=entry["input"],
                                                         outp=entry["output"])  # 生成提示
            output = run_chatgpt(prompt=prompt, client=client, system_prompt=system_prompt)  # 调用 ChatGPT API
            new_response = extract_response(output)  # 提取新答案

            if not len(new_response):  # 如果未提取到新答案，使用原始答案
                new_response = entry["output"]

            new_entry = {"instruction": entry["instruction"], "input": entry["input"], "output": new_response[0]}  # 创建新条目
            new_json_data.append(new_entry)  # 添加到新数据列表

    return new_json_data  # 返回新数据列表


new_json_data = reflect_responses(data_to_process, client)  # 调用函数重新生成答案

# 打印新数据的前三个条目
for i in new_json_data[:3]:
    pprint(i)
    print("\n\n")

# 将新数据保存到文件
with open("response-reflected.json", "w") as file:
    json.dump(new_json_data, file, indent=4)
