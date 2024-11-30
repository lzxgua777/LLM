# 版权所有 （c） Sebastian Raschka，Apache 许可证 2.0（参见 LICENSE.txt）。
# “从头开始构建大型语言模型” 的源代码
# - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch

"""
将 Project Gutenberg 文件处理为较少的较大文件的脚本。
"""
import argparse  # 导入用于解析命令行参数的模块
import os  # 导入操作系统相关的模块（例如文件路径操作）
import re  # 导入正则表达式模块，用于处理字符串匹配和替换
from tqdm import tqdm  # 导入tqdm模块，用于显示进度条
from gutenberg.src.cleanup import strip_headers  # 从gutenberg包导入strip_headers函数，用于清理文件头部

# 定义一个函数，判断文本是否主要为英文
def is_english(text, threshold=0.9):
    ascii_chars = sum(1 for c in text if ord(c) < 128)  # 统计文本中ASCII字符的数量
    return ascii_chars / len(text) > threshold  # 如果ASCII字符占比超过threshold，认为是英文

# 定义一个函数，合并多个文件，分割成较大的文件
def combine_files(file_paths, target_dir, max_size_mb=500, separator="<|endoftext|>", fallback_encoding="latin1"):
    if not os.path.exists(target_dir):  # 如果目标目录不存在
        os.makedirs(target_dir)  # 创建目标目录

    current_content = []  # 用于存储当前文件的内容
    current_size = 0  # 当前文件的大小（以字节为单位）
    file_counter = 1  # 文件计数器，用于给输出文件命名

    # 遍历所有文件路径
    for file_path in tqdm(file_paths):  # 使用tqdm来显示进度条
        try:
            with open(file_path, "r", encoding="utf-8") as file:  # 尝试以utf-8编码打开文件
                content = file.read()  # 读取文件内容
        except UnicodeDecodeError:  # 如果遇到解码错误，使用备用编码
            tqdm.write(f"Warning: UnicodeDecodeError encountered. Trying fallback encoding for {file_path}")
            with open(file_path, "r", encoding=fallback_encoding) as file:
                content = file.read()  # 使用备用编码读取文件内容

        if not is_english(content):  # 如果文件内容不是英文，则跳过
            tqdm.write(f"Skipping {file_path} as it does not contain primarily English text.")
            continue  # 跳过该文件，继续下一个

        content = strip_headers(content)  # 清除文件头部信息

        # 使用正则表达式将多个空白行替换为单个空白行
        content = re.sub(r'\n\s*\n', '\n\n', content)
        estimated_size = len(content.encode("utf-8"))  # 估算内容的大小，以字节为单位

        # 如果当前文件的大小超过最大文件大小限制，则保存当前内容到目标文件
        if current_size + estimated_size > max_size_mb * 1024 * 1024:
            target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")  # 生成目标文件路径
            with open(target_file_path, "w", encoding="utf-8") as target_file:
                target_file.write(separator.join(current_content))  # 将当前内容写入目标文件
            file_counter += 1  # 更新文件计数器
            current_content = [content]  # 重置当前内容为新的文件内容
            current_size = estimated_size  # 重置当前文件大小
        else:
            current_content.append(content)  # 将内容添加到当前文件的内容中
            current_size += estimated_size  # 更新当前文件大小

    # 如果有剩余的内容，将其保存为一个新的文件
    if current_content:
        target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
        with open(target_file_path, "w", encoding="utf-8") as target_file:
            target_file.write(separator.join(current_content))  # 写入剩余内容
    return file_counter  # 返回总共生成的文件数

# 主函数部分
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess and combine text files for pretraining")  # 创建命令行解析器

    parser.add_argument("--data_dir", type=str, default="gutenberg/data/raw",  # 添加数据目录参数
                        help="Directory containing the downloaded raw training data")
    parser.add_argument("--max_size_mb", type=int, default=500,  # 添加最大文件大小参数（单位：MB）
                        help="The maximum file size for each concatenated file in megabytes")
    parser.add_argument("--output_dir", type=str, default="gutenberg_preprocessed",  # 添加输出目录参数
                        help="Directory where the preprocessed data will be saved")

    args = parser.parse_args()  # 解析命令行参数

    # 获取指定目录下所有的文本文件路径
    all_files = [os.path.join(path, name) for path, subdirs, files in os.walk(args.data_dir)
                 for name in files if name.endswith((".txt", ".txt.utf8"))]

    print(f"{len(all_files)} file(s) to process.")  # 输出将要处理的文件数量
    file_counter = combine_files(all_files, args.output_dir, max_size_mb=args.max_size_mb)  # 调用合并文件函数
    print(f"{file_counter} file(s) saved in {os.path.abspath(args.output_dir)}")  # 输出保存的文件数量及目录