# 版权所有 （c） Sebastian Raschka，Apache 许可证 2.0（参见 LICENSE.txt）。
# “从头开始构建大型语言模型” 的源代码
# - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch

# 供内部使用的文件（单元测试）

from pathlib import Path  # 导入Path类，用于操作路径
import os  # 导入os模块，用于操作文件和目录
import subprocess  # 导入subprocess模块，用于执行外部命令


# 测试预训练过程的函数
def test_pretraining():

    # 定义一个简单的字符串序列
    sequence = "a b c d"
    repetitions = 1000  # 重复1000次
    content = sequence * repetitions  # 将字符串序列重复1000次生成测试数据

    # 设置保存数据的文件夹路径
    folder_path = Path("gutenberg") / "data"
    file_name = "repeated_sequence.txt"  # 设置要保存的文件名

    # 如果文件夹不存在，则创建它
    os.makedirs(folder_path, exist_ok=True)

    # 打开文件并将生成的内容写入文件
    with open(folder_path / file_name, "w") as file:
        file.write(content)

    # 执行外部Python脚本 pretraining_simple.py 进行预训练，并捕获其输出
    result = subprocess.run(
        ["python", "pretraining_simple.py", "--debug", "true"],  # 执行命令
        capture_output=True,  # 捕获标准输出和标准错误
        text=True  # 以文本形式返回输出（而非字节）
    )
    # 打印执行结果的标准输出
    print(result.stdout)

    # 确保输出中包含"Maximum GPU memory allocated"字符串，验证训练过程是否如预期进行
    assert "Maximum GPU memory allocated" in result.stdout
