# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# 这是一个内部使用的文件（单元测试）

import subprocess  # 导入subprocess模块，用于执行外部命令

def test_gpt_class_finetune():  # 定义一个函数，用于测试GPT模型的微调
    command = ["python", "ch06/01_main-chapter-code/gpt_class_finetune.py", "--test_mode"]  # 定义要执行的命令
    # 这里指定了Python解释器的路径，脚本的位置，以及一个标志来指示测试模式

    result = subprocess.run(command, capture_output=True, text=True)  # 执行命令并捕获输出
    # subprocess.run() 函数用于运行命令，capture_output=True 表示捕获输出，text=True 表示输出为文本格式

    assert result.returncode == 0, f"Script exited with errors: {result.stderr}"  # 断言脚本执行成功
    # assert 语句用于检查脚本是否成功执行（返回码为0），如果失败，则打印错误信息
    # result.returncode 是脚本的返回码，result.stderr 包含标准错误输出的内容
