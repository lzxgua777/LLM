# 版权声明：Sebastian Raschka 根据 Apache License 2.0（参见 LICENSE.txt）持有版权。
# "Build a Large Language Model From Scratch" 的来源
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch

# 这个文件用于内部测试

import subprocess  # 导入 subprocess 模块，用于执行外部命令


# 定义一个测试函数，用于测试 GPT 模型微调的脚本
def test_gpt_class_finetune():
    # 定义要执行的命令，包括 Python 解释器、脚本路径和测试模式参数
    command = ["python", "ch06/01_main-chapter-code/gpt_class_finetune.py", "--test_mode"]

    # 使用 subprocess.run 执行命令，并捕获输出结果和错误信息
    result = subprocess.run(command, capture_output=True, text=True)

    # 断言返回码为 0，表示脚本执行成功；如果返回码非 0，则打印错误信息
    assert result.returncode == 0, f"Script exited with errors: {result.stderr}"
