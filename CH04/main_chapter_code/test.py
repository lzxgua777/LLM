# 版权所有 (c) Sebastian Raschka 在 Apache License 2.0 下（参见 LICENSE.txt）。
# “从头构建大型语言模型”的源代码
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch

# 此文件用于内部使用（单元测试）

from gpt import main

# 定义预期的输出字符串
expected = """
==================================================
                      输入
==================================================

输入文本：Hello, I am
编码后的输入文本：[15496, 11, 314, 716]
encoded_tensor.shape: torch.Size([1, 4])


==================================================
                      输出
==================================================

输出：tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267,
         49706, 43231, 47062, 34657]])
输出长度：14
输出文本：Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous
"""

# 定义测试main函数的测试函数
def test_main(capsys):
    # 调用main函数
    main()
    # 捕获标准输出
    captured = capsys.readouterr()

    # 将预期输出和实际输出的换行符标准化，并去除每行末尾的空白字符
    normalized_expected = '\n'.join(line.rstrip() for line in expected.splitlines())
    normalized_output = '\n'.join(line.rstrip() for line in captured.out.splitlines())

    # 比较标准化后的字符串
    assert normalized_output == normalized_expected
