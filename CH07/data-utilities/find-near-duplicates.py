# 版权声明：Sebastian Raschka 根据 Apache License 2.0（参见 LICENSE.txt）持有版权。
# "Build a Large Language Model From Scratch" 的来源
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch

import argparse  # 导入argparse库，用于解析命令行参数
import json  # 导入json库，用于处理JSON数据
import re  # 导入re库，用于正则表达式操作
from sklearn import __version__ as sklearn_version  # 导入scikit-learn的版本号
from sklearn.feature_extraction.text import TfidfVectorizer  # 导入TF-IDF向量化器
from sklearn.metrics.pairwise import cosine_similarity  # 导入余弦相似度计算函数

# 示例JSON数据集
example_data = [
    {"instruction": "What is the capital of Italy?",
     "input": "", "output": "The capital of Italy is Rome."
     },
    {"instruction": "What's the capital city of Italy?",
     "input": "", "output": "The capital city is Rome."
     },
    {"instruction": "Identify the main verb in the sentence: 'The cat sleeps on the couch.'",
     "input": "", "output": "The verb is 'sleeps'."
     },
    {"instruction": "Identify the verb in the following sentence: The cat sleeps on the couch.",
     "input": "", "output": "The verb in the sentence is \"sleeps.\""
     },
    # ...
]

# 定义文本预处理函数


def preprocess_text(text):
    # 将文本转换为小写
    text = text.lower()
    # 移除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    return text

    # 定义查找近似重复项的函数


def find_near_duplicates(json_data, threshold=0.75, key="instruction"):
    """阈值越高，匹配的文本必须越相似"""

    # 提取指令
    text = [preprocess_text(item[key]) for item in json_data if item[key]]
    near_duplicates = []  # 存储近似重复项
    indices_to_remove = set()  # 存储需要移除的索引

    if not text:  # 如果文本列表为空
        return {}, near_duplicates

    # 向量化文本数据
    vectorizer = TfidfVectorizer(stop_words=None, analyzer='char', ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(text)

    # 计算每对条目的余弦相似度
    cos_sim_matrix = cosine_similarity(tfidf_matrix)

    # 根据阈值查找近似重复的指令

    for i in range(len(cos_sim_matrix)):
        for j in range(i + 1, len(cos_sim_matrix)):
            if cos_sim_matrix[i, j] > threshold:
                if len(json_data[i][key]) <= 1 or len(json_data[j][key]) <= 1:
                    continue
                near_duplicates.append((json_data[i], json_data[j], cos_sim_matrix[i, j]))
                if key in ("input", "output"):  # 不基于指令移除重复项
                    indices_to_remove.add(j)  # 标记第二个条目以移除

    # 移除近似重复的条目
    filtered_json_data = [item for index, item in enumerate(json_data) if index not in indices_to_remove]

    return filtered_json_data, near_duplicates

    # 定义查找、打印并移除近似重复项的函数


def find_print_and_remove_near_duplicates(json_data, remove_duplicates=False, threshold=0.75):
    """
    在JSON对象列表中搜索每个键的重复项。
    如果找到重复项，则打印出来。
    """
    for key in json_data[0].keys():
        if remove_duplicates:
            json_data, near_duplicates = find_near_duplicates(json_data, key=key, threshold=threshold)
        else:
            _, near_duplicates = find_near_duplicates(json_data, key=key, threshold=threshold)
        separator = 50 * '='  # 分隔符
        print(f"\n\n{separator}\nSearching '{key}' for duplicates ...\n{separator}")
        if not near_duplicates:
            print("No duplicates found")
        else:
            for dup in near_duplicates:
                print(
                    f"Duplicate pair found with similarity {dup[2]:.2f}:\n"
                    f"1. {dup[0][key]}\n2. {dup[1][key]}\n"
                )
    return json_data

    # 如果是主程序，则执行以下代码


if __name__ == "__main__":
    print("scikit-learn version:", sklearn_version)  # 打印scikit-learn版本号

    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument(
        "--json_file",
        type=str,
        help=("Path to the dataset JSON file")
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help=("A sensitivity threshold between 0 and 1 where 1 is strictest")
    )
    parser.add_argument(
        "--remove_duplicates",
        action='store_true',
        default=False,
        help=(
            "Removes duplicates based on the 'input' or 'output' keys "
            " (but not the 'instruction') and saves the cleaned JSON file as --json_output_file"
        )
    )
    parser.add_argument(
        "--json_output_file",
        type=str,
        help=("Path to the dataset JSON file")
    )

    args = parser.parse_args()  # 解析参数

    if args.remove_duplicates and not args.json_output_file:
        raise ValueError(
            "Provide an output file via --json_output_file "
            "to save the cleaned JSON data."
        )

    if not args.json_file:  # 如果没有指定JSON文件路径
        json_data = example_data  # 使用示例数据

    else:
        with open(args.json_file, "r") as file:  # 读取指定的JSON文件
            json_data = json.load(file)

    json_data = find_print_and_remove_near_duplicates(  # 查找、打印并移除近似重复项
        json_data=json_data,
        remove_duplicates=args.remove_duplicates,
        threshold=args.threshold
    )

    if args.remove_duplicates:  # 如果需要移除重复项
        with open(args.json_output_file, "w") as file:  # 写入清理后的JSON数据到文件
            json.dump(json_data, file, indent=4)