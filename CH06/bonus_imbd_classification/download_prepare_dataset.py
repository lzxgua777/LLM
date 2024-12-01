import os  # 导入os模块，用于操作文件和目录
import sys  # 导入sys模块，用于访问与Python解释器密切相关的变量和函数
import tarfile  # 导入tarfile模块，用于读取和写入tar文件
import time  # 导入time模块，用于时间相关的操作
import urllib.request  # 导入urllib.request模块，用于URL请求
import pandas as pd  # 导入pandas模块，用于数据处理


def reporthook(count, block_size, total_size):  # 定义一个回调函数，用于显示下载进度
    global start_time  # 使用global关键字声明start_time变量
    if count == 0:  # 如果是第一次调用
        start_time = time.time()  # 记录开始时间
    else:
        duration = time.time() - start_time  # 计算经过的时间
        progress_size = int(count * block_size)  # 计算已下载的数据量
        percent = count * block_size * 100 / total_size  # 计算下载进度百分比

        speed = int(progress_size / (1024 * duration)) if duration else 0  # 计算下载速度
        sys.stdout.write(  # 打印下载进度信息
            f"\r{int(percent)}% | {progress_size / (1024**2):.2f} MB "
            f"| {speed:.2f} MB/s | {duration:.2f} sec elapsed"
        )
        sys.stdout.flush()  # 刷新标准输出缓冲区


def download_and_extract_dataset(dataset_url, target_file, directory):  # 定义下载和解压数据集的函数
    if not os.path.exists(directory):  # 如果目标目录不存在
        if os.path.exists(target_file):  # 如果目标文件已存在，则删除
            os.remove(target_file)
        urllib.request.urlretrieve(dataset_url, target_file, reporthook)  # 下载文件，并使用reporthook显示进度
        print("\nExtracting dataset ...")  # 打印解压提示信息
        with tarfile.open(target_file, "r:gz") as tar:  # 打开tar.gz文件
            tar.extractall()  # 解压文件
    else:
        print(f"Directory `{directory}` already exists. Skipping download.")  # 如果目录已存在，则跳过下载


def load_dataset_to_dataframe(basepath="aclImdb", labels={"pos": 1, "neg": 0}):  # 定义加载数据集到DataFrame的函数
    data_frames = []  # 初始化一个空列表，用于存储每个数据块的DataFrame
    for subset in ("test", "train"):  # 遍历测试集和训练集
        for label in ("pos", "neg"):  # 遍历正面和负面标签
            path = os.path.join(basepath, subset, label)  # 构造文件路径
            for file in sorted(os.listdir(path)):  # 遍历目录中的文件
                with open(os.path.join(path, file), "r", encoding="utf-8") as infile:  # 打开文件
                    # 创建一个DataFrame，并将其添加到列表中
                    data_frames.append(pd.DataFrame({"text": [infile.read()], "label": [labels[label]]}))
    # 将所有DataFrame合并为一个
    df = pd.concat(data_frames, ignore_index=True)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)  # 打乱DataFrame
    return df  # 返回DataFrame


def partition_and_save(df, sizes=(35000, 5000, 10000)):  # 定义分割和保存DataFrame的函数
    # 打乱DataFrame
    df_shuffled = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 获取分割数据的索引
    train_end = sizes[0]  # 训练集大小
    val_end = sizes[0] + sizes[1]  # 验证集大小

    # 分割DataFrame
    train = df_shuffled.iloc[:train_end]  # 训练集
    val = df_shuffled.iloc[train_end:val_end]  # 验证集
    test = df_shuffled.iloc[val_end:]  # 测试集

    # 保存到CSV文件
    train.to_csv("train.csv", index=False)
    val.to_csv("validation.csv", index=False)
    test.to_csv("test.csv", index=False)


if __name__ == "__main__":  # 如果是主程序，则执行以下代码
    dataset_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"  # 数据集URL
    print("Downloading dataset ...")  # 打印下载提示信息
    download_and_extract_dataset(dataset_url, "aclImdb_v1.tar.gz", "aclImdb")  # 下载和解压数据集
    print("Creating data frames ...")  # 打印创建DataFrame的提示信息
    df = load_dataset_to_dataframe()  # 加载数据集到DataFrame
    print("Partitioning and saving data frames ...")  # 打印分割和保存DataFrame的提示信息
    partition_and_save(df)  # 分割和保存DataFrame