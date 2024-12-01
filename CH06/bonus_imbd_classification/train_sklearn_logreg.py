# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# 来源于 "Build a Large Language Model From Scratch"
# 书籍链接: https://www.manning.com/books/build-a-large-language-model-from-scratch
# 源代码链接: https://github.com/rasbt/LLMs-from-scratch

# 导入必要的库
import pandas as pd  # 用于数据处理和操作
from sklearn.feature_extraction.text import CountVectorizer  # 将文本转换为词袋模型（Bag-of-Words）
from sklearn.linear_model import LogisticRegression  # 逻辑回归模型
from sklearn.metrics import accuracy_score  # 用于计算模型的准确率
# from sklearn.metrics import balanced_accuracy_score  # 平衡准确率（此处未使用，可选）
from sklearn.dummy import DummyClassifier  # 简单的基线分类器

# 定义一个函数，用于加载训练、验证和测试数据
def load_dataframes():
    # 读取训练数据
    df_train = pd.read_csv("train.csv")
    # 读取验证数据
    df_val = pd.read_csv("validation.csv")
    # 读取测试数据
    df_test = pd.read_csv("test.csv")

    # 返回三个数据集
    return df_train, df_val, df_test


# 定义评估函数
def eval(model, X_train, y_train, X_val, y_val, X_test, y_test):
    # 使用模型对训练集、验证集和测试集进行预测
    y_pred_train = model.predict(X_train)  # 预测训练集标签
    y_pred_val = model.predict(X_val)      # 预测验证集标签
    y_pred_test = model.predict(X_test)    # 预测测试集标签

    # 计算准确率
    accuracy_train = accuracy_score(y_train, y_pred_train)  # 训练集准确率
    # balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)  # 平衡准确率（注释掉）

    accuracy_val = accuracy_score(y_val, y_pred_val)        # 验证集准确率
    # balanced_accuracy_val = balanced_accuracy_score(y_val, y_pred_val)  # 平衡准确率（注释掉）

    accuracy_test = accuracy_score(y_test, y_pred_test)     # 测试集准确率
    # balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)  # 平衡准确率（注释掉）

    # 打印结果
    print(f"Training Accuracy: {accuracy_train*100:.2f}%")  # 打印训练集准确率
    print(f"Validation Accuracy: {accuracy_val*100:.2f}%")  # 打印验证集准确率
    print(f"Test Accuracy: {accuracy_test*100:.2f}%")       # 打印测试集准确率

    # 如果需要，也可以打印平衡准确率（此处被注释掉）
    # print(f"\nTraining Balanced Accuracy: {balanced_accuracy_train*100:.2f}%")
    # print(f"Validation Balanced Accuracy: {balanced_accuracy_val*100:.2f}%")
    # print(f"Test Balanced Accuracy: {balanced_accuracy_test*100:.2f}%")


# 主程序入口
if __name__ == "__main__":
    # 加载数据集
    df_train, df_val, df_test = load_dataframes()

    #########################################
    # 将文本转换为词袋模型
    vectorizer = CountVectorizer()  # 创建 CountVectorizer 对象
    #########################################

    # 通过训练集生成词袋模型并转换文本数据为稀疏矩阵
    X_train = vectorizer.fit_transform(df_train["text"])
    # 直接使用训练好的词袋模型转换验证集和测试集
    X_val = vectorizer.transform(df_val["text"])
    X_test = vectorizer.transform(df_test["text"])
    # 提取标签
    y_train, y_val, y_test = df_train["label"], df_val["label"], df_test["label"]

    #####################################
    # 模型训练和评估
    #####################################

    # 创建一个基线分类器，策略为预测最频繁的类别
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)  # 用训练数据拟合基线分类器

    print("Dummy classifier:")  # 打印基线分类器结果标题
    eval(dummy_clf, X_train, y_train, X_val, y_val, X_test, y_test)  # 评估基线分类器性能

    print("\n\nLogistic regression classifier:")  # 打印逻辑回归标题
    # 创建逻辑回归模型，设置最大迭代次数为1000
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)  # 用训练数据拟合逻辑回归模型
    eval(model, X_train, y_train, X_val, y_val, X_test, y_test)  # 评估逻辑回归模型性能
