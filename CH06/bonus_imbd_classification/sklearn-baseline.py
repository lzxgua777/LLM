


import pandas as pd  # 导入 pandas 库，用于处理数据框（DataFrame）

# 加载训练、验证和测试数据集
train_df = pd.read_csv("train.csv")  # 从 train.csv 文件加载训练数据
val_df = pd.read_csv("validation.csv")  # 从 validation.csv 文件加载验证数据
test_df = pd.read_csv("test.csv")  # 从 test.csv 文件加载测试数据

# 显示训练数据的前 5 行，方便快速查看数据内容和结构
train_df.head()

# 导入用于文本向量化和模型训练的工具
from sklearn.feature_extraction.text import CountVectorizer  # 文本转为词频向量
from sklearn.linear_model import LogisticRegression  # 逻辑回归模型
from sklearn.metrics import accuracy_score, balanced_accuracy_score  # 计算分类模型的准确率

# 初始化 CountVectorizer，将文本转化为词频表示
vectorizer = CountVectorizer()

# 将训练集文本数据转化为词频向量，同时训练词典
X_train = vectorizer.fit_transform(train_df["text"])
# 使用训练好的词典将验证集文本数据转化为词频向量
X_val = vectorizer.transform(val_df["text"])
# 使用训练好的词典将测试集文本数据转化为词频向量
X_test = vectorizer.transform(test_df["text"])

# 将训练集、验证集和测试集的标签列提取出来
y_train, y_val, y_test = train_df["label"], val_df["label"], test_df["label"]

# 定义评估函数，计算模型的性能指标
def eval(model, X_train, y_train, X_val, y_val, X_test, y_test):
    # 使用模型对训练集、验证集和测试集进行预测
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # 计算训练集的准确率和加权准确率
    accuracy_train = accuracy_score(y_train, y_pred_train)
    balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)

    # 计算验证集的准确率和加权准确率
    accuracy_val = accuracy_score(y_val, y_pred_val)
    balanced_accuracy_val = balanced_accuracy_score(y_val, y_pred_val)

    # 计算测试集的准确率和加权准确率
    accuracy_test = accuracy_score(y_test, y_pred_test)
    balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)

    # 打印训练、验证和测试集的准确率
    print(f"Training Accuracy: {accuracy_train * 100:.2f}%")
    print(f"Validation Accuracy: {accuracy_val * 100:.2f}%")
    print(f"Test Accuracy: {accuracy_test * 100:.2f}%")

# 导入一个基准分类器
from sklearn.dummy import DummyClassifier  # 用于构建简单的基准模型

# 创建一个基准分类器，预测策略为选择训练集中出现频率最高的类别
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)  # 在训练数据集上拟合基准模型

# 使用评估函数评估基准模型的性能
eval(dummy_clf, X_train, y_train, X_val, y_val, X_test, y_test)

# 创建一个逻辑回归模型，并设置最大迭代次数为 1000
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)  # 在训练数据集上拟合逻辑回归模型

# 使用评估函数评估逻辑回归模型的性能
eval(model, X_train, y_train, X_val, y_val, X_test, y_test)
