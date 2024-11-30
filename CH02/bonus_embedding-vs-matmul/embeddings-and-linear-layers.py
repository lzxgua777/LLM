
import torch

print("PyTorch version:", torch.__version__)        # 打印PyTorch的版本


# 假设我们有以下3个训练样本，它们可能代表LLM（大型语言模型）上下文中的token ID
idx = torch.tensor([2, 3, 1])

# 通过获取最大的token ID + 1，可以确定嵌入矩阵的行数
# 如果最高的token ID是3，那么我们想要4行，用于可能的token ID 0, 1, 2, 3
num_idx = max(idx)+1

# 期望的嵌入维度是一个超参数
out_dim = 5

# 我们使用随机种子以确保可重复性，因为嵌入层中的权重是用小的随机值初始化的
torch.manual_seed(123)
# 创建嵌入层
embedding = torch.nn.Embedding(num_idx, out_dim)

print(embedding.weight)             # 打印嵌入层的权重
print(embedding(torch.tensor([1]))) # 打印嵌入层对特定token ID的嵌入向量
print(embedding(torch.tensor([2])))
idx = torch.tensor([2, 3, 1])       # 重新定义idx变量
print(embedding(idx))               # 打印嵌入层对idx中所有token ID的嵌入向量

onehot = torch.nn.functional.one_hot(idx)   # 使用one_hot函数将idx转换为one-hot编码
print(onehot)                               # 打印one-hot编码结果


torch.manual_seed(123)                                  # 重新设置随机种子
linear = torch.nn.Linear(num_idx, out_dim, bias=False)  # 创建线性层，不使用偏置项
print(linear.weight)                                    # 打印线性层的权重

linear.weight = torch.nn.Parameter(embedding.weight.T)  # 将嵌入层的权重转置后赋值给线性层的权重
print(linear(onehot.float()))                           # 打印使用线性层和one-hot编码得到的输出

print(embedding(idx))                                   # 打印嵌入层对idx中所有token ID的嵌入向量，与之前相同

