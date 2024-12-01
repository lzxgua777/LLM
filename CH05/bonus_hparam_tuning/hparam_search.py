# 版权所有 （c） Sebastian Raschka，Apache 许可证 2.0（参见 LICENSE.txt）。
# “从头开始构建大型语言模型” 的源代码
# - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch
# 导入所需的库
import itertools
import math
import os
import tiktoken
import torch
from ch04.main_chapter_code.ch04 import GPTModel
from ch04.main_chapter_code.gpt import create_dataloader_v1

# 定义一个超参数网格，用于搜索最佳参数组合
HPARAM_GRID = {
    "batch_size": [2, 4, 8, 16],  # 批处理大小的不同选项
    "drop_rate": [0.0, 0.1, 0.2],  # Dropout率的不同选项
    "warmup_iters": [10, 20, 30],  # 预热迭代次数的不同选项
    "weight_decay": [0.1, 0.01, 0.0],  # 权重衰减的不同选项
    "peak_lr": [0.0001, 0.0005, 0.001, 0.005],  # 最高学习率的不同选项
    "initial_lr": [0.00005, 0.0001],  # 初始学习率的不同选项
    "min_lr": [0.00005, 0.00001, 0.0001],  # 最低学习率的不同选项
    "n_epochs": [5, 10, 15, 20, 25],  # 训练周期的不同选项
}

# 计算数据加载器的损失
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.  # 初始化总损失为0
    if len(data_loader) == 0:  # 如果数据加载器为空，返回NaN
        return float("nan")
    elif num_batches is None:  # 如果没有指定批次数量，则使用数据加载器的全部批次
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))  # 如果指定的批次数量大于数据加载器的批次数量，则取较小值
    for i, (input_batch, target_batch) in enumerate(data_loader):  # 遍历数据加载器中的批次
        if i < num_batches:  # 如果当前批次小于指定的批次数量
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算当前批次的损失
            total_loss += loss.item()  # 将当前批次的损失加到总损失中
        else:
            break  # 如果已经处理了指定的批次数量，则退出循环
    return total_loss / num_batches  # 返回平均损失

# 计算单个批次的损失
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将输入和目标数据移动到指定的设备

    logits = model(input_batch)  # 通过模型计算得到logits
    logits = logits.view(-1, logits.size(-1))  # 调整logits的形状以匹配目标数据
    loss = torch.nn.functional.cross_entropy(logits, target_batch.view(-1))  # 计算交叉熵损失
    return loss  # 返回损失

# 评估模型
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 在不计算梯度的情况下执行
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)  # 计算训练集的损失
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)  # 计算验证集的损失
    model.train()  # 将模型设置回训练模式
    return train_loss, val_loss  # 返回训练集和验证集的损失

# 训练模型
def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter,
                encoded_start_context, tokenizer, warmup_iters=10,
                initial_lr=3e-05, min_lr=1e-6):
    global_step = 0  # 初始化全局步数为0

    max_lr = optimizer.param_groups[0]["lr"]  # 获取最大学习率

    # 计算总迭代次数
    total_training_iters = len(train_loader) * n_epochs

    # 计算预热期间每一步的学习率增量
    lr_increment = (optimizer.param_groups[0]["lr"] - initial_lr) / warmup_iters

    for epoch in range(n_epochs):  # 遍历每个训练周期
        model.train()  # 将模型设置为训练模式
        for input_batch, target_batch in train_loader:  # 遍历训练数据加载器中的批次
            optimizer.zero_grad()  # 清零梯度

            # 在迭代开始时增加全局步数
            global_step += 1

            # 预热：线性调整学习率
            if global_step <= warmup_iters:
                lr = initial_lr + global_step * lr_increment  # 计算当前学习率
            # 余弦退火阶段
            else:
                progress = (global_step - warmup_iters) / (total_training_iters - warmup_iters)  # 计算进度
                lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))  # 计算当前学习率

            # 应用计算出的学习率
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr  # 更新学习率

            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算损失
            loss.backward()  # 反向传播

            # 应用梯度裁剪
            if global_step >= warmup_iters:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 裁剪梯度

            optimizer.step()  # 优化器更新参数

    # 在训练结束后评估模型
    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)

    return train_loss, val_loss  # 返回训练集和验证集的损失

if __name__ == "__main__":  # 如果是直接运行此脚本，执行以下代码块

    # 生成所有超参数的组合
    hyperparameter_combinations = list(itertools.product(*HPARAM_GRID.values()))  # 获取超参数网格的所有组合
    total_combinations = len(hyperparameter_combinations)  # 计算组合的总数
    print(f"Total hyperparameter configurations: {total_combinations}")  # 输出总的超参数组合数

    # 用于保存最佳验证损失和最佳超参数的占位符
    best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大
    best_hparams = {}  # 初始化最佳超参数为空字典

    # 获取当前脚本的路径和目录
    script_path = os.path.abspath(__file__)  # 获取脚本的绝对路径
    script_dir = os.path.dirname(script_path)  # 获取脚本所在的目录
    with open(os.path.join(script_dir, "the-verdict.txt"), "r", encoding="utf-8") as file:
        text_data = file.read()  # 读取文件内容

    tokenizer = tiktoken.get_encoding("gpt2")  # 获取GPT-2的tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为GPU（如果可用），否则为CPU

    train_ratio = 0.95  # 设置训练集占数据的比例为95%
    split_idx = int(train_ratio * len(text_data))  # 根据比例计算训练集的分割索引

    torch.manual_seed(123)  # 设置PyTorch的随机种子，以确保可复现性

    interrupted = False  # 标记是否中断
    current_config = 0  # 当前配置编号
    for combination in hyperparameter_combinations:  # 遍历所有的超参数组合

        try:
            current_config += 1  # 增加当前配置的编号
            print(f"Evaluating configuration {current_config} of {total_combinations}")  # 输出当前配置的评估进度

            # 解包当前超参数组合
            HPARAM_CONFIG = dict(zip(HPARAM_GRID.keys(), combination))  # 将当前超参数组合与超参数网格的键对应起来

            GPT_CONFIG_124M = {
                "vocab_size": 50257,    # 词汇表大小
                "context_length": 256,  # 上下文长度（从原始的1024个token减少为256）
                "emb_dim": 768,         # 嵌入维度
                "n_heads": 12,          # 注意力头的数量
                "n_layers": 12,         # 层数
                "drop_rate": HPARAM_CONFIG["drop_rate"],  # 从超参数配置中获取dropout率
                "qkv_bias": False,     # 是否使用QKV的偏置
            }

            torch.manual_seed(123)  # 每次开始新配置时重置随机种子
            # 创建训练数据加载器
            train_loader = create_dataloader_v1(
                text_data[:split_idx],  # 使用训练集的数据
                batch_size=HPARAM_CONFIG["batch_size"],  # 从超参数配置中获取批次大小
                max_length=GPT_CONFIG_124M["context_length"],  # 使用GPT模型的上下文长度配置
                stride=GPT_CONFIG_124M["context_length"],  # 步幅与上下文长度相同
                drop_last=True,  # 丢弃最后一个不完整的批次
                shuffle=True,  # 打乱数据
                num_workers=0  # 使用0个工作线程
            )

            # 创建验证数据加载器
            val_loader = create_dataloader_v1(
                text_data[split_idx:],  # 使用验证集的数据
                batch_size=HPARAM_CONFIG["batch_size"],  # 从超参数配置中获取批次大小
                max_length=GPT_CONFIG_124M["context_length"],  # 使用GPT模型的上下文长度配置
                stride=GPT_CONFIG_124M["context_length"],  # 步幅与上下文长度相同
                drop_last=False,  # 不丢弃最后一个批次
                shuffle=False,  # 不打乱数据
                num_workers=0  # 使用0个工作线程
            )

            # 初始化GPT模型
            model = GPTModel(GPT_CONFIG_124M)
            model.to(device)  # 将模型移到指定的设备（GPU或CPU）

            # 使用AdamW优化器
            optimizer = torch.optim.AdamW(
                model.parameters(),  # 优化器优化模型的参数
                lr=HPARAM_CONFIG["peak_lr"],  # 从超参数配置中获取峰值学习率
                weight_decay=HPARAM_CONFIG["weight_decay"]  # 从超参数配置中获取权重衰减
            )

            encoded_start_context = tokenizer.encode("Nevertheless")  # 编码一个起始上下文
            encoded_tensor = torch.tensor(encoded_start_context).unsqueeze(0)  # 将编码的上下文转换为张量，并增加一个批次维度

            # 训练模型并返回训练损失和验证损失
            train_loss, val_loss = train_model(
                model, train_loader, val_loader, optimizer, device,
                n_epochs=HPARAM_CONFIG["n_epochs"],  # 从超参数配置中获取训练周期数
                eval_freq=5, eval_iter=1,  # 每5步进行一次评估
                encoded_start_context=encoded_tensor,  # 编码的起始上下文
                tokenizer=tokenizer,  # Tokenizer
                warmup_iters=HPARAM_CONFIG["warmup_iters"],  # 从超参数配置中获取warmup迭代次数
                initial_lr=HPARAM_CONFIG["initial_lr"],  # 从超参数配置中获取初始学习率
                min_lr=HPARAM_CONFIG["min_lr"]  # 从超参数配置中获取最小学习率
            )

            # 根据验证损失记录最佳超参数
            if val_loss < best_val_loss:  # 如果当前验证损失更小，则更新最佳超参数
                best_val_loss = val_loss  # 更新最佳验证损失
                best_train_loss = train_loss  # 更新最佳训练损失
                best_hparams = HPARAM_CONFIG  # 更新最佳超参数

        except KeyboardInterrupt:  # 如果手动中断训练
            print("Hyperparameter search completed.")  # 输出搜索已完成
            print(f"Best hyperparameters: {best_hparams}")  # 输出最佳超参数
            print(f"Best Val loss: {best_val_loss} | Training loss {train_loss}")  # 输出最佳验证损失和训练损失
            interrupted = True  # 标记为中断
            break  # 跳出循环

    if not interrupted:  # 如果没有被中断
        print("Hyperparameter search completed.")  # 输出搜索已完成
        print(f"Best hyperparameters: {best_hparams}")  # 输出最佳超参数
        print(f"Best Val loss: {best_val_loss} | Training loss {train_loss}")  # 输出最佳验证损失和训练损失
