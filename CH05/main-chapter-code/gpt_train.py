# 版权所有 （c） Sebastian Raschka，Apache 许可证 2.0（参见 LICENSE.txt）。
# “从头开始构建大型语言模型” 的源代码
# - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch
import matplotlib.pyplot as plt
import os
import torch
import urllib.request
import tiktoken

# 从本地文件导入
from ch04.main_chapter_code.ch04 import GPTModel, generate_text_simple
from ch04.main_chapter_code.gpt import create_dataloader_v1

# 将文本转换为token ID
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)  # 使用tokenizer对文本进行编码，将其转换为token的ID
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 转换为Tensor，并添加一个batch维度
    return encoded_tensor  # 返回处理后的tensor

# 将token ID转换为文本
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # 删除batch维度
    return tokenizer.decode(flat.tolist())  # 使用tokenizer解码为文本

# 计算一批数据的损失
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将数据转移到指定设备
    logits = model(input_batch)  # 使用模型计算输出
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())  # 计算交叉熵损失
    return loss  # 返回损失

# 计算数据加载器中的损失
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.  # 初始化总损失
    if len(data_loader) == 0:
        return float("nan")  # 如果数据加载器为空，返回NaN
    elif num_batches is None:
        num_batches = len(data_loader)  # 如果没有指定批次数量，使用数据加载器的总长度
    else:
        num_batches = min(num_batches, len(data_loader))  # 使用最小值来限制批次数量
    for i, (input_batch, target_batch) in enumerate(data_loader):  # 遍历数据加载器
        if i < num_batches:  # 如果当前批次小于设定的批次数量
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算当前批次的损失
            total_loss += loss.item()  # 累加损失
        else:
            break  # 达到批次数量限制时退出循环
    return total_loss / num_batches  # 返回平均损失

# 评估模型的训练和验证损失
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不需要计算梯度
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)  # 计算训练损失
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)  # 计算验证损失
    model.train()  # 设置模型回到训练模式
    return train_loss, val_loss  # 返回训练和验证损失

# 生成并打印样本
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()  # 设置模型为评估模式
    context_size = model.pos_emb.weight.shape[0]  # 获取位置嵌入的维度
    encoded = text_to_token_ids(start_context, tokenizer).to(device)  # 将初始文本转换为token ID
    with torch.no_grad():  # 不需要计算梯度
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )  # 生成50个新的token
        decoded_text = token_ids_to_text(token_ids, tokenizer)  # 将token ID转换为文本
        print(decoded_text.replace("\n", " "))  # 打印生成的文本，换行符被替换为空格
    model.train()  # 设置模型回到训练模式

# 简单训练模型的函数
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # 初始化用于跟踪损失和tokens数量的列表
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0  # 已处理的token数量
    global_step = -1  # 全局训练步数（计数器）

    # 主训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 清除上一步的梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算当前批次的损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
            tokens_seen += input_batch.numel()  # 累计处理的tokens数量
            global_step += 1  # 增加训练步数

            # 可选的评估步骤
            if global_step % eval_freq == 0:  # 每隔一定步数进行一次评估
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)  # 记录训练损失
                val_losses.append(val_loss)  # 记录验证损失
                track_tokens_seen.append(tokens_seen)  # 记录已处理的tokens数量
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 每个epoch结束后打印一条样本
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen  # 返回训练过程中的损失和tokens数量

# 绘制损失图表的函数
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()  # 创建一个图形对象和一个坐标轴

    # 绘制训练损失和验证损失随epoch变化的曲线
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")  # 设置x轴标签为Epochs（训练轮数）
    ax1.set_ylabel("Loss")  # 设置y轴标签为Loss（损失）
    ax1.legend(loc="upper right")  # 显示图例

    # 创建一个共享y轴的第二个x轴，用于显示已处理的tokens数量
    ax2 = ax1.twiny()  # 创建一个共享y轴的新x轴
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 绘制一条不可见的曲线来对齐x轴刻度
    ax2.set_xlabel("Tokens seen")  # 设置第二个x轴标签为Tokens seen（已处理tokens）

    fig.tight_layout()  # 调整布局，以避免标签重叠
    # plt.show()  # 可选的显示图表，当前代码被注释掉

def main(gpt_config, settings):

    # 设置随机种子以确保结果的可重复性
    torch.manual_seed(123)
    # 设置设备为GPU（如果可用），否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################
    # 下载数据（如果需要的话）
    ##############################
    file_path = "the-verdict.txt"  # 目标文件路径
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"  # 数据来源URL

    # 如果本地没有该文件，则从网上下载
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:  # 请求URL并读取数据
            text_data = response.read().decode('utf-8')  # 解码为UTF-8格式
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)  # 保存数据到文件
    else:
        # 如果文件已存在，直接读取
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()  # 读取文本数据

    ##############################
    # 初始化模型
    ##############################
    # 使用配置文件初始化GPT模型
    model = GPTModel(gpt_config)
    model.to(device)  # 将模型转移到指定的设备（GPU或CPU）
    # 使用AdamW优化器，并传入学习率和权重衰减参数
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    ##############################
    # 设置数据加载器
    ##############################
    # 训练集和验证集的比例设置为90%训练，10%验证
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))  # 根据比例计算训练集和验证集的分割索引

    # 创建训练集数据加载器
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],  # 步长设置为与上下文长度相同
        drop_last=True,  # 丢弃最后不完整的批次
        shuffle=True,  # 打乱数据
        num_workers=0  # 不使用多线程加载
    )

    # 创建验证集数据加载器
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,  # 不丢弃最后一个不完整的批次（验证集可以有一个较小的批次）
        shuffle=False,  # 不打乱验证集数据
        num_workers=0
    )

    ##############################
    # 训练模型
    ##############################
    # 获取GPT-2的tokenizer，用于文本的编码和解码
    tokenizer = tiktoken.get_encoding("gpt2")

    # 调用训练函数进行训练，并返回训练损失、验证损失和已处理的tokens数量
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, model  # 返回训练过程中的信息


# 程序入口
if __name__ == "__main__":

    # GPT配置（124M模型）
    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # 词汇表大小
        "context_length": 256,  # 上下文长度（原始值为1024，缩短为256）
        "emb_dim": 768,         # 嵌入维度
        "n_heads": 12,          # 注意力头数
        "n_layers": 12,         # Transformer层数
        "drop_rate": 0.1,       # Dropout率
        "qkv_bias": False       # 是否使用QKV偏置
    }

    # 其他设置，包括学习率、训练轮数、批次大小等
    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1
    }

    ###########################
    # 启动训练
    ###########################
    # 调用main函数开始训练
    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)

    ###########################
    # 训练后操作
    ###########################

    # 绘制损失图
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))  # 生成训练轮数的张量
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)  # 绘制训练损失和验证损失
    plt.savefig("loss.pdf")  # 将图保存为PDF文件

    # 保存并加载模型
    torch.save(model.state_dict(), "model.pth")  # 保存模型的参数
    model = GPTModel(GPT_CONFIG_124M)  # 重新初始化模型
    model.load_state_dict(torch.load("model.pth"), weights_only=True)  # 加载保存的模型参数
