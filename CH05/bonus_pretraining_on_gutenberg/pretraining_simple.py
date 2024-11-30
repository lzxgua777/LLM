# 版权所有 （c） Sebastian Raschka，Apache 许可证 2.0（参见 LICENSE.txt）。
# “从头开始构建大型语言模型” 的源代码
# - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch

"""
用于预训练小型 GPT-2 124M 参数模型的脚本
在 Project Gutenberg 的书籍上。

在运行此脚本之前，请确保您已下载并
按照 README.md 中所述处理数据集。
"""
import argparse  # 导入argparse模块，用于解析命令行参数
import os  # 导入os模块，用于操作文件和目录
from pathlib import Path  # 从pathlib模块导入Path类，用于路径操作
import time  # 导入time模块，用于时间管理
import tiktoken  # 导入tiktoken库，用于文本编码和解码
import torch  # 导入PyTorch库
from previous_chapters import (  # 从previous_chapters模块导入需要的函数和类
    create_dataloader_v1,  # 创建数据加载器的函数
    GPTModel,  # GPT模型类
    generate_and_print_sample,  # 用于生成并打印文本样本的函数
    calc_loss_batch,  # 计算每个批次损失的函数
    evaluate_model,  # 评估模型的函数
    plot_losses  # 绘制损失曲线的函数
)


# 读取文本文件的函数
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()  # 读取文件内容
    return text_data


# 创建训练集和验证集的数据加载器的函数
def create_dataloaders(text_data, train_ratio, batch_size, max_length, stride, num_workers=0):
    split_idx = int(train_ratio * len(text_data))  # 根据train_ratio计算训练集的分割位置
    train_loader = create_dataloader_v1(
        text_data[:split_idx],  # 训练集数据
        batch_size=batch_size,  # 每批次大小
        max_length=max_length,  # 最大长度
        stride=stride,  # 步幅
        drop_last=True,  # 丢弃最后一个不完整的批次
        shuffle=True,  # 打乱数据
        num_workers=num_workers  # 加载数据的线程数
    )
    val_loader = create_dataloader_v1(
        text_data[split_idx:],  # 验证集数据
        batch_size=batch_size,  # 每批次大小
        max_length=max_length,  # 最大长度
        stride=stride,  # 步幅
        drop_last=False,  # 不丢弃最后一个批次
        shuffle=False,  # 不打乱数据
        num_workers=num_workers  # 加载数据的线程数
    )
    return train_loader, val_loader  # 返回训练集和验证集的数据加载器


# 将秒数转换为小时、分钟和秒的函数
def convert_time(seconds):
    hours, rem = divmod(seconds, 3600)  # 将总秒数转换为小时和剩余秒数
    minutes, seconds = divmod(rem, 60)  # 将剩余秒数转换为分钟和秒
    return int(hours), int(minutes), int(seconds)  # 返回小时、分钟和秒


# 打印预计剩余时间（ETA）的函数
def print_eta(start_time, book_start_time, index, total_files):
    book_end_time = time.time()  # 获取当前时间，表示处理本书的结束时间
    elapsed_time = book_end_time - book_start_time  # 计算当前书籍的已用时间
    total_elapsed_time = book_end_time - start_time  # 计算从开始到现在的总时间
    books_remaining = total_files - index  # 计算剩余的书籍数量
    average_time_per_book = total_elapsed_time / index  # 计算每本书的平均处理时间
    eta = average_time_per_book * books_remaining  # 计算预计剩余时间

    # 将时间转换为小时、分钟、秒
    book_h, book_m, book_s = convert_time(elapsed_time)
    total_h, total_m, total_s = convert_time(total_elapsed_time)
    eta_h, eta_m, eta_s = convert_time(eta)

    # 打印书籍处理时间、总时间和预计剩余时间
    print(f"Book processed {book_h}h {book_m}m {book_s}s"
          f"\nTotal time elapsed {total_h}h {total_m}m {total_s}s"
          f"\nETA for remaining books: {eta_h}h {eta_m}m {eta_s}s")


# 简化的模型训练函数
def train_model_simple(model, optimizer, device, n_epochs,
                       eval_freq, eval_iter, print_sample_iter, start_context,
                       output_dir, save_ckpt_freq, tokenizer,
                       batch_size=1024, train_ratio=0.90):

    train_losses, val_losses, track_tokens_seen = [], [], []  # 初始化训练损失、验证损失和已处理的token数量
    tokens_seen = 0  # 初始化已处理的token数量
    global_step = -1  # 初始化全局步数
    start_time = time.time()  # 记录训练开始的时间

    try:
        # 遍历每个epoch
        for epoch in range(n_epochs):

            # 遍历训练集中的每一本书
            for index, file_path in enumerate(all_files, 1):
                book_start_time = time.time()  # 记录当前书籍的开始时间
                text_data = read_text_file(file_path) + " <|endoftext|> "  # 读取当前书籍的文本数据，并添加结束符
                print(f"Tokenizing file {index} of {total_files}: {file_path}")

                # 为每本书初始化数据加载器
                train_loader, val_loader = create_dataloaders(
                    text_data,
                    train_ratio=train_ratio,
                    batch_size=batch_size,
                    max_length=GPT_CONFIG_124M["context_length"],  # 使用GPT模型的上下文长度配置
                    stride=GPT_CONFIG_124M["context_length"],  # 步幅与上下文长度相同
                    num_workers=0  # 使用0个工作线程
                )
                print("Training ...")
                model.train()  # 设置模型为训练模式
                # 遍历训练数据
                for input_batch, target_batch in train_loader:
                    optimizer.zero_grad()  # 清空梯度
                    loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算损失
                    loss.backward()  # 反向传播
                    optimizer.step()  # 更新模型参数
                    tokens_seen += input_batch.numel()  # 更新已处理的token数量
                    global_step += 1  # 增加全局步数

                    # 每隔一定步数进行评估
                    if global_step % eval_freq == 0:
                        train_loss, val_loss = evaluate_model(
                            model, train_loader, val_loader, device, eval_iter)
                        train_losses.append(train_loss)  # 保存训练损失
                        val_losses.append(val_loss)  # 保存验证损失
                        track_tokens_seen.append(tokens_seen)  # 保存已处理的token数量
                        print(f"Ep {epoch+1} (Step {global_step}): "
                              f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                    # 每隔一定步数生成并打印样本
                    if global_step % print_sample_iter == 0:
                        generate_and_print_sample(
                            model, tokenizer, device, start_context
                        )

                # 每隔一定步数保存一次模型检查点
                if global_step % save_ckpt_freq:
                    file_name = output_dir / f"model_pg_{global_step}.pth"  # 设置保存模型的文件名
                    torch.save(model.state_dict(), file_name)  # 保存模型参数
                    print(f"Saved {file_name}")

                # 打印处理进度和预计剩余时间
                print_eta(start_time, book_start_time, index, total_files)
    # 除了KeyboardInterrupt异常（例如用户中断训练），保存当前模型状态
    except KeyboardInterrupt:
        file_name = output_dir / f"model_pg_{global_step}_interrupted.pth"
        torch.save(model.state_dict(), file_name)
        print(f"Saved {file_name}")

    # 返回训练和验证的损失以及观察到的token数量
    return train_losses, val_losses, track_tokens_seen

    # 如果这是主程序，则执行以下代码
if __name__ == "__main__":

        # 创建一个参数解析器来处理命令行参数
        parser = argparse.ArgumentParser(description='GPT Model Training Configuration')

        # 添加命令行参数
        parser.add_argument('--data_dir', type=str, default='gutenberg/data',
                            help='Directory containing the training data')
        parser.add_argument('--output_dir', type=str, default='model_checkpoints',
                            help='Directory where the model checkpoints will be saved')
        parser.add_argument('--n_epochs', type=int, default=1,
                            help='Number of epochs to train the model')
        parser.add_argument('--print_sample_iter', type=int, default=1000,
                            help='Iterations between printing sample outputs')
        parser.add_argument('--eval_freq', type=int, default=100,
                            help='Frequency of evaluations during training')
        parser.add_argument('--save_ckpt_freq', type=int, default=100_000,
                            help='Frequency of saving model checkpoints during training')
        parser.add_argument('--lr', type=float, default=5e-4,
                            help='Learning rate for the optimizer')
        parser.add_argument('--batch_size', type=int, default=4,
                            help='Batch size for training')
        parser.add_argument('--debug', type=bool, default=False,
                            help='Uses a very small model for debugging purposes')

        # 解析命令行参数
        args = parser.parse_args()

        # 如果启用调试模式，则使用较小的模型配置
        if args.debug:
            GPT_CONFIG_124M = {
                "vocab_size": 50257,  # 词汇表大小
                "context_length": 10,  # 上下文长度
                "emb_dim": 12,  # 嵌入维度
                "n_heads": 2,  # 注意力头数
                "n_layers": 2,  # 层数
                "drop_rate": 0.0,  # Dropout率，LLMs中不再推荐使用dropout，因此设置为0.0
                "qkv_bias": False  # Query-key-value偏置
            }

        # 否则，使用完整的模型配置
        else:
            GPT_CONFIG_124M = {
                "vocab_size": 50257,  # 词汇表大小
                "context_length": 1024,  # 上下文长度
                "emb_dim": 768,  # 嵌入维度
                "n_heads": 12,  # 注意力头数
                "n_layers": 12,  # 层数
                "drop_rate": 0.1,  # Dropout率
                "qkv_bias": False  # Query-key-value偏置
            }

        # 设置设备，优先使用GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 设置随机种子以确保结果可复现
        torch.manual_seed(123)
        # 创建GPT模型实例
        model = GPTModel(GPT_CONFIG_124M)
        # 将模型发送到设备（GPU或CPU）
        model.to(device)
        # 创建优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
        # 获取tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")

        # 设置数据目录
        data_dir = args.data_dir
        # 搜索数据目录下的所有.txt文件
        all_files = [os.path.join(path, name) for path, subdirs, files
                     in os.walk(data_dir) for name in files if name.endswith((".txt"))]
        # 计算文件总数
        total_files = len(all_files)

        # 如果没有找到文件，则打印错误信息并退出
        if total_files == 0:
            print("No training text files found. Make sure you "
                  "selected the correct input directory")
            quit()
        # 打印文件总数
        print("Total files:", total_files)

        # 设置输出目录并确保其存在
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 调用train_model_simple函数训练模型
        train_losses, val_losses, tokens_seen = train_model_simple(
            model, optimizer, device,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            eval_freq=args.eval_freq,
            eval_iter=1,
            print_sample_iter=args.print_sample_iter,
            output_dir=output_dir,
            save_ckpt_freq=args.save_ckpt_freq,
            start_context="Every effort moves you",
            tokenizer=tokenizer
        )

        # 创建一个表示epoch的张量，用于绘图
        epochs_tensor = torch.linspace(0, args.n_epochs, len(train_losses))
        # 绘制损失曲线
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, output_dir)

        # 保存最终模型状态
        torch.save(model.state_dict(), output_dir / "model_pg_final.pth")
        # 打印GPU最大内存分配量
        print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
