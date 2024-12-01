# 版权所有 （c） Sebastian Raschka，Apache 许可证 2.0（参见 LICENSE.txt）。
# “从头开始构建大型语言模型” 的源代码
# - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch
#
# 此文件收集了我们到目前为止介绍的所有相关代码
# 贯穿第 2-6 章。
# 此文件可以作为独立脚本运行。

import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘图
from matplotlib.ticker import MaxNLocator  # 导入matplotlib的MaxNLocator，用于设置坐标轴刻度
import numpy as np  # 导入numpy库，用于进行科学计算
import tiktoken  # 导入tiktoken库，用于处理GPT模型的分词
import torch  # 导入PyTorch库，用于深度学习模型的构建和训练
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch的数据集和数据加载器模块

#####################################
# Chapter 2
####################################

# 定义GPTDatasetV1类，用于处理GPT模型的数据集
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer  # 分词器
        self.input_ids = []  # 输入的标记ID列表
        self.target_ids = []  # 目标的标记ID列表

        # 对整个文本进行分词
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分割成长度为max_length的重叠序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 输入序列
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标序列
            self.input_ids.append(torch.tensor(input_chunk))  # 将输入序列转换为张量并添加到列表
            self.target_ids.append(torch.tensor(target_chunk))  # 将目标序列转换为张量并添加到列表

    def __len__(self):
        return len(self.input_ids)  # 返回数据集中的样本数量

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]  # 根据索引返回对应的输入和目标序列

# 定义create_dataloader_v1函数，用于创建GPT数据集的数据加载器
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

#####################################
# Chapter 3
#####################################

# 定义多头注意力机制类
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out 必须能被 n_heads 整除"

        self.d_out = d_out  # 输出维度
        self.num_heads = num_heads  # 头的数量
        self.head_dim = d_out // num_heads  # 每个头的维度

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 查询（Q）的线性层
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # 键（K）的线性层
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 值（V）的线性层
        self.out_proj = nn.Linear(d_out, d_out)  # 结合头输出的线性层
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))  # 注册一个上三角掩码缓冲区

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # 获取输入的形状

        keys = self.W_key(x)  # 形状：(batch_size, num_tokens, d_out)
        queries = self.W_query(x)  # 形状：(batch_size, num_tokens, d_out)
        values = self.W_value(x)  # 形状：(batch_size, num_tokens, d_out)

        # 通过添加一个`num_heads`维度隐式地分割矩阵
        # 展开最后一个维度：(batch_size, num_tokens, d_out) -> (batch_size, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置：(batch_size, num_tokens, num_heads, head_dim) -> (batch_size, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算缩放点积注意力（即自注意力）并使用因果掩码
        attn_scores = queries @ keys.transpose(2, 3)  # 每个头的点积

        # 将原始掩码截断到token数量，并转换为布尔值
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码填充注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 形状：(batch_size, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并头，其中 self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 可选的投影

        return context_vec
#####################################
# Chapter 4
#####################################

# 定义层归一化（Layer Normalization）类，用于归一化神经网络层的输出
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 用于防止除以0的极小值
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 归一化尺度参数，可学习的参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 归一化偏移参数，可学习的参数

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # 计算输入张量在最后一个维度上的均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算输入张量在最后一个维度上的方差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 进行归一化处理
        return self.scale * norm_x + self.shift  # 应用尺度和偏移参数

# 定义GELU（Gaussian Error Linear Unit）激活函数类
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 计算GELU激活函数的值，GELU是深度学习中的一种激活函数
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *  # 计算GELU函数中的一个常数因子
            (x + 0.044715 * torch.pow(x, 3))  # 计算GELU函数中的tanh部分
        ))

# 定义前馈网络（Feed Forward Network）类，用于Transformer模型中的前馈网络结构
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 第一个线性层，将输入扩大四倍
            GELU(),  # GELU激活函数
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 第二个线性层，将输入恢复到原来的维度
        )

    def forward(self, x):
        return self.layers(x)  # 通过前馈网络

# 定义Transformer块类，用于构建Transformer模型的基本单元
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(  # 多头自注意力机制
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)  # 前馈网络
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 第一个层归一化
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 第二个层归一化
        self.drop_resid = nn.Dropout(cfg["drop_rate"])  # 残差连接的dropout

    def forward(self, x):
        # 注意力块的快捷连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # 通过自注意力机制
        x = self.drop_resid(x)
        x = x + shortcut  # 残差连接

        # 前馈网络块的快捷连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  # 残差连接

        return x

# 定义GPT模型类，用于构建基于Transformer的GPT模型
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 词嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入层
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # 嵌入层的dropout

        self.trf_blocks = nn.Sequential(  # Transformer块序列
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])  # 最终层归一化
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 输出层

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape  # 获取输入张量的批次大小和序列长度
        tok_embeds = self.tok_emb(in_idx)  # 词嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 位置嵌入
        x = tok_embeds + pos_embeds  # 将词嵌入和位置嵌入相加
        x = self.drop_emb(x)  # 应用dropout
        x = self.trf_blocks(x)  # 通过Transformer块序列
        x = self.final_norm(x)  # 应用最终层归一化
        logits = self.out_head(x)  # 通过输出层
        return logits

# 定义一个简单的文本生成函数，用于生成新文本
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx是当前上下文的索引数组
    for _ in range(max_new_tokens):
        # 如果当前上下文超过支持的上下文大小，则裁剪
        idx_cond = idx[:, -context_size:]

        # 获取预测结果
        with torch.no_grad():
            logits = model(idx_cond)

        # 只关注最后一个时间步
        logits = logits[:, -1, :]

        # 获取具有最高logits值的词汇表索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 将采样的索引追加到正在运行的序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx
#####################################
# Chapter 5
#####################################

# 定义一个文本生成函数，该函数使用给定的模型、上下文索引、最大新令牌数、上下文大小、温度参数、最高k值采样和结束序列标记
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # 与之前相同：获取logits，并只关注最后一个时间步
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # 裁剪当前上下文以适应支持的上下文大小
        with torch.no_grad():  # 不计算梯度，提高效率
            logits = model(idx_cond)  # 获取模型的预测logits
        logits = logits[:, -1, :]  # 只关注最后一个时间步的logits

        # 根据top_k采样过滤logits
        if top_k is not None:
            # 保留top_k个值
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # 应用温度缩放
        if temperature > 0.0:
            logits = logits / temperature

            # 应用softmax获取概率分布
            probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)

            # 从概率分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # 否则，与之前相同：获取具有最高logits值的词汇表索引
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # 如果遇到指定的结束序列标记eos_id，则提前停止生成
        if idx_next == eos_id:
            break

        # 与之前相同：将采样的索引追加到正在运行的序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx  # 返回生成的索引序列


# 定义一个简单的模型训练函数，用于训练模型并跟踪损失和看到的标记数
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # 初始化列表以跟踪损失和看到的标记数
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # 主训练循环
    for epoch in range(num_epochs):  # 遍历每个epoch
        model.train()  # 将模型设置为训练模式

        for input_batch, target_batch in train_loader:  # 遍历训练数据加载器中的批次
            optimizer.zero_grad()  # 清零梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 根据梯度更新模型权重
            tokens_seen += input_batch.numel()  # 更新看到的标记数
            global_step += 1  # 更新全局步数

            # 可选的评估步骤
            if global_step % eval_freq == 0:  # 如果达到评估频率
                train_loss, val_loss = evaluate_model(  # 评估模型
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)  # 记录训练损失
                val_losses.append(val_loss)  # 记录验证损失
                track_tokens_seen.append(tokens_seen)  # 记录看到的标记数
                print(f"Ep {epoch+1} (Step {global_step:06d}): "  # 打印当前epoch和步数
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 每个epoch后打印一个样本文本
        generate_and_print_sample(  # 生成并打印样本文本
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen  # 返回损失和看到的标记数


# 定义一个评估模型性能的函数，该函数计算训练和验证损失
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 不计算梯度，提高效率
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)  # 计算训练损失
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)  # 计算验证损失
    model.train()  # 将模型设置回训练模式
    return train_loss, val_loss  # 返回训练和验证损失


# 定义一个生成并打印样本文本的函数
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()  # 将模型设置为评估模式
    context_size = model.pos_emb.weight.shape[0]  # 获取位置嵌入的上下文大小
    encoded = text_to_token_ids(start_context, tokenizer).to(device)  # 将起始文本转换为标记ID并移动到设备
    with torch.no_grad():  # 不计算梯度，提高效率
        token_ids = generate_text_simple(  # 生成文本
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)  # 将标记ID转换回文本
        print(decoded_text.replace("\n", " "))  # 打印文本，替换换行符为空格
    model.train()  # 将模型设置回训练模式
# 定义一个函数，用于将两个张量赋值，如果形状不匹配则抛出错误
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"形状不匹配。左边：{left.shape}，右边：{right.shape}")
    return torch.nn.Parameter(torch.tensor(right))  # 返回一个可训练的张量参数

# 定义一个函数，用于将预训练的权重加载到GPT模型中
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])  # 位置嵌入权重赋值
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])  # 词嵌入权重赋值

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)  # 分割自注意力层的权重
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)  # 查询（Q）权重赋值
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)  # 键（K）权重赋值
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)  # 值（V）权重赋值

        q_b, k_b, v_b = np.split(params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)  # 分割自注意力层的偏置
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)  # 查询（Q）偏置赋值
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)  # 键（K）偏置赋值
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)  # 值（V）偏置赋值

        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)  # 自注意力输出层权重赋值
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])  # 自注意力输出层偏置赋值

        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)  # 前馈网络第一层权重赋值
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])  # 前馈网络第一层偏置赋值
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)  # 前馈网络输出层权重赋值
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])  # 前馈网络输出层偏置赋值

        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])  # 第一层层归一化尺度赋值
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])  # 第一层层归一化偏移赋值
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])  # 第二层层归一化尺度赋值
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])  # 第二层层归一化偏移赋值

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])  # 最终层归一化尺度赋值
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])  # 最终层归一化偏移赋值
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])  # 输出层权重赋值

# 定义一个函数，将文本转换为标记ID
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})  # 使用tokenizer进行编码
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加批次维度
    return encoded_tensor

# 定义一个函数，将标记ID转换回文本
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # 移除批次维度
    return tokenizer.decode(flat.tolist())  # 将ID列表解码为文本

# 定义一个函数，用于计算一批数据的损失
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将数据移动到指定设备
    logits = model(input_batch)  # 获取模型的预测logits
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())  # 计算交叉熵损失
    return loss

# 定义一个函数，用于计算数据加载器的损失
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:  # 如果数据加载器为空
        return float("nan")  # 返回NaN
    elif num_batches is None:  # 如果没有指定批次数量
        num_batches = len(data_loader)  # 使用数据加载器中的所有批次
    else:
        num_batches = min(num_batches, len(data_loader))  # 如果指定的批次数量超过了数据加载器中的批次数量，则取较小值
    for i, (input_batch, target_batch) in enumerate(data_loader):  # 遍历数据加载器
        if i < num_batches:  # 如果在指定的批次数量内
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算损失
            total_loss += loss.item()  # 累加损失
        else:
            break
    return total_loss / num_batches  # 返回平均损失

# 定义一个函数，用于绘制训练和验证损失
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))  # 创建一个图形和y轴

    # 绘制训练和验证损失随epoch的变化
    ax1.plot(epochs_seen, train_losses, label="训练损失")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="验证损失")
    ax1.set_xlabel("Epochs")  # 设置x轴标签
    ax1.set_ylabel("Loss")  # 设置y轴标签
    ax1.legend(loc="upper right")  # 设置图例位置
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # x轴只显示整数标签

    # 创建第二个x轴，用于显示看到的标记数
    ax2 = ax1.twiny()  # 创建第二个x轴，共享y轴
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 绘制一个不可见的图，用于对齐刻度
    ax2.set_xlabel("Tokens seen")  # 设置第二个x轴标签

    fig.tight_layout()  # 调整布局
    plt.savefig("loss-plot.pdf")  # 保存图形
    plt.show()  # 显示图形