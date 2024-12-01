# 版权所有 （c） Sebastian Raschka，Apache 许可证 2.0（参见 LICENSE.txt）。
# “从头开始构建大型语言模型” 的源代码
# - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch
#
# 此文件收集了我们到目前为止介绍的所有相关代码
# 贯穿第 2-4 章。
# 此文件可以作为独立脚本运行。
import tiktoken  # 导入tiktoken库，用于处理GPT模型的token。
import torch  # 导入PyTorch库，用于深度学习。
import torch.nn as nn  # 导入PyTorch的神经网络模块。
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch的数据集和数据加载器模块。

#####################################
# 第2章
#####################################

class GPTDatasetV1(Dataset):
    # 定义一个数据集类，用于生成GPT模型的训练数据。
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []  # 存储输入序列的token ids。
        self.target_ids = []  # 存储目标序列的token ids。

        # 将整个文本进行token化，得到token ids。
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分割成重叠的序列，每个序列长度为max_length。
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 输入序列。
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标序列。
            self.input_ids.append(torch.tensor(input_chunk))  # 将输入序列转换为PyTorch张量并添加到列表。
            self.target_ids.append(torch.tensor(target_chunk))  # 将目标序列转换为PyTorch张量并添加到列表。

    def __len__(self):
        # 返回数据集中样本的数量。
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 根据索引idx获取一个样本，返回输入序列和目标序列的token ids。
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 定义一个函数，用于创建数据加载器。
    # 初始化tokenizer。
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建数据集实例。
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器，用于批量加载数据，并设置是否打乱数据顺序。
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader  # 返回数据加载器。

#####################################
# 第3章
#####################################
class MultiHeadAttention(nn.Module):
    # 定义一个多头注意力类，继承自PyTorch的nn.Module。
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out必须能被n_heads整除"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 计算每个头的维度。

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 查询的线性变换。
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # 键的线性变换。
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 值的线性变换。
        self.out_proj = nn.Linear(d_out, d_out)  # 用于合并头部输出的线性层。
        self.dropout = nn.Dropout(dropout)  # dropout层。
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))  # 注册一个上三角掩码。

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # 计算键。
        queries = self.W_query(x)  # 计算查询。
        values = self.W_value(x)  # 计算值。

        # 将矩阵按head维度分割，并展开最后一个维度。
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置以适应多头注意力计算。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算缩放点积注意力（即自注意力）与因果掩码。
        attn_scores = queries @ keys.transpose(2, 3)  # 对每个头进行点积。

        # 将掩码截断到token数量并转换为布尔值。
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码填充注意力分数。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量。
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并头部。
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 可选投影。

        return context_vec  # 返回上下文向量。
#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 设置一个小的epsilon值，用于防止除零错误
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 学习可训练的缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 学习可训练的偏移参数

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # 计算输入x的均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算输入x的方差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 对x进行标准化（归一化处理）
        return self.scale * norm_x + self.shift  # 通过缩放和偏移恢复输出
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))  # GELU激活函数


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 输入到隐藏层的线性变换
            GELU(),  # 激活函数
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 从隐藏层到输出层的线性变换
        )

    def forward(self, x):
        return self.layers(x)  # 前向传播
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention模块的快捷连接
        shortcut = x
        x = self.norm1(x)  # 对输入x进行归一化
        x = self.att(x)    # 通过多头注意力机制处理
        x = self.drop_shortcut(x)  # dropout
        x = x + shortcut  # 加上原始输入作为快捷连接

        # FeedForward模块的快捷连接
        shortcut = x
        x = self.norm2(x)  # 对输入x进行归一化
        x = self.ff(x)     # 通过前馈网络处理
        x = self.drop_shortcut(x)  # dropout
        x = x + shortcut  # 加上原始输入作为快捷连接

        return x
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 词嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入层
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # 嵌入层的dropout

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]  # 多个Transformer块
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])  # 最后的LayerNorm层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 输出层（生成预测）

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # 获取token嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 获取位置嵌入
        x = tok_embeds + pos_embeds  # 将词嵌入和位置嵌入相加
        x = self.drop_emb(x)  # 应用dropout
        x = self.trf_blocks(x)  # 通过多个Transformer块
        x = self.final_norm(x)  # 最后的LayerNorm
        logits = self.out_head(x)  # 输出层得到logits
        return logits
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx是当前上下文的索引（B, T）
    for _ in range(max_new_tokens):

        # 如果上下文超出支持的最大长度，则裁剪上下文
        idx_cond = idx[:, -context_size:]  # 截取最新的context_size个token

        # 获取预测结果
        with torch.no_grad():
            logits = model(idx_cond)

        # 只关注最后一个时间步的logits（预测下一个token）
        logits = logits[:, -1, :]

        # 获取logits中值最大的词的索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 将预测的token添加到当前序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx



#####################################
# Chapter 5
#####################################

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)  # 将文本编码为token id
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 增加批处理维度
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # 去掉批处理维度
    return tokenizer.decode(flat.tolist())  # 解码为文本

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # 与之前相同：获取logits，并只关注最后一个时间步
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # 新增：使用top_k采样
        if top_k is not None:
            # 保留logits中top_k个最大值
            top_logits, _ = torch.topk(logits, top_k)  # 从logits中选出top_k个最大值
            min_val = top_logits[:, -1]  # 找到top_k中的最小值
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)
            # 将logits中小于最小top_k值的项替换为负无穷，这样它们就不会被选中
        # 新增：应用温度缩放（temperature scaling）
        if temperature > 0.0:
            logits = logits / temperature  # 将logits除以温度值进行缩放

            # 应用softmax以得到概率分布
            probs = torch.softmax(logits, dim=-1)  # 对logits进行softmax转换为概率，返回一个概率分布

            # 从概率分布中进行采样
            idx_next = torch.multinomial(probs, num_samples=1)  # 使用多项式分布从概率中采样下一个token的索引


        # 否则，和之前一样：获取logits中最大值对应的词的索引
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # 直接选择logits中概率最大的词
        if idx_next == eos_id:  # 如果遇到eos_token，提前停止生成
            break

        # 和之前一样：将采样的token索引添加到当前序列中
        idx = torch.cat((idx, idx_next), dim=1)  # 将新的token索引连接到输入序列中

    return idx      #返回最终的token序列
