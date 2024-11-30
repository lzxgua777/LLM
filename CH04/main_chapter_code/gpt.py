# 此文件收集了我们到目前为止介绍的所有相关代码
# 贯穿第 2-4 章。
# 此文件可以作为独立脚本运行。
import tiktoken  # 导入 tiktoken 库用于分词
import torch  # 导入 PyTorch，用于构建神经网络和张量操作
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，用于定义层和模型
from torch.utils.data import Dataset, DataLoader  # 导入 Dataset 和 DataLoader，用于处理训练数据


#####################################
# 第2章
#####################################

class GPTDatasetV1(Dataset):
    # 自定义数据集类，用于准备 GPT 模型的文本数据
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []  # 用于存储标记化后的输入序列
        self.target_ids = []  # 用于存储标记化后的目标序列（用于监督学习）

        # 使用提供的 tokenizer 对整个文本进行标记化
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口对标记化后的文本进行切分，创建重叠的最大长度序列
        # 我们使用 `stride` 作为滑动步长
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 当前窗口的输入序列，大小为 `max_length`
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 当前窗口的目标序列（偏移 1）
            self.input_ids.append(torch.tensor(input_chunk))  # 将输入序列转换为张量并加入列表
            self.target_ids.append(torch.tensor(target_chunk))  # 将目标序列转换为张量并加入列表

    def __len__(self):
        # 返回数据集中的样本数量，即生成的序列对的数量
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 根据索引返回对应的输入和目标张量，用于 DataLoader 中批处理
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 创建 DataLoader 函数，负责批处理、打乱数据等
    tokenizer = tiktoken.get_encoding("gpt2")  # 使用 tiktoken 初始化 GPT-2 分词器

    # 使用提供的文本、分词器和切分参数创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建 DataLoader 用于加载数据
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# 第3章
#####################################

class MultiHeadAttention(nn.Module):
    # 实现多头自注意力机制，是 Transformer 模型的核心组件之一
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out 必须能被 num_heads 整除"  # 确保输出维度能被头数整除

        self.d_out = d_out  # 每个注意力头的输出维度
        self.num_heads = num_heads  # 注意力头的数量
        self.head_dim = d_out // num_heads  # 每个头的维度（输出维度除以头数）

        # 将输入映射到查询、键、值空间的线性层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 输出投影线性层，用于合并头的输出
        self.out_proj = nn.Linear(d_out, d_out)

        # Dropout 层用于正则化
        self.dropout = nn.Dropout(dropout)

        # 用于自注意力的因果（上三角）掩码，确保每个 token 只能关注到前面的 tokens
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # 多头注意力机制的前向传播

        b, num_tokens, d_in = x.shape  # b: 批大小，num_tokens: 序列长度，d_in: 每个 token 的输入维度

        # 通过线性层获得查询（queries）、键（keys）和值（values）
        keys = self.W_key(x)  # 形状: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 将查询、键、值的形状调整为多头注意力需要的形状
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置维度，使得 `num_heads` 维度在批次维度后：
        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算缩放点积注意力（即自注意力）
        attn_scores = queries @ keys.transpose(2, 3)  # 点积，形状为 (b, num_heads, num_tokens, num_tokens)

        # 使用掩码防止模型关注到未来的 tokens（因果掩码）
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # 根据当前序列长度截取掩码
        attn_scores.masked_fill_(mask_bool, -torch.inf)  # 将未来的位置填充为负无穷大，阻止其参与计算

        # 对注意力得分应用 softmax，得到注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)  # 用键的维度开方进行归一化
        attn_weights = self.dropout(attn_weights)  # 对注意力权重应用 dropout

        # 通过将注意力权重与值相乘，计算得到上下文向量
        context_vec = (attn_weights @ values).transpose(1,
                                                        2)  # (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)

        # 将各个头的输出合并
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # (b, num_tokens, d_out)

        # 可选的线性投影，将输出调整为期望的维度
        context_vec = self.out_proj(context_vec)

        return context_vec  # 返回从多头注意力计算得到的上下文向量

####################################
# 第4章
####################################
# 定义LayerNorm类，实现层归一化
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 用于防止除以零的一个小值
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 学习缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 学习偏移参数

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # 计算均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算方差，不使用无偏估计
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 归一化
        return self.scale * norm_x + self.shift  # 应用缩放和偏移

# 定义GELU类，实现GELU激活函数
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(  # GELU激活函数的计算公式
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *  # 计算平方根
            (x + 0.044715 * torch.pow(x, 3))  # 计算三次方并乘以系数
        ))

# 定义FeedForward类，实现前馈网络
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(  # 定义一个顺序容器，包含两个线性层和一个GELU激活函数
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 第一个线性层，将嵌入维度扩展4倍
            GELU(),  # GELU激活函数
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])  # 第二个线性层，将维度还原回原始嵌入维度
        )

    def forward(self, x):
        return self.layers(x)  # 前向传播函数，将输入x通过定义好的层

# 定义TransformerBlock类，实现Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(  # 初始化多头自注意力模块
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)  # 初始化前馈网络模块
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 初始化第一个层归一化模块
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 初始化第二个层归一化模块
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # 初始化快捷连接的dropout模块

    def forward(self, x):
        # 注意力块的快捷连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # 形状 [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # 将原始输入加回来

        # 前馈网络块的快捷连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 将原始输入加回来

        return x

# 定义GPTModel类，实现GPT模型
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 定义token嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 定义位置嵌入层
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # 定义dropout层

        self.trf_blocks = nn.Sequential(  # 使用TransformerBlock构建Transformer层的序列
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])  # 定义最终的层归一化
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 定义输出层

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape  # 提取输入张量的形状
        tok_embeds = self.tok_emb(in_idx)  # 计算token嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 计算位置嵌入
        x = tok_embeds + pos_embeds  # 合并token嵌入和位置嵌入
        x = self.drop_emb(x)  # 应用dropout
        x = self.trf_blocks(x)  # 通过Transformer层
        x = self.final_norm(x)  # 应用最终的层归一化
        logits = self.out_head(x)  # 计算logits
        return logits

# 定义一个函数，用于简单生成文本
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx是当前上下文的索引数组
    for _ in range(max_new_tokens):
        # 如果当前上下文超过支持的上下文大小，则裁剪
        idx_cond = idx[:, -context_size:]

        # 不计算梯度，获取预测
        with torch.no_grad():
            logits = model(idx_cond)

        # 只关注最后一个时间步
        logits = logits[:, -1, :]

        # 获取概率最高的词汇表索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 将采样的索引追加到运行序列
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

# 定义主函数
def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # 词汇表大小
        "context_length": 1024,  # 上下文长度
        "emb_dim": 768,          # 嵌入维度
        "n_heads": 12,           # 注意力头数
        "n_layers": 12,          # 层数
        "drop_rate": 0.1,        # dropout率
        "qkv_bias": False        # QKV偏置
    }

    torch.manual_seed(123)  # 设置随机种子
    model = GPTModel(GPT_CONFIG_124M)  # 创建GPTModel实例
    model.eval()  # 设置模型为评估模式，禁用dropout

    start_context = "Hello, I am"  # 定义起始上下文

    tokenizer = tiktoken.get_encoding("gpt2")  # 初始化tokenizer
    encoded = tokenizer.encode(start_context)  # 编码起始上下文
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 将编码后的文本转换为张量

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")  # 打印输入部分的分隔线
    print("\nInput text:", start_context)  # 打印输入文本
    print("Encoded input text:", encoded)  # 打印编码后的输入文本
    print("encoded_tensor.shape:", encoded_tensor.shape)  # 打印编码张量的形状

    out = generate_text_simple(  # 调用生成文本的函数
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())  # 解码输出文本

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")  # 打印输出部分的分隔线
    print("\nOutput:", out)  # 打印输出
    print("Output length:", len(out[0]))  # 打印输出长度
    print("Output text:", decoded_text)  # 打印解码后的输出文本

# 如果当前脚本为主程序，则运行main函数
if __name__ == "__main__":
    main()