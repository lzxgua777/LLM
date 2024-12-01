from importlib.metadata import version  # 导入version函数，用于获取库的版本信息

pkgs = [
    "tiktoken",    # 分词器库
    "torch",       # 深度学习库
]
for p in pkgs:
    print(f"{p} version: {version(p)}")  # 打印每个库的版本信息

import json  # 导入json库，用于处理JSON数据

# 定义文件路径
file_path = "instruction-data-with-preference.json"

# 读取JSON文件
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# 打印数据集的条目数量
print("Number of entries:", len(data))

import pprint  # 导入pprint库，用于美观地打印数据

# 打印数据集中的第51个和第1000个条目
pprint.pp(data[50])
pprint.pp(data[999])

# 定义格式化输入的函数
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

# 格式化第51个条目的输入
model_input = format_input(data[50])
print(model_input)

# 打印第51个条目的期望响应和可能响应
desired_response = f"### Response:\n{data[50]['chosen']}"
print(desired_response)

possible_response = f"### Response:\n{data[50]['rejected']}"
print(possible_response)

# 划分数据集为训练集、测试集和验证集
train_portion = int(len(data) * 0.85)  # 85%用于训练
test_portion = int(len(data) * 0.1)    # 10%用于测试
val_portion = len(data) - train_portion - test_portion  # 剩余5%用于验证

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

# 打印各数据集的长度
print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))

import torch  # 导入PyTorch库
from torch.utils.data import Dataset  # 导入PyTorch的数据集模块

# 定义PreferenceDataset类
class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # 预标记化文本
        self.encoded_texts = []
        for entry in data:
            prompt = format_input(entry)
            rejected_response = entry["rejected"]
            chosen_response = entry["chosen"]

            prompt_tokens = tokenizer.encode(prompt)
            chosen_full_text = f"{prompt}\n\n### Response:\n{chosen_response}"
            rejected_full_text = f"{prompt}\n\n### Response:\n{rejected_response}"
            chosen_full_tokens = tokenizer.encode(chosen_full_text)
            rejected_full_tokens = tokenizer.encode(rejected_full_text)

            self.encoded_texts.append({
                "prompt": prompt_tokens,
                "chosen": chosen_full_tokens,
                "rejected": rejected_full_tokens,
            })

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

# 定义自定义的批处理函数
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    allowed_max_length=None,
    mask_prompt_tokens=True,
    device="cpu"
):
    # 初始化列表以保存批次数据
    batch_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "rejected_mask": [],
        "chosen_mask": []

    }

    # 确定最长序列以设置公共填充长度
    max_length_common = 0
    if batch:
        for key in ["chosen", "rejected"]:
            current_max = max(len(item[key])+1 for item in batch)
            max_length_common = max(max_length_common, current_max)

    # 处理批次中的每个条目
    for item in batch:
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)

        for key in ["chosen", "rejected"]:
            # 根据公共最大长度调整填充
            sequence = item[key]
            padded = sequence + [pad_token_id] * (max_length_common - len(sequence))
            mask = torch.ones(len(padded)).bool()

            # 将所有填充标记的掩码设置为False
            mask[len(sequence):] = False

            # 将所有输入标记的掩码设置为False
            # +2将"### Response"前的两个换行符设置为False
            if mask_prompt_tokens:
                mask[:prompt.shape[0]+2] = False

            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(mask)

    # 最终处理
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        # 将所有序列堆叠成一个张量
        tensor_stack = torch.stack(batch_data[key])

        # 可选地截断到最大序列长度
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]

        # 移动到指定设备
        batch_data[key] = tensor_stack.to(device)

    return batch_data

from functools import partial  # 导入partial函数，用于部分应用函数

# 设置设备，如果GPU可用则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 部分应用自定义批处理函数
customized_collate_fn = partial(
    custom_collate_fn,
    device=device,            # 直接将数据放在GPU上（如果可用）
    mask_prompt_tokens=True,  # 这是可选的
    allowed_max_length=1024   # 模型支持的上下文长度
)

# 获取数据集的前两个条目作为示例
example_data = data[:2]

# 打印示例数据
for i in example_data:
    print()
    pprint.pp(i)
import tiktoken  # 导入tiktoken库，用于分词
from torch.utils.data import DataLoader  # 导入PyTorch的数据加载器模块

# 初始化GPT-2分词器
tokenizer = tiktoken.get_encoding("gpt2")

# 创建示例数据集
example_dataset = PreferenceDataset(example_data, tokenizer)

# 创建示例数据加载器
example_dataloader = DataLoader(
    example_dataset,
    batch_size=2,
    collate_fn=customized_collate_fn,
    shuffle=False
)

# 从数据加载器中获取一个批次并中断循环
for batch in example_dataloader:
    break

# 打印批次中的键
print("batch.keys:", batch.keys())

# 访问批次中的“prompt”和“chosen”数据
batch["prompt"]
batch["chosen"]

# 定义一个函数，用于从批次中解码标记
def decode_tokens_from_batch(token_ids, tokenizer):
    ids_in_python_list = token_ids.flatten().tolist()
    return tokenizer.decode(ids_in_python_list)

# 解码并打印“prompt”数据
text = decode_tokens_from_batch(
    token_ids=batch["prompt"][0],  # 使用批次中的第一个条目
    tokenizer=tokenizer,
)
print(text)

# 解码并打印“chosen”数据
text = decode_tokens_from_batch(
    token_ids=batch["chosen"][0],
    tokenizer=tokenizer,
)
print(text)

# 解码并打印“rejected”数据
text = decode_tokens_from_batch(
    token_ids=batch["rejected"][0],
    tokenizer=tokenizer,
)
print(text)

# 打印“chosen”输入和掩码的形状
print("chosen inputs:", batch["chosen"][0].shape)
print("chosen mask:  ", batch["chosen_mask"][0].shape)

# 访问“chosen”掩码
batch["chosen_mask"][0]

# 解码并打印应用“chosen”掩码后的数据
text = decode_tokens_from_batch(
    token_ids=batch["chosen"][0][batch["chosen_mask"][0]],
    tokenizer=tokenizer,
)
print(text)

# 解码并打印应用“rejected”掩码后的数据
text = decode_tokens_from_batch(
    token_ids=batch["rejected"][0][batch["rejected_mask"][0]],
    tokenizer=tokenizer,
)
print(text)

from torch.utils.data import DataLoader  # 再次导入DataLoader

# 设置工作线程数和批次大小
num_workers = 0
batch_size = 8

# 设置随机种子以确保结果可复现
torch.manual_seed(123)

# 创建训练、验证和测试数据集
train_dataset = PreferenceDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = PreferenceDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = PreferenceDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

# 打印训练加载器的形状信息
print("Train loader:")
for batch in train_loader:
    print(
        batch["chosen"].shape,
        batch["rejected"].shape,
    )

# 导入os和pathlib模块，用于文件操作
import os
from pathlib import Path
import shutil  # 导入shutil模块，用于文件复制

# 定义微调模型的路径
finetuned_model_path = Path("gpt2-medium355M-sft.pth")
if not finetuned_model_path.exists():

    # 尝试在本地找到模型检查点
    relative_path = Path("..") / "01_main-chapter-code" / finetuned_model_path
    if relative_path.exists():
        shutil.copy(relative_path, ".")

    # 如果这个笔记本在Google Colab上运行，从Google Drive文件夹中获取
    elif "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        from google.colab import drive
        drive.mount("/content/drive")
        google_drive_path = "/content/drive/My Drive/Books/LLMs-From-Scratch/ch07/colab/gpt2-medium355M-sft.pth"  # 读者需要调整这个路径
        shutil.copy(google_drive_path, ".")

    else:
        print(
            f"Could not find '{finetuned_model_path}'.\n"
            "Run the `ch07.ipynb` notebook to finetune and save the finetuned model."
        )

# 从本地文件导入GPTModel类
from ch04.main_chapter_code.ch04 import GPTModel

# 定义基础配置
BASE_CONFIG = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "drop_rate": 0.0,        # Dropout比率
    "qkv_bias": True         # Query-key-value偏置
}

# 定义不同模型的配置
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# 选择模型配置
CHOOSE_MODEL = "gpt2-medium (355M)"

# 更新基础配置
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# 实例化模型
model = GPTModel(BASE_CONFIG)
# 加载预训练的GPT-2模型状态
model.load_state_dict(
    torch.load(
        "gpt2-medium355M-sft.pth",
        map_location=torch.device("cpu"),
        weights_only=True
    )
)
model.eval();

# 定义一个指令性提示
prompt = """Below is an instruction that describes a task. Write a response
that appropriately completes the request.

### Instruction:
Convert the active sentence to passive: 'The chef cooks the meal every day.'
"""

# 从指定的章节导入generate、text_to_token_ids和token_ids_to_text函数
from ch05.main_chapter_code.ch05 import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

# 设置随机种子以确保结果可复现
torch.manual_seed(123)

# 使用模型生成响应
token_ids = generate(
    model=model,
    idx=text_to_token_ids(prompt, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256
)

# 将标记ID转换回文本
response = token_ids_to_text(token_ids, tokenizer)
print(response)

# 定义一个函数，用于从完整的响应文本中提取实际的响应内容
def extract_response(response_text, input_text):
    return response_text[len(input_text):].replace("### Response:", "").strip()

# 提取响应并打印
response = extract_response(response, prompt)
print(response)

# 实例化策略模型（即我们要优化的模型）
policy_model = model

# 实例化参考模型（即我们用来比较的模型）
reference_model = GPTModel(BASE_CONFIG)
reference_model.load_state_dict(
    torch.load(
        "gpt2-medium355M-sft.pth",
        map_location=torch.device("cpu"),
        weights_only=True
    )
)
reference_model.eval()

# 将策略模型和参考模型移动到指定的设备（CPU或GPU）
policy_model.to(device)
reference_model.to(device);

# 导入PyTorch的函数库
import torch.nn.functional as F

# 定义计算DPO（直接偏好优化）损失的函数
def compute_dpo_loss(
      model_chosen_logprobs,
      model_rejected_logprobs,
      reference_chosen_logprobs,
      reference_rejected_logprobs,
      beta=0.1,
    ):
    """
    计算一批策略和参考模型对数概率的DPO损失。

    参数：
        model_chosen_logprobs：策略模型对优选响应的对数概率。形状：（批量大小，）
        model_rejected_logprobs：策略模型对非优选响应的对数概率。形状：（批量大小，）
        reference_chosen_logprobs：参考模型对优选响应的对数概率。形状：（批量大小，）
        reference_rejected_logprobs：参考模型对非优选响应的对数概率。形状：（批量大小，）
        beta：DPO损失的温度参数；通常在0.1到0.5之间。当beta趋近于0时，我们忽略参考模型。
        label_smoothing：DPO损失的保守性。

    返回：
        一个包含三个张量的元组：（损失，优选奖励，非优选奖励）。
    """

    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_logratios - reference_logratios

    # DPO损失计算（参考论文中的公式7）
    losses = -F.logsigmoid(beta * logits)

    # 可选的值，用于在训练期间跟踪进度
    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    # .mean()用于对批量样本求平均
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

# 定义计算对数概率的函数
def compute_logprobs(logits, labels, selection_mask=None):
    """
    计算对数概率。

    参数：
      logits：形状为（批量大小，num_tokens，词汇表大小）的张量
      labels：形状为（批量大小，num_tokens）的张量
      selection_mask：形状为（批量大小，num_tokens）的张量

    返回：
      mean_log_prob：排除填充标记的平均对数概率。
    """

    # 标签是输入的位移版本
    labels = labels[:, 1:].clone()

    # 截断logits以匹配labels的num_tokens
    logits = logits[:, :-1, :]

    log_probs = F.log_softmax(logits, dim=-1)

    # 收集实际标签的对数概率
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    if selection_mask is not None:
        mask = selection_mask[:, 1:].clone()

        # 应用掩码以过滤填充标记
        selected_log_probs = selected_log_probs * mask

        # 计算排除填充标记的平均对数概率
        # 这在标记上取平均，因此形状是（批量大小，num_tokens）
        avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)

        return avg_log_prob

    else:
        return selected_log_probs.mean(-1)
# 示例数据
logits = torch.tensor(
    [[2.0, 1.0, 0.1],  # 第一行是3个类别的logits值
     [0.5, 2.5, 0.3]])  # 第二行是3个类别的logits值
targets = torch.tensor([0, 2])  # 目标类别索引，第一行为0，第二行为2

# 使用torch.gather手动计算损失
log_softmax_logits = F.log_softmax(logits, dim=1)  # 计算log-softmax值，沿着类别维度（dim=1）计算
selected_log_probs = torch.gather(
    input=log_softmax_logits,  # 从log_softmax_logits中按目标类别选择
    dim=1,  # 选择类别的维度
    index=targets.unsqueeze(1),  # 将目标类别索引的形状从(2,)调整为(2, 1)
).squeeze(1)  # 去掉额外的维度，恢复为(2,)
manual_loss = -selected_log_probs.mean()  # 计算批次的平均损失（取负对数概率并计算均值）

# 使用PyTorch的交叉熵损失函数计算损失
cross_entropy_loss = F.cross_entropy(logits, targets)  # 使用PyTorch内置的交叉熵损失函数计算损失

# 打印手动计算的损失和PyTorch的损失进行比较
print(manual_loss, cross_entropy_loss)  # 输出两种损失的值进行比较

# 示例：使用torch.gather根据索引从张量中抽取元素
t = torch.tensor(
  [[1., 2.],  # 示例张量，包含2行2列
   [3., 4.]])  # 第二行的数据
m = torch.tensor(
  [[1, 1],  # 索引张量，选择从t中对应位置的元素
   [0, 1]])  # 每个位置的值表示从t中要选择的列
torch.gather(input=t, dim=-1, index=m)  # 根据m的索引从t中抽取元素，得到对应的元素

# 定义计算DPO损失的函数
def compute_dpo_loss_batch(batch, policy_model, reference_model, beta):
    """计算输入批次的DPO损失"""

    # 计算策略模型对优选响应的对数概率
    policy_chosen_log_probas = compute_logprobs(
        logits=policy_model(batch["chosen"]),  # 通过策略模型计算优选响应的logits
        labels=batch["chosen"],  # 优选响应的标签
        selection_mask=batch["chosen_mask"]  # 选择掩码，指示哪些响应被选择
    )
    # 计算策略模型对非优选响应的对数概率
    policy_rejected_log_probas = compute_logprobs(
        logits=policy_model(batch["rejected"]),  # 通过策略模型计算非优选响应的logits
        labels=batch["rejected"],  # 非优选响应的标签
        selection_mask=batch["rejected_mask"]  # 非优选响应的选择掩码
    )
    # 计算参考模型对优选响应的对数概率
    ref_chosen_log_probas = compute_logprobs(
        logits=reference_model(batch["chosen"]),  # 通过参考模型计算优选响应的logits
        labels=batch["chosen"],  # 优选响应的标签
        selection_mask=batch["chosen_mask"]  # 选择掩码，指示哪些响应被选择
    )
    # 计算参考模型对非优选响应的对数概率
    ref_rejected_log_probas = compute_logprobs(
        logits=reference_model(batch["rejected"]),  # 通过参考模型计算非优选响应的logits
        labels=batch["rejected"],  # 非优选响应的标签
        selection_mask=batch["rejected_mask"]  # 非优选响应的选择掩码
    )
    # 计算DPO损失
    loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_log_probas,  # 策略模型对优选响应的log概率
        model_rejected_logprobs=policy_rejected_log_probas,  # 策略模型对非优选响应的log概率
        reference_chosen_logprobs=ref_chosen_log_probas,  # 参考模型对优选响应的log概率
        reference_rejected_logprobs=ref_rejected_log_probas,  # 参考模型对非优选响应的log概率
        beta=beta  # Beta参数，用于权衡奖励
    )
    return loss, chosen_rewards, rejected_rewards  # 返回损失和奖励

# 不计算梯度的情况下计算DPO损失
with torch.no_grad():  # 在不计算梯度的上下文中进行操作
    loss = compute_dpo_loss_batch(batch, policy_model, reference_model, beta=0.1)  # 计算DPO损失
print(loss)  # 输出DPO损失

# 定义计算整个数据加载器DPO损失的函数
def compute_dpo_loss_loader(data_loader, policy_model, reference_model, beta, num_batches=None):
    """对整个数据加载器应用compute_dpo_loss_batch"""

    total_loss, total_chosen_rewards, total_rejected_rewards = 0., 0., 0.  # 初始化总损失和奖励
    if len(data_loader) == 0:  # 如果数据加载器为空，返回NaN
        return float("nan")

    elif num_batches is None:  # 如果没有指定批次数，则使用数据加载器中的所有批次
        num_batches = len(data_loader)
    else:
        # 如果num_batches超过数据加载器中的批次总数，则取两者中的较小值
        num_batches = min(num_batches, len(data_loader))
    for i, batch in enumerate(data_loader):  # 遍历数据加载器中的批次
        if i < num_batches:
            # 计算每个批次的DPO损失
            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )
            # 累加损失和奖励
            total_loss += loss.item()
            total_chosen_rewards += chosen_rewards.item()
            total_rejected_rewards += rejected_rewards.item()

        else:
            break  # 如果超过了批次数，则停止

    # 计算平均损失
    total_loss /= num_batches
    total_chosen_rewards /= num_batches
    total_rejected_rewards /= num_batches
    return total_loss, total_chosen_rewards, total_rejected_rewards  # 返回总损失和奖励

# 定义评估DPO损失的函数
def evaluate_dpo_loss_loader(policy_model, reference_model, train_loader, val_loader, beta, eval_iter):
    """计算训练和验证数据集的DPO损失"""

    policy_model.eval()  # 将策略模型设置为评估模式
    with torch.no_grad():  # 不计算梯度
        # 计算训练集的DPO损失
        train_loss, train_chosen_rewards, train_rejected_rewards = compute_dpo_loss_loader(
            data_loader=train_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter  # 使用指定数量的批次
        )

        # 计算验证集的DPO损失
        val_loss, val_chosen_rewards, val_rejected_rewards = compute_dpo_loss_loader(
            data_loader=val_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter  # 使用指定数量的批次
        )

    # 返回评估结果
    res = {
        "train_loss": train_loss,
        "train_chosen_reward": train_chosen_rewards,
        "train_rejected_reward": train_rejected_rewards,
        "val_loss": val_loss,
        "val_chosen_reward": val_chosen_rewards,
        "val_rejected_reward": val_rejected_rewards
    }

    policy_model.train()  # 将策略模型设置回训练模式
    return res  # 返回评估结果


from ch07.main_chapter_code.previous_chapters import generate_and_print_sample


def train_model_dpo_simple(
    policy_model, reference_model, train_loader, val_loader,
    optimizer, num_epochs, beta,
    eval_freq, eval_iter, start_context, tokenizer
):

    # Initialize lists to track losses and tokens seen
    tracking = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "tokens_seen": []
    }
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        policy_model.train()  # Set model to training mode

        for batch_idx, batch in enumerate(train_loader):

            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration

            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )

            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients

            tokens_seen += batch["chosen"].numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                res = evaluate_dpo_loss_loader(
                    policy_model=policy_model,
                    reference_model=reference_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    beta=beta,
                    eval_iter=eval_iter
                )
                tracking["train_losses"].append(res["train_loss"])
                tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                tracking["val_losses"].append(res["val_loss"])
                tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                tracking["val_rejected_rewards"].append(res["val_rejected_reward"])
                tracking["tokens_seen"].append(tokens_seen)
                train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                val_reward_margin = res["val_chosen_reward"] - res["val_rejected_reward"]

                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                    f"Train reward margins {train_reward_margin:.3f}, "
                    f"Val reward margins {val_reward_margin:.3f}"
                )

        # Print a sample text after each epoch
        generate_and_print_sample(
            model=model,
            tokenizer=tokenizer,
            device=loss.device,
            start_context=start_context
        )

    return tracking

torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

res = evaluate_dpo_loss_loader(
    policy_model=policy_model,
    reference_model=reference_model,
    train_loader=train_loader,
    val_loader=val_loader,
    beta=0.1,
    eval_iter=5
)

print("Training loss:", res["train_loss"])
print("Validation loss:", res["val_loss"])

print("Train reward margin:", res["train_chosen_reward"] - res["train_rejected_reward"])
print("Val reward margin:", res["val_chosen_reward"] - res["val_rejected_reward"])

torch.manual_seed(123)


for entry in val_data[:3]:

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
)

    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("\n-------------------------------------\n")

import time

start_time = time.time()

torch.manual_seed(123)


optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6, weight_decay=0.01)

num_epochs = 1
tracking = train_model_dpo_simple(
    policy_model=policy_model,
    reference_model=reference_model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    num_epochs=num_epochs,
    beta=0.1, # value between 0.1 and 0.5
    eval_freq=5,
    eval_iter=5,
    start_context=format_input(val_data[2]),
    tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

from ch07.main_chapter_code.ch07 import plot_losses


epochs_tensor = torch.linspace(0, num_epochs, len(tracking["train_losses"]))
plot_losses(
    epochs_seen=epochs_tensor,
    tokens_seen=tracking["tokens_seen"],
    train_losses=tracking["train_losses"],
    val_losses=tracking["val_losses"],
    label="loss"
)

train_reward_margins = [i-j for i,j in zip(tracking["train_chosen_rewards"], tracking["train_rejected_rewards"])]
val_reward_margins = [i-j for i,j in zip(tracking["val_chosen_rewards"], tracking["val_rejected_rewards"])]

plot_losses(
    epochs_seen=epochs_tensor,
    tokens_seen=tracking["tokens_seen"],
    train_losses=train_reward_margins,
    val_losses=val_reward_margins,
    label="loss"
)

torch.manual_seed(123)


for entry in val_data[:3]:

    input_text = format_input(entry)

    token_ids = generate(
        model=reference_model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    reference_response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )

    token_ids = generate(
        model=policy_model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    policy_response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )

    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nReference model response:\n>> {reference_response_text.strip()}")
    print(f"\nPolicy model response:\n>> {policy_response_text.strip()}")
    print("\n-------------------------------------\n")

torch.manual_seed(123)


for entry in test_data[:3]:

    input_text = format_input(entry)

    token_ids = generate(
        model=reference_model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    reference_response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )

    token_ids = generate(
        model=policy_model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    policy_response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )

    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nReference model response:\n>> {reference_response_text.strip()}")
    print(f"\nPolicy model response:\n>> {policy_response_text.strip()}")
    print("\n-------------------------------------\n")

