# 版权所有 （c） Sebastian Raschka，Apache 许可证 2.0（参见 LICENSE.txt）。
# “从头开始构建大型语言模型” 的源代码
# - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch
#
# 此文件收集了我们到目前为止介绍的所有相关代码
# 贯穿第 2-4 章。
# 此文件可以作为独立脚本运行。
import torch  # 导入PyTorch库

#####################################
# 第5章
#####################################

# 将文本转换为token ID的函数
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)  # 使用tokenizer将文本编码为token ID序列
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 将token ID序列转换为张量，并增加批处理维度（batch dimension）
    return encoded_tensor  # 返回处理后的token ID张量

# 将token ID转换回文本的函数
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # 删除批处理维度（将张量扁平化）
    return tokenizer.decode(flat.tolist())  # 将token ID序列解码回文本

# 生成函数，根据给定的上下文生成文本
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # 对生成的每个新token进行循环
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # 取出上下文的最后context_size个token作为条件
        with torch.no_grad():  # 在推理时不计算梯度
            logits = model(idx_cond)  # 获取模型的logits输出
        logits = logits[:, -1, :]  # 仅关注最后一个时间步的logits（最新生成的token）

        # 新增：使用top_k采样方法过滤logits
        if top_k is not None:
            # 保留logits中值最大的top_k个
            top_logits, _ = torch.topk(logits, top_k)  # 获取top_k个最大值及其索引
            min_val = top_logits[:, -1]  # 获取top_k中最小的那个值
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)  # 小于最小值的logits设为负无穷，过滤掉不需要的部分

        # 新增：应用温度缩放（Temperature Scaling）
        if temperature > 0.0:
            logits = logits / temperature  # 按照温度缩放logits

            # 使用softmax将logits转化为概率分布
            probs = torch.softmax(logits, dim=-1)  # 计算概率分布（batch_size, vocab_size）

            # 从概率分布中采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)  # 根据概率分布随机采样一个token索引（batch_size, 1）

        # 如果没有温度缩放，则按照logits最大值选择token
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # 选择logits值最大的索引作为下一个token（batch_size, 1）

        # 如果遇到eos_id（结束符），则提前停止生成
        if idx_next == eos_id:  # 如果生成的token是结束符，且指定了eos_id，则停止生成
            break

        # 将生成的token加入到当前的序列中
        idx = torch.cat((idx, idx_next), dim=1)  # 将生成的token追加到当前序列的末尾，形成新的序列（batch_size, num_tokens+1）

