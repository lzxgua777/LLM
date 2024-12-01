from importlib.metadata import version  # 导入版本检查模块

from ch05.user_interface.app_orig import device  # 从指定模块导入device对象

pkgs = [
    "torch",  # 定义需要检查版本的包列表，这里只有PyTorch
]
for p in pkgs:  # 遍历包列表并打印每个包的版本
    print(f"{p} version: {version(p)}")  # 打印包的版本号

import gc  # 导入垃圾回收模块
import time  # 导入时间模块
import torch  # 导入PyTorch库

# 定义一个函数来初始化GPU内存跟踪
def start_memory_tracking():
    if torch.cuda.is_available():  # 如果CUDA可用
        torch.cuda.reset_peak_memory_stats()  # 重置峰值内存统计
    else:
        print("This notebook is intended for CUDA GPUs but CUDA is not available.")  # 提示CUDA不可用

# 定义一个函数来打印当前GPU内存使用情况
def print_memory_usage():
    max_gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # 将字节转换为GB
    print(f"Maximum GPU memory allocated: {max_gpu_memory:.1f} GB")  # 打印已分配的最大GPU内存

# 定义一个函数来清理内存并打印当前GPU内存使用情况
def cleanup():
    gc.collect()  # 运行Python垃圾回收器
    torch.cuda.empty_cache()  # 清空CUDA缓存
    time.sleep(3)  # 等待3秒以允许内存清理
    torch.cuda.reset_peak_memory_stats()  # 重置峰值内存统计
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # 获取设备上分配的最大内存
    print(f"Maximum GPU memory allocated: {max_memory_allocated:.1f} GB")  # 打印设备上分配的最大GPU内存

from ch04.main_chapter_code.ch04 import GPTModel  # 从指定模块导入GPTModel类

# 定义基础配置
BASE_CONFIG = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "drop_rate": 0.0,        # Dropout率
    "qkv_bias": True         # Query-key-value偏置
}

# 定义不同规模的GPT模型配置
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-xl (1558M)"  # 选择要使用的模型配置

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])  # 更新基础配置为所选模型配置

start_memory_tracking()  # 开始跟踪内存使用

model = GPTModel(BASE_CONFIG)  # 创建GPT模型实例
device = torch.device("cuda")  # 获取CUDA设备
model.to(device)  # 将模型部署到CUDA设备

print_memory_usage()  # 打印内存使用情况

# 测试模型是否工作正常（这里不需要跟踪内存）
test_input = torch.tensor([[1, 2, 3]]).to(device)  # 创建测试输入并部署到CUDA设备
model.eval()  # 将模型设置为评估模式

with torch.no_grad():  # 在不计算梯度的上下文中
    model(test_input)  # 运行模型

# 训练代码将放在这里...

model.train()  # 将模型设置为训练模式
torch.save(model.state_dict(), "model.pth")  # 保存模型权重

del model, test_input  # 删除模型和测试输入以释放内存
cleanup()  # 清理内存

# 然后加载预训练权重

start_memory_tracking()  # 开始跟踪内存使用

model = GPTModel(BASE_CONFIG)  # 重新创建GPT模型实例
model.to(device)  # 将模型部署到CUDA设备

model.load_state_dict(  # 加载模型权重
    torch.load("model.pth", map_location=device, weights_only=True)
)
model.to(device)  # 将模型部署到CUDA设备
model.eval();  # 将模型设置为评估模式

print_memory_usage()  # 打印内存使用情况

# 测试模型是否工作正常（这里不需要跟踪内存）
test_input = torch.tensor([[1, 2, 3]]).to(device)  # 创建测试输入并部署到CUDA设备
model.eval()  # 将模型设置为评估模式

with torch.no_grad():  # 在不计算梯度的上下文中
    model(test_input)  # 运行模型

del model, test_input  # 删除模型和测试输入以释放内存
cleanup()  # 清理内存

start_memory_tracking()  # 开始跟踪内存使用

model = GPTModel(BASE_CONFIG).to(device)  # 创建GPT模型实例并部署到CUDA设备

state_dict = torch.load("model.pth", map_location="cpu", weights_only=True)  # 加载模型权重到CPU

print_memory_usage()  # 打印内存使用情况

# 顺序地将权重复制到模型的参数中
with torch.no_grad():  # 在不计算梯度的上下文中
    for name, param in model.named_parameters():  # 遍历模型的所有参数
        if name in state_dict:  # 如果权重存在于状态字典中
            param.copy_(state_dict[name].to(device))  # 将权重复制到参数
        else:
            print(f"Warning: {name} not found in state_dict.")  # 提示权重未找到

print_memory_usage()  # 打印内存使用情况

# 测试模型是否工作正常（这里不需要跟踪内存）
test_input = torch.tensor([[1, 2, 3]]).to(device)  # 创建测试输入并部署到CUDA设备
model.eval()  # 将模型设置为评估模式

with torch.no_grad():  # 在不计算梯度的上下文中
    model(test_input)  # 运行模型

del model, test_input, state_dict, param  # 删除模型、测试输入、状态字典和参数以释放内存
cleanup()  # 清理内存

import os  # 导入操作系统接口模块
import psutil  # 导入系统和进程实用工具库
from threading import Thread  # 导入线程模块

# 定义一个函数来测量函数运行期间的内存使用情况（以GB为单位）
def memory_usage_in_gb(func, *args, **kwargs):
    process = psutil.Process(os.getpid())  # 获取当前进程

    # 测量运行函数前的基线内存使用情况
    baseline_mem = process.memory_info().rss / 1024 ** 3  # 以GB为单位

    # 在单独的线程中开始监控内存
    mem_usage = []
    done = False

    def monitor_memory():  # 定义监控内存的函数
        while not done:
            mem_usage.append(process.memory_info().rss / 1024 ** 3)  # 转换为GB
            time.sleep(0.1)

    t = Thread(target=monitor_memory)  # 创建线程
    t.start()  # 启动线程

    # 运行函数
    func(*args, **kwargs)

    # 停止监控
    done = True
    t.join()  # 等待线程结束

    peak_mem_usage_gb = max(mem_usage) - baseline_mem  # 计算峰值内存使用情况
    return peak_mem_usage_gb  # 返回峰值内存使用情况
def load_sequentially():
    start_memory_tracking()  # 开始跟踪内存使用

    model = GPTModel(BASE_CONFIG).to(device)  # 创建GPT模型实例并部署到设备

    state_dict = torch.load("model.pth", map_location="cpu", weights_only=True)  # 加载模型权重到CPU

    print_memory_usage()  # 打印当前内存使用情况

    # 顺序地将权重复制到模型的参数中
    with torch.no_grad():  # 在不计算梯度的上下文中
        for name, param in model.named_parameters():  # 遍历模型的所有参数
            if name in state_dict:  # 如果权重存在于状态字典中
                param.copy_(state_dict[name].to(device))  # 将权重复制到参数
            else:
                print(f"Warning: {name} not found in state_dict.")  # 提示权重未找到

    print_memory_usage()  # 再次打印内存使用情况


peak_memory_used = memory_usage_in_gb(load_sequentially)  # 测量函数运行期间的CPU内存峰值
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")  # 打印CPU内存峰值


def load_sequentially_with_meta():
    start_memory_tracking()  # 开始跟踪内存使用

    with torch.device("meta"):  # 使用meta设备创建模型
        model = GPTModel(BASE_CONFIG)

    model = model.to_empty(device=device)  # 将模型部署到指定设备

    state_dict = torch.load("model.pth", map_location=device, weights_only=True)  # 加载模型权重到设备

    print_memory_usage()  # 打印当前内存使用情况

    # 顺序地将权重复制到模型的参数中
    with torch.no_grad():  # 在不计算梯度的上下文中
        for name, param in model.named_parameters():  # 遍历模型的所有参数
            if name in state_dict:  # 如果权重存在于状态字典中
                param.copy_(state_dict[name])  # 将权重复制到参数
            else:
                print(f"Warning: {name} not found in state_dict.")  # 提示权重未找到

    print_memory_usage()  # 再次打印内存使用情况


peak_memory_used = memory_usage_in_gb(load_sequentially_with_meta)  # 测量函数运行期间的CPU内存峰值
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")  # 打印CPU内存峰值


def baseline():
    start_memory_tracking()  # 开始跟踪内存使用

    model = GPTModel(BASE_CONFIG)  # 创建GPT模型实例
    model.to(device)  # 将模型部署到设备

    model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))  # 加载模型权重
    model.to(device)  # 再次将模型部署到设备
    model.eval();  # 将模型设置为评估模式

    print_memory_usage()  # 打印当前内存使用情况


peak_memory_used = memory_usage_in_gb(baseline)  # 测量函数运行期间的CPU内存峰值
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")  # 打印CPU内存峰值


def best_practices():
  with torch.device("meta"):  # 使用meta设备创建模型
      model = GPTModel(BASE_CONFIG)

  model.load_state_dict(  # 加载模型权重
      torch.load("model.pth", map_location=device, weights_only=True, mmap=True),
      assign=True
  )

  print_memory_usage()  # 打印当前内存使用情况


peak_memory_used = memory_usage_in_gb(best_practices)  # 测量函数运行期间的CPU内存峰值
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")  # 打印CPU内存峰值


model = GPTModel(BASE_CONFIG)  # 创建GPT模型实例
# 假设`model`是你的训练模型
state_dict = model.state_dict()  # 获取模型的状态字典

# 创建一个目录来存储单独的参数文件
os.makedirs("model_parameters", exist_ok=True)

# 分别保存每个参数张量
for name, param in state_dict.items():  # 遍历状态字典中的参数
    torch.save(param.cpu(), f"model_parameters/{name}.pt")  # 将参数保存到文件

del model  # 删除模型以释放内存


def load_individual_weights():
    start_memory_tracking()  # 开始跟踪内存使用

    with torch.device("meta"):  # 使用meta设备创建模型
        model = GPTModel(BASE_CONFIG)

    model = model.to_empty(device=device)  # 将模型部署到指定设备

    print_memory_usage()  # 打印当前内存使用情况
    param_dir = "model_parameters"  # 参数目录路径

    with torch.no_grad():  # 在不计算梯度的上下文中
        for name, param in model.named_parameters():  # 遍历模型的所有参数
            weight_path = os.path.join(param_dir, f"{name}.pt")  # 参数文件路径
            if os.path.exists(weight_path):  # 如果参数文件存在
                param_data = torch.load(weight_path, map_location="cpu", weights_only=True)  # 加载参数数据
                param.copy_(param_data)  # 将参数数据复制到模型参数
                del param_data  # 删除参数数据以释放内存
            else:
                print(f"Warning: {name} not found in {param_dir}.")  # 提示参数文件未找到

    print_memory_usage()  # 再次打印内存使用情况


peak_memory_used = memory_usage_in_gb(load_individual_weights)  # 测量函数运行期间的CPU内存峰值
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")  # 打印CPU内存峰值






