import torch
#验证pytorch是cpu版本还是gpu版本
# 检查是否有可用的 GPU
gpu_available = torch.cuda.is_available()
print(f"CUDA Available: {gpu_available}")

# 检查当前使用的设备
current_device = torch.cuda.current_device() if gpu_available else "CPU"
print(f"Current Device: {current_device}")

# 如果有 GPU，输出 GPU 的名称
if gpu_available:
    gpu_name = torch.cuda.get_device_name(current_device)
    print(f"GPU Name: {gpu_name}")
else:
    print("Using CPU")