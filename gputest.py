import torch
print("CUDA是否可用:", torch.cuda.is_available())
print("可用的GPU数量:", torch.cuda.device_count())
print("当前GPU:", torch.cuda.current_device())
print("GPU名称:", torch.cuda.get_device_name(0))