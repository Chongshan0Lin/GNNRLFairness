import torch

gpu_index = 2
device = torch.device(f"cuda:{gpu_index}"if torch.cuda.is_available() else "cpu")
print(device)