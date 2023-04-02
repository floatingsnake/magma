import torch

device_count = torch.cuda.device_count()
a = 5
b = a%device_count
print(b)

print(f"Number of available CUDA devices: {device_count}")
