import torch

device='cuda:0'
cur_mem = torch.cuda.max_memory_allocated(device)
print("current used memory is {:.2f}".format(cur_mem/1073741824))

torch.cuda.empty_cache()
a = torch.zeros(4000,1000,1000).to(device)
cur_mem = torch.cuda.max_memory_allocated(device)
print("{:.2f} GB".format(torch.cuda.max_memory_allocated('cuda:0')/1073741824))
