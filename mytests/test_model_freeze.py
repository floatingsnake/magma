import sys
sys.path.append('..')
from magma.magma import Magma
import deepspeed
import torch

config = r'/home/lfsm/code/magma/configs/benchmark_mbs1.yml'
device='cuda:0'

model = Magma(config,device)
for name, param in model.lm.named_parameters():  # freeze lm weights
    if param.requires_grad == False:
        print(name)
print('freezed check done')
