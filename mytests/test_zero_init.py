import sys
sys.path.append('..')
import torch
import torch.nn as nn
from magma.utils import parse_args
from magma.config import MultimodalConfig
import deepspeed

class HugeNet(nn.Module):
    def __init__(self,config):
        super(HugeNet, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(1000, 10000)
        self.fc2 = nn.Linear(10000, 100000)
        self.fc3 = nn.Linear(100000, 10000)
        self.fc4 = nn.Linear(10000, 10000)

        self.device=None

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        
args = parse_args()
config = MultimodalConfig.from_yml(args.config)
device = torch.device("cuda", int(args.local_rank))

with deepspeed.zero.Init(config_dict_or_path=config.deepspeed_config_params):
    model = HugeNet(config)
print('model params is {} M'.format(sum(p.numel() for p in model.parameters())/1e6))
print("{} GPU used memory is {:.2f} GB".format(device,torch.cuda.max_memory_allocated('cuda:0')/1073741824))

opt = torch.optim.AdamW(
    model.parameters(),
    model.config.lr,
    betas=(0.9, 0.95),
    weight_decay=model.config.weight_decay,
)

model_engine, opt, _, lr_scheduler = deepspeed.initialize(
    model=model,
    optimizer=opt,
    config_params=model.config.deepspeed_config_params,
)
print("{} GPU used memory is {:.2f} GB".format(device,torch.cuda.max_memory_allocated('cuda:0')/1073741824))

# mbs=1
# x = torch.Tensor(mbs,1000).half().cuda()
# print("GPU used memory is {:.2f} GB".format(torch.cuda.max_memory_allocated(device)/1073741824))
# outputs = model_engine(x)
