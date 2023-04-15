import sys
sys.path.append('..')
import torch
import os
import deepspeed
from magma.magma import Magma
from magma.utils import parse_args, configure_param_groups


args = parse_args()
deepspeed.init_distributed()

args.local_rank =int(os.environ['LOCAL_RANK'])
args.world_size =int(os.environ['WORLD_SIZE'])


device = torch.device("cuda", int(args.local_rank))
from transformers.deepspeed import HfDeepSpeedConfig
deepspeed_config='./ds_config.json'
dschf = HfDeepSpeedConfig(deepspeed_config)
#with deepspeed.zero.Init():
model = Magma(args.config, device=device)


config = model.config
print("{} GPU used memory is {:.2f} GB".format(model.device,torch.cuda.max_memory_allocated(device)/1073741824))

trainable_parameters = configure_param_groups(model,config)
opt = torch.optim.AdamW(
    trainable_parameters,
    config.lr,
    betas=(0.9, 0.95),
    weight_decay=config.weight_decay,
)

model_engine, opt, _, lr_scheduler = deepspeed.initialize(
    model=model,
    optimizer=opt,
    config_params=config.deepspeed_config_params,
)

print("GPU used memory is {:.2f} GB".format(torch.cuda.max_memory_allocated('cuda:0')/1073741824))

print(model.device)
