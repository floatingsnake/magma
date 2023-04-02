import sys
sys.path.append('..')
import os
from magma import Magma
import deepspeed
import torch
from magma.utils import parse_args, configure_param_groups

args = parse_args()
deepspeed.init_distributed()

args.local_rank =int(os.environ['LOCAL_RANK'])
args.world_size =int(os.environ['WORLD_SIZE'])

device = torch.device("cuda", int(args.local_rank))
model = Magma(args.config,
              device)

config = model.config

trainable_parameters = configure_param_groups(model,config)
opt = torch.optim.AdamW(
    trainable_parameters,
    config.lr,
    betas=(0.9, 0.95),
    weight_decay=config.weight_decay,
)

model_engine, opt, train_loader, lr_scheduler = deepspeed.initialize(
    model=model,
    optimizer=opt,
    config_params=config.deepspeed_config_params,
)
print('init_done')

print("GPU used memory is {:.2f} GB".format(torch.cuda.max_memory_allocated(device)/1073741824))
mbs=1
images = torch.Tensor(mbs,3,224,224).half().cuda()
captions = torch.zeros(mbs,2048).long().cuda()
outputs = model_engine(images, captions)
print("GPU used memory is {:.2f} GB".format(torch.cuda.max_memory_allocated(device)/1073741824))
loss = outputs.loss
model_engine.backward(loss)
print("GPU used memory is {:.2f} GB".format(torch.cuda.max_memory_allocated(device)/1073741824))
