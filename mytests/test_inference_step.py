import sys
sys.path.append('..')
import torch
import os
import deepspeed
from magma.magma import Magma
from magma.utils import parse_args, configure_param_groups
from torchvision.utils import make_grid

args=parse_args() 
deepspeed.init_distributed()

args.local_rank =int(os.environ['LOCAL_RANK'])
# args.local_rank = 0

args.world_size =int(os.environ['WORLD_SIZE'])
device = torch.device("cuda", int(args.local_rank))

model = Magma(
    args.config,
    device)  
# for finetuning one might want to load the model via Magma.from_checkpoint(...) here
config = model.config

# filter frozen from trainable parameters:
trainable_parameters = configure_param_groups(model, config)
opt = torch.optim.AdamW(
    trainable_parameters,
    config.lr,
    betas=(0.9, 0.95),
    weight_decay=config.weight_decay,
)
if args.local_rank==0:
    print("Current GPU memory usage: {:.2f} GB".format(torch.cuda.memory_allocated() / (1024*1024*1024)))

model_engine, _, _, lr_scheduler = deepspeed.initialize(
    args=args,
    model=model,
    optimizer=opt,
    model_parameters=trainable_parameters,
    config_params=config.deepspeed_config_params,
)

if args.local_rank==0:
    print("Af deepspeed init GPU memory usage: {:.2f} GB".format(torch.cuda.memory_allocated() / (1024*1024*1024)))

mbs=1
# images = torch.Tensor(mbs,3,224,224).half().cuda()
# import pdb;pdb.set_trace()
# images = torch.zeros_like(images)
# embeddings = model.embed(images)
embeddings=torch.Tensor(1,1,512).half().cuda()
embeddings = torch.zeros_like(embeddings)

captions = model_engine.generate(
    embeddings,
    max_steps = 6,
    temperature = 0.7,
    top_k = 0,
)  # [caption1, caption2, ... b]
# width = min(2, images.shape[0])
# image_grid = make_grid(images[:width])
# caption = ""
# for i in range(width):
#     caption += f"Caption {i}: \n{captions[i]}\n"
print(captions)