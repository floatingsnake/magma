import sys
sys.path.append('..')
import torch
import os
import deepspeed
from magma.magma import Magma
from magma.utils import parse_args, configure_param_groups

def ensure_path_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

args = parse_args()
deepspeed.init_distributed()

args.local_rank =int(os.environ['LOCAL_RANK'])
args.world_size =int(os.environ['WORLD_SIZE'])

if args.local_rank == 0:
    ensure_path_exist(os.environ['LOG_PATH'])

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

with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=0),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(os.environ['LOG_PATH']),
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:  
    model_engine, opt, train_loader, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=opt,
        config_params=config.deepspeed_config_params,
    )
    mbs=1
    images = torch.Tensor(mbs,3,224,224).half().cuda()
    captions = torch.zeros(mbs,2048).long().cuda()
    outputs = model_engine(images, captions)
    loss = outputs.loss
    model_engine.backward(loss)
    prof.step()

'''
- pythia70m+openclipH 2.71 GB
    - with extra store grads
        - after deepspeed init 9.20 GB
        - after forward 11.35 GB
        - after backward 10.71 GB
    - free
        - after deepspeed init 8.42 GB
        - after forward 10.48 GB
        - after backward 9.79 GB
        
- pythia1b+openclipH 6.38 GB
    after deepspeed init
    - with extra store grads   21.96 GB
    - free                     10.66 GB
    - stage2 2 gpus             6.92 GB
    - stage3 2 gpus             6.91 GB
    - stage2 4 gpus             5.06 GB
    - stage3 4 gpus             3.95 GB
    
'''


