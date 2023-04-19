import sys
sys.path.append('..')
sys.path.append('/home/lfsm/code/magma/magma')

import torch
import torch.nn as nn
from megatron.neox_arguments import NeoXArgs
from megatron.initialize import initialize_megatron
from megatron import mpu
import deepspeed
from megatron.model import GPT2ModelPipe

# make config
torch.manual_seed(7)
neox_args = NeoXArgs.from_ymls(['/home/lfsm/code/magma/configs/800M.yml','/home/lfsm/code/magma/configs/local_setup.yml'])
neox_args.configure_distributed_args()
neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab

# init model,opt
initialize_megatron(neox_args=neox_args)
model = GPT2ModelPipe(
            neox_args=neox_args,
            num_tokentypes=0,
            parallel_output=True,
            topology=mpu.get_topology(),
        ) 

# add adapter
# from test_add_adapter import add_adapters
# add_adapters(neox_args,model,location='mlp') 
# add_adapters(neox_args,model,location='attention') 

# opt, lr
optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)

# deepspeed engine init
model, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    args=neox_args,
    dist_init_required=False,
    # model_parameters=_model_params,
    lr_scheduler = lr_scheduler,
    config_params=neox_args.deepspeed_config,
    mpu=mpu if not neox_args.is_pipe_parallel else None,
)

device = torch.device('cuda',neox_args.local_rank)
print("{} GPU used memory is {:.2f} GB".format(neox_args.local_rank,torch.cuda.max_memory_allocated(device)/1073741824))
