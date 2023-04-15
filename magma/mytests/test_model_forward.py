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

neox_args = NeoXArgs.from_ymls(['/home/lfsm/code/magma/configs/19M.yml','/home/lfsm/code/magma/configs/local_setup.yml'])
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
from test_add_adapter import add_adapters
# add_adapters(neox_args,model,location='mlp') 
# add_adapters(neox_args,model,location='attention') 
optimizer = torch.optim.Adam(model.parameters())
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
from functools import partial
from megatron.training import get_batch_pipe
model.set_batch_fn(
    partial(
        get_batch_pipe, neox_args=neox_args, curr_scheduler=None
    )
)
# pesudo dataset
data_list = list()
context_tokens_tensor = torch.randint(
    0, neox_args.padded_vocab_size, (4, neox_args.seq_length + 1)
).to(torch.int64)
for i in range(10):
    data_list.append({"text": context_tokens_tensor.clone()})
data_iterator = iter(data_list)
# training test
from megatron.training import train_step
from megatron.utils import Timers
timers = Timers(use_wandb=False, tensorboard_writer=None)


for i in range(5):
    model.train_batch(data_iter=data_iterator)
    print(f'finish {i} step')
