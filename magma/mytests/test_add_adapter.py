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
from magma.adapters import Adapter,AdapterWrapper

neox_args = NeoXArgs.from_ymls(['/home/lfsm/code/magma/configs/19M.yml','/home/lfsm/code/magma/configs/local_setup.yml'])
neox_args.configure_distributed_args()
neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab

initialize_megatron(neox_args=neox_args)
lm = GPT2ModelPipe(
            neox_args=neox_args,
            num_tokentypes=0,
            parallel_output=True,
            topology=mpu.get_topology(),
        ) 


from typing import Literal
def add_adapters(
        model,
        downsample_factor: int = 4,
        # adapter_type: Literal["normal", "parallel", "scaled_parallel"] = "normal",
        location: Literal["mlp", "attention"] = "mlp",
        ff_attr: str = "mlp",
        attn_attr: str = "attention",
        **adapter_kwargs,    
):
    for _, module in model.named_modules():
        names = [name for name,module in module.named_modules()]
        if location in names and location==ff_attr:
            mlp = getattr(module,'mlp')
            adapter = Adapter(
                        dim=8888,
                        downsample_factor=downsample_factor,
                        **adapter_kwargs
                        )
            adapter_layer = nn.Sequential(
                *[
                    mlp,
                    adapter,
                ]
            )
            setattr(module,ff_attr,adapter_layer)   
        elif location in names and location==attn_attr:
            attn = getattr(module,attn_attr)
            adapter_layer = AdapterWrapper(
                        attn_block=attn,
                        dim=8888,
                        downsample_factor=downsample_factor,
                        **adapter_kwargs
                        )
            setattr(module,attn_attr,adapter_layer)          
    return model                 

lm_adapted = add_adapters(lm) 

# magma

from magma.magma import Magma
config = r'/home/lfsm/code/magma/configs/benchmark_mbs1_s2.yml'
magma = Magma(config=config)
import pdb;pdb.set_trace()


'''
# pesudo dataset
data_list = list()
context_tokens_tensor = torch.randint(
    0, neox_args.padded_vocab_size, (4, neox_args.seq_length + 1)
).to(torch.int64)
for i in range(100):
    data_list.append({"text": context_tokens_tensor.clone()})
data_iterator = iter(data_list)

# training test
from megatron.training import train_step
from megatron.utils import Timers
timers = Timers(use_wandb=False, tensorboard_writer=None)

loss_dict, skipped_iter = train_step(
    neox_args=neox_args,
    timers=timers,
    data_iterator=data_iterator,
    model=model,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
)
device = torch.device('cuda',neox_args.local_rank)
print("{} GPU used memory is {:.2f} GB".format(neox_args.local_rank,torch.cuda.max_memory_allocated(device)/1073741824))
'''
