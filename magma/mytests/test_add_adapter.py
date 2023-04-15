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
        neox_args,
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
            mlp = getattr(module,ff_attr)
            adapter = Adapter(
                        dim=neox_args.hidden_size,
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
                        dim=neox_args.hidden_size,
                        downsample_factor=downsample_factor,
                        **adapter_kwargs
                        )
            setattr(module,attn_attr,adapter_layer)          
    return model                 

add_adapters(neox_args,lm,location='mlp') 
add_adapters(neox_args,lm,location='attention') 

# magma

from magma.magma import Magma
config = r'/home/lfsm/code/magma/configs/summit_clipH_pythia70m_web.yml'
magma = Magma(config=config)
print(lm)
print(magma)

'''
original magma:
          (attention): AdapterWrapper(
        (adapter): Sequential(
          (0): Linear(in_features=512, out_features=64, bias=True)
          (1): ReLU()
          (2): Linear(in_features=64, out_features=512, bias=True)
        )
        (attn_block): GPTNeoXAttention(
          (rotary_emb): RotaryEmbedding()
          (query_key_value): Linear(in_features=512, out_features=1536, bias=True)
          (dense): Linear(in_features=512, out_features=512, bias=True)
        )
      )
      (mlp): Sequential(
        (0): GPTNeoXMLP(
          (dense_h_to_4h): Linear(in_features=512, out_features=2048, bias=True)
          (dense_4h_to_h): Linear(in_features=2048, out_features=512, bias=True)
          (act): GELUActivation()
        )
        (1): Adapter(
          (adapter): Sequential(
            (0): Linear(in_features=512, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=512, bias=True)
          )
        )

my model
    (attention): AdapterWrapper(
      (adapter): Sequential(
        (0): Linear(in_features=512, out_features=128, bias=True)
        (1): ReLU()
        (2): Linear(in_features=128, out_features=512, bias=True)
      )
      (attn_block): ParallelSelfAttention(
        (query_key_value): ColumnParallelLinear()
        (rotary_emb): RotaryEmbedding()
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0, inplace=False)
        (dense): RowParallelLinear()
      )
    )
    (mlp): Sequential(
      (0): ParallelMLP(
        (dense_h_to_4h): ColumnParallelLinear()
        (dense_4h_to_h): RowParallelLinear()
      )
      (1): Adapter(
        (adapter): Sequential(
          (0): Linear(in_features=8888, out_features=2222, bias=True)
          (1): ReLU()
          (2): Linear(in_features=2222, out_features=8888, bias=True)
        )
      )

'''
