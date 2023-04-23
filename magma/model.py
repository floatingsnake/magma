import torch
import torch.nn as nn
from megatron.neox_arguments import NeoXArgs
from megatron.initialize import initialize_megatron
from megatron import mpu
import deepspeed
from megatron.model import GPT2ModelPipe
from magma.adapters import Adapter,AdapterWrapper
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
    for names, module in model.named_modules():
        if 'image_prefix' in names:
          # avoid add adapter in image_prefix, may need modification when change the image_prefixe
          continue
        names = [name for name,module in module.named_modules()]
        if location in names and location==ff_attr:
            mlp = getattr(module,ff_attr)
            adapter_layer = AdapterWrapper(
                        attn_block=mlp,
                        dim=neox_args.hidden_size,
                        downsample_factor=downsample_factor,
                        **adapter_kwargs
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

def get_3d_parallel_magma(neox_args):
    """return a magama based on gpt-neox-pipeline"""
    model = GPT2ModelPipe(
            neox_args=neox_args,
            num_tokentypes=0,
            parallel_output=True,
            topology=mpu.get_topology(),
        )  
    # add adapter
    # if neox_args.add_adapter:
    add_adapters(neox_args,model,location='mlp') 
    add_adapters(neox_args,model,location='attention')