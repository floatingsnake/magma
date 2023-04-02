import sys
sys.path.append('..')
from magma.magma import Magma
import deepspeed
import torch

config = r'/home/lfsm/code/magma/configs/benchmark_mbs1.yml'
device='cuda:0'

model = Magma(config,device)
for name, param in model.lm.named_parameters():  # freeze lm weights
    if param.requires_grad == False:
        print(name)
print('freezed check done')
'''
Before:
freezed check done

After:
gpt_neox.embed_in.weight
gpt_neox.layers.0.input_layernorm.weight
gpt_neox.layers.0.input_layernorm.bias
gpt_neox.layers.0.post_attention_layernorm.weight
gpt_neox.layers.0.post_attention_layernorm.bias
gpt_neox.layers.0.attention.attn_block.query_key_value.weight
gpt_neox.layers.0.attention.attn_block.query_key_value.bias
gpt_neox.layers.0.attention.attn_block.dense.weight
gpt_neox.layers.0.attention.attn_block.dense.bias
gpt_neox.layers.0.mlp.0.dense_h_to_4h.weight
gpt_neox.layers.0.mlp.0.dense_h_to_4h.bias
gpt_neox.layers.0.mlp.0.dense_4h_to_h.weight
gpt_neox.layers.0.mlp.0.dense_4h_to_h.bias
...
gpt_neox.final_layer_norm.weight
gpt_neox.final_layer_norm.bias
embed_out.weight
freezed check done
'''