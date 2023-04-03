import sys
sys.path.append('..')
from magma.magma import Magma
from magma.utils import configure_param_groups
import torch

def print_num_params(model):
    num_params = sum(torch.numel(p) for p in model.parameters())
    print(f'model size is {num_params/1000000} M')

config = r'/home/lfsm/code/magma/configs/benchmark_1b_s3.yml'
device='cuda:0'

model = Magma(config,device)
trainable_parameters = configure_param_groups(model, model.config)

print(sum(p.numel() for p in trainable_parameters[0]['params'] if p.requires_grad)/1e6)
print(sum(p.numel() for p in trainable_parameters[1]['params'] if p.requires_grad)/1e6)
print(sum(p.numel() for p in model.image_prefix.parameters())/1e6)

print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)
print(sum(p.numel() for p in model.parameters())/1e6)

# import pdb;pdb.set_trace()
# for param in trainable_parameters[0]['params']:
#     if param.requires_grad == False:
#         print(param)
# for name, param in model.lm.named_parameters():  # freeze lm weights
#     if param.requires_grad == False:
#         print(name)
# print('freezed check done')
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