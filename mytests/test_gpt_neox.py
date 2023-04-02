import torch
from transformers import GPTNeoForCausalLM, GPTNeoXForCausalLM, AutoConfig
from typing import Optional

def print_num_params(model):
    num_params = sum(torch.numel(p) for p in model.parameters())
    print(f'model size is {num_params/1000000} M')

def neox_config(path: Optional[str] = None):
    config = AutoConfig.from_pretrained(path if path is not None else "EleutherAI/pythia-70m-deduped",cache_dir='/gpfs/alpine/scratch/lfsm/csc499/neox_weights')
    config.rotary = True
    config.rotary_dim = 64
    config.jax = True
    config.gradient_checkpointing = True
    return config

def gptj_config():
    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B")
    config.attention_layers = ["global"] * 28
    config.attention_types = [["global"], 28]
    config.num_layers = 28
    config.num_heads = 16
    config.hidden_size = 256 * config.num_heads
    config.vocab_size = 50400
    config.rotary = True
    config.rotary_dim = 64
    config.jax = True
    config.gradient_checkpointing = True
    return config

cache_dir='/gpfs/alpine/scratch/lfsm/csc499/neox_weights'
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped",cache_dir=cache_dir)
print_num_params(model)

#config=neox_config('EleutherAI/gpt-neox-20b')
#model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b",config=config,cache_dir=cache_dir)
#print_num_params(model)
config=neox_config()
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-19m",config=config,cache_dir=cache_dir,ignore_mismatched_sizes=True)
print_num_params(model)
'''
config = AutoConfig.from_pretrained("EleutherAI/pythia-19m")
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-19m",config=config,ignore_mismatched_sizes=True)
print_num_params(model)
'''

'''
config = gptj_config()
model = GPTNeoForCausalLM(config=config)
print_num_params(model)

config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = GPTNeoForCausalLM(config=config)
print_num_params(model)
'''
# result:
# model size is 70.426624 M
# model size is 70.426624 M
# model size is 916.434944 M
# model size is 70.426624 M
# model size is 5853.126656 M
# model size is 2651.30752 M
