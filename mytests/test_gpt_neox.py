import torch
from transformers import GPTNeoXForCausalLM, AutoConfig
from typing import Optional

def print_num_params(model):
    num_params = sum(torch.numel(p) for p in model.parameters())
    print(f'model size is {num_params/1000000} M')

model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped")
print_num_params(model)


model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-19m")
print_num_params(model)

def neox_config(path: Optional[str] = None):
    config = AutoConfig.from_pretrained(path if path is not None else "EleutherAI/pythia-19m")
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
config=neox_config()
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-19m",config=config,ignore_mismatched_sizes=True)
print_num_params(model)

config = AutoConfig.from_pretrained("EleutherAI/pythia-19m")
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-19m",config=config,ignore_mismatched_sizes=True)
print_num_params(model)


# result:
# model size is 70.426624 M
# model size is 70.426624 M
# model size is 916.434944 M
# model size is 70.426624 M
