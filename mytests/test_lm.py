from transformers import GPTNeoXForCausalLM, AutoConfig, GPT2LMHeadModel
import torch
config = {
    "vocab_size": 50257,
    "hidden_size": 2048,
    "num_layers": 40,
    "num_heads": 32,
    "max_position_embeddings": 2048,
    "embedding_size": 2048,
    "dropout": 0.1,
}
model = GPTNeoXForCausalLM(config)
model.to('cuda:0')
print("Current GPU memory usage: {:.2f} GB".format(torch.cuda.memory_allocated() / (1024*1024*1024)))
