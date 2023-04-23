from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig
import torch

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
config = GPTNeoXConfig.from_pretrained("EleutherAI/pythia-70m")
config.is_decoder = True
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m", config=config)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
print(outputs)
prediction_logits = outputs.logits
