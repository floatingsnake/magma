import subprocess

result = subprocess.run(['ninja','--version'],stdout=subprocess.PIPE)
print(result.stdout.decode('utf-8'))

import pdb;pdb.set_trace()
import ninja
import apex
import deepspeed
from transformers import GPTNeoForCausalLM, GPTNeoXForCausalLM, AutoConfig, GPT2LMHeadModel
print('dependence done!')
