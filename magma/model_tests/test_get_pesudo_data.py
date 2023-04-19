import sys
sys.path.append('..')
sys.path.append('/home/lfsm/code/magma/magma')
from megatron.neox_arguments import NeoXArgs
import torch

neox_args = NeoXArgs.from_ymls(['/home/lfsm/code/magma/configs/19M.yml','/home/lfsm/code/magma/configs/local_setup.yml'])
neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab

data_list = list()
context_tokens_tensor = torch.randint(
    0, neox_args.padded_vocab_size, (4, neox_args.seq_length + 1)
).to(torch.int64)
for i in range(10):
    data_list.append({"text": context_tokens_tensor.clone()})
data_iterator = iter(data_list)
batch = next(data_iterator)

from megatron.training import get_batch_pipe
a = get_batch_pipe(batch,neox_args)
lm_input = torch.load('./model_tests/lm_inputs.pt')

## build label