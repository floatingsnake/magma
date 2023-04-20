import sys
sys.path.append('..')
sys.path.append('/home/lfsm/code/magma/magma')
from megatron.neox_arguments import NeoXArgs
import torch

neox_args = NeoXArgs.from_ymls(['/home/lfsm/code/magma/configs/800M.yml','/home/lfsm/code/magma/configs/local_setup.yml'])
neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab

# data_list = list()
# context_tokens_tensor = torch.randint(
#     0, neox_args.padded_vocab_size, (4, neox_args.seq_length + 1)
# ).to(torch.int64)
# for i in range(10):
#     data_list.append({"text": context_tokens_tensor.clone()})
# data_iterator = iter(data_list)
# batch = next(data_iterator)

mbs = 4
data_list = list()
images, captions = torch.Tensor(mbs,3,224,224).half().cuda(), torch.Tensor(mbs,2048).long().cuda()
for i in range(10):
    data_list.append([images, captions])
data_iterator = iter(data_list)
batch = next(data_iterator)

from functools import partial
from test_get_batch import get_pipeline_batch
process = partial(
        get_pipeline_batch, eos_token=0
    )
out = process(*batch)

print(out)
for i in out:
    for j in i:
        print(j.shape)

## build label

