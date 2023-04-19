import sys
sys.path.append('..')
sys.path.append('/home/lfsm/code/magma/magma')

import torch
from megatron.neox_arguments import NeoXArgs
from megatron.initialize import initialize_megatron
from megatron import mpu
import deepspeed
from megatron.model import GPT2ModelPipe

def freeze_model(model):
    for p in model.parameters():
        p.require_grad=False

neox_args = NeoXArgs.from_ymls(['/home/lfsm/code/magma/configs/800M.yml','/home/lfsm/code/magma/configs/local_setup.yml'])
neox_args.configure_distributed_args()
neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab

from config import MultimodalConfig
from image_prefix import ImagePrefix
device = torch.device('cuda',neox_args.local_rank)
model = ImagePrefix(
    config=MultimodalConfig.from_yml('/home/lfsm/code/magma/configs/summit_clipH_pythia70m_web.yml'),
    out_dim=neox_args.hidden_size,
)

# initialize_megatron(neox_args=neox_args)
# # model = Combined_Net(neox_args=neox_args)
# model = GPT2ModelPipe(
#             neox_args=neox_args,
#             num_tokentypes=0,
#             parallel_output=True,
#             topology=mpu.get_topology(),
#         ) 

# # for p in model.named_modules():
# #     print(p)

# print(model)

# # import pdb;pdb.set_trace()
model,_,_,_ = deepspeed.initialize(
    model = model,
    args = neox_args,
    config_params = neox_args.deepspeed_config
)

# device = torch.device('cuda',neox_args.local_rank)

print("{} GPU used memory is {:.2f} GB".format(neox_args.local_rank,torch.cuda.max_memory_allocated(device)/1073741824))

'''
gpt-neox-800M:
    2gpus: 17GB
        0 GPU used memory is 7.54 GB
        1 GPU used memory is 9.42 GB

pure image_prefix:
        0 GPU used memory is 4.73 GB
        1 GPU used memory is 5.91 GB

w image_prefix
    2gpus: 27GB
        0 GPU used memory is 12.27 GB
        1 GPU used memory is 15.33 GB
    4gpus: 36GB
        0 GPU used memory is 7.67 GB
        2 GPU used memory is 9.20 GB
        1 GPU used memory is 9.20 GB
        3 GPU used memory is 9.20 GB
        
w adapter
    2gpus:

'''