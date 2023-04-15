import torch
from megatron.neox_arguments import NeoXArgs
from megatron.initialize import initialize_megatron
from megatron import mpu
import deepspeed
from megatron.model import GPT2ModelPipe
def freeze_model(model):
    for p in model.parameters():
        p.require_grad=False

class Combined_Net(torch.nn.Module):
    '''Pipeline Model with nn.Module'''
    def __init__(
        self,
        neox_args,
        use_cache=False,
    ):
        device = torch.device('cuda',neox_args.local_rank)
        super().__init__()
        self.lm = GPT2ModelPipe(
            neox_args=neox_args,
            num_tokentypes=0,
            parallel_output=True,
            topology=mpu.get_topology(),
            use_cache=use_cache,
        )
        import pdb;pdb.set_trace()
        self.net = torch.nn.Linear(9000,40000)
        # self.net = torch.nn.Linear(9000,40000).to(device)
        self.lm.insert_layers(self.net, 0)


neox_args = NeoXArgs.from_ymls(['/home/lfsm/code/magma/configs/800M.yml','/home/lfsm/code/magma/configs/local_setup.yml'])
neox_args.configure_distributed_args()
neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab

initialize_megatron(neox_args=neox_args)
# model = Combined_Net(neox_args=neox_args)
model = GPT2ModelPipe(
            neox_args=neox_args,
            num_tokentypes=0,
            parallel_output=True,
            topology=mpu.get_topology(),
        ) 

for p in model.named_modules():
    print(p)

import pdb;pdb.set_trace()
# model,_,_,_ = deepspeed.initialize(
#     model = model,
#     args = neox_args,
#     config_params = neox_args.deepspeed_config
# )

device = torch.device('cuda',neox_args.local_rank)

print("{} GPU used memory is {:.2f} GB".format(neox_args.local_rank,torch.cuda.max_memory_allocated(device)/1073741824))
