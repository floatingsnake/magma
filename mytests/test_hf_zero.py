import sys
sys.path.append('..')
import torch
import os
import deepspeed
from magma.magma import Magma
from magma.utils import parse_args, configure_param_groups
from transformers import GPTNeoForCausalLM, GPTNeoXForCausalLM, AutoConfig


args = parse_args()
deepspeed.init_distributed()

args.local_rank =int(os.environ['LOCAL_RANK'])
args.world_size =int(os.environ['WORLD_SIZE'])


device = torch.device("cuda", int(args.local_rank))

class MixNet(torch.nn.Module):
    def __init__(self, device=None, init_weights=True):
        super().__init__()

        self.ln = torch.nn.Linear(10,2048)
        config = AutoConfig.from_pretrained("EleutherAI/pythia-1b")
        self.lm = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-1b",config=config,ignore_mismatched_sizes=True)
        self.device=device

    def forward(self,x):
        x = self.ln(x)
        x = self.lm(x)
        return x 

from transformers.deepspeed import HfDeepSpeedConfig
deepspeed_config='./ds_config.json'
dschf = HfDeepSpeedConfig(deepspeed_config)
config = AutoConfig.from_pretrained("EleutherAI/pythia-1b")
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-1b",config=config)
    # model = MixNet(device=device)

# opt = torch.optim.AdamW(
#     model.parameters(),
#     1e-8,
#     betas=(0.9, 0.95),
# )

# model_engine, opt, _, lr_scheduler = deepspeed.initialize(
#     model=model,
#     optimizer=opt,
#     config_params=deepspeed_config,
# )

print("{} GPU used memory is {:.2f} GB".format(device,torch.cuda.max_memory_allocated('cuda:0')/1073741824))

