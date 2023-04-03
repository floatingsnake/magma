import sys
sys.path.append('..')
import os
from magma import Magma
import deepspeed
import torch
from magma.utils import parse_args, configure_param_groups

args = parse_args()
deepspeed.init_distributed()
args.local_rank =int(os.environ['LOCAL_RANK'])
args.world_size =int(os.environ['WORLD_SIZE'])

if __name__ == "__main__":
    args=parse_args() 
    deepspeed.init_distributed()
    #args.local_rank, args.world_rank, args.world_size = world_info_from_env()
    model = Magma(
	args.config,
	#device=torch.device("cuda",args.local_rank)
    )  # for finetuning one might want to load the model via Magma.from_checkpoint(...) here
    tokenizer, config, transforms = model.tokenizer, model.config, model.transforms
    


    # filter frozen from trainable parameters:
    trainable_parameters = configure_param_groups(model, config)
    opt = torch.optim.AdamW(
        trainable_parameters,
        config.lr,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )

    model_engine, _, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        #optimizer=opt,
        model_parameters=trainable_parameters,
        config_params=config.deepspeed_config_params,
    )
    mbs = 1
    images, captions = torch.Tensor(mbs,3,224,224).half().cuda(), torch.Tensor(mbs,2048).half().cuda()
    outputs = model_engine(images, captions)
    loss = outputs.loss
    model_engine.backward(loss)
    model_engine.step()


