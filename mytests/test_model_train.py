import sys
sys.path.append('..')
import argparse
import torch
import os
import deepspeed
from magma.magma import Magma
from magma.webdataset import get_wds_dataset
from magma.utils import configure_param_groups

def world_info_from_env():
    local_rank = 0
    for v in ('SLURM_LOCALID', 'MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('SLURM_PROCID', 'PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('SLURM_NTASKS', 'PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=False, help="path to your training config",
	default='/ccs/home/lfsm/code/magma/configs/benchmark_20b_mbs1.yml')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args=parse_args() 
    #deepspeed.init_distributed()
    #args.local_rank, args.world_rank, args.world_size = world_info_from_env()
    model = Magma(
	args.config,
	#device=torch.device("cuda",args.local_rank)
    )  # for finetuning one might want to load the model via Magma.from_checkpoint(...) here
    tokenizer, config, transforms = model.tokenizer, model.config, model.transforms
    
    import pdb;pdb.set_trace()

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


