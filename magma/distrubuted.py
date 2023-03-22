import os

import torch

def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)

def is_using_distributed():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False


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


def init_distributed(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    if is_using_distributed():
        if 'SLURM_PROCID' in os.environ:
            # DDP via SLURM
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ['LOCAL_RANK'] = str(args.local_rank)
            os.environ['RANK'] = str(args.rank)
            os.environ['WORLD_SIZE'] = str(args.world_size)
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
        else:
            master_addr = os.getenv("MASTER_ADDR", default="localhost")
            master_port = os.getenv('MASTER_PORT', default='8888')
            args.dist_url = "tcp://{}:{}".format(master_addr, master_port)
            args.local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
            args.rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
            args.world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1'))
            torch.distributed.init_process_group("nccl", init_method=args.dist_url, rank=args.rank, world_size=args.world_size)
            os.environ['LOCAL_RANK'] = str(args.local_rank)
            os.environ['RANK'] = str(args.rank)
            os.environ['WORLD_SIZE'] = str(args.world_size)
 
            # DDP via torchrun, torch.distributed.launch
            #args.local_rank, _, _ = world_info_from_env()
            #torch.distributed.init_process_group(
            #    backend=args.dist_backend,
            #    init_method=args.dist_url)
            #args.world_size = torch.distributed.get_world_size()
            #args.rank = torch.distributed.get_rank()
        args.distributed = True
