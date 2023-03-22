import time
import torch
import os
import deepspeed
import wandb
from torch.utils.data import random_split, ConcatDataset
from torch.optim import AdamW
from tqdm import tqdm
from functools import partial
from magma.magma import (
    Magma,
)
from magma.utils import (
    cycle,
    parse_args,
    wandb_log,
    wandb_init,
    print_main,
    configure_param_groups,
)
from magma.webdataset import get_wds_dataset
from magma.train_loop import (
    train_step,
)

# tell tokenizers not to do parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

if __name__ == "__main__":
    
    # parse command line arguments:
    args = parse_args()
    
    # distrubited init
    deepspeed.init_distributed()
    args.local_rank, args.world_rank, args.world_size = world_info_from_env()
  	
    device=torch.device("cuda",args.local_rank)
    model = Magma(
        args.config,
        device=device
    )  # for finetuning one might want to load the model via Magma.from_checkpoint(...) here
    #device=torch.device("cuda", args.local_rank)
    print("GPU used memory is {:.2f} GB".format(torch.cuda.max_memory_allocated(device)/1073741824))
    tokenizer, config, transforms = model.tokenizer, model.config, model.transforms

    # filter frozen from trainable parameters:
    trainable_parameters = configure_param_groups(model, config)

    # load data:
    #train_dataset, eval_dataset = get_pretraining_datasets(
    #    config, tokenizer, transforms
    #)
    def preprocess_text(text):
        return tokenizer.encode(
                    text,
                    return_tensors="pt",
                    max_length=2048,
                    padding="max_length",
                    truncation=True,)
    config.world_size=args.world_size
    data = get_wds_dataset(config, transforms, preprocess_text, is_train=True)
    data.set_epoch(0)
    train_loader=data.dataloader
   
    #for batch in train_loader:
    #    print('micro batch size is ',batch[0].squeeze(1).shape)
    #    break

    #print_main(f"Loaded train dataset with {len(train_dataset)} samples")
    #print_main(f"Loaded eval dataset with {len(eval_dataset)} samples")

    opt = AdamW(
        trainable_parameters,
        config.lr,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    print('after init opt, data, model')
    print("GPU used memory is {:.2f} GB".format(torch.cuda.max_memory_allocated(device)/1073741824))

    model_engine, opt, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=opt,
        model_parameters=trainable_parameters,
        config_params=config.deepspeed_config_params,
    )
    print("deepspeed init done")
    print("GPU used memory is {:.2f} GB".format(torch.cuda.max_memory_allocated(device)/1073741824))

    #eval_loader = cycle(model_engine.deepspeed_io(eval_dataset))
    train_loader = cycle(train_loader)

    # initialize training
    
    wandb_init(
        project=config.wandb_project,
        name=config.name or wandb.util.generate_id(),
        config=config,
        dir=os.environ['WANDB_DIR']
    )

    # training loop
    optimization_step = 0
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=5, active=10, repeat=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.environ['LOG_PATH']),
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
	) as prof:       
        for i in tqdm(range(0,config.bench_steps+config.bench_warmup)):
            if optimization_step >= config.bench_steps + config.bench_warmup:
                break
            elif optimization_step == config.bench_warmup:
                t0 = time.perf_counter()
            ##### train step
            loss = train_step(config, train_loader, model_engine)
            prof.step()
            optimization_step += 1
            model_engine.train()
    t = time.perf_counter()
    if args.world_rank == 0:
        print(f"Benchmarking completed in {t-t0} seconds.")
        print("Average Sample Throughput: " + 
            f"{config.micro_batch_size *config.world_size  * config.bench_steps / (t-t0)} Samples/Second")

