#!/bin/bash

# compute node don't have write permisson and ability to connect internet
export WANDB_DIR=/gpfs/alpine/scratch/lfsm/csc499/wandb
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/scratch/lfsm/csc499/mycache/torch_extensions/
export WANDB_MODE=dryrun

# set up module and other env variable
# this env and shell are built following 
# https://docs.google.com/document/d/1gyMVFonqgTRToaHckEPYjGTX-06cZcO04VoT-GHoXAc/edit#heading=h.zh8lrk889uso
source ~/scripts/setup.sh
source ~/scratch/miniconda3/bin/activate
conda activate gpt-neox-3.9 

# first do
# ln -s /minoconda3/lib/libstdc++.so.6 /current_env_path/lib
# this is to ensure system can use new libstdc++.so.6 
# while don't use incompatiable openssl in /minoconda3/lib/ 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/scratch/miniconda3/envs/gpt-neox-3.9/lib

NNODE=2
export OMP_NUM_THREADS=1
#export WORLD_SIZE=$(($NNODE*6))
jsrun -n 12 -a 1 -c 1 -g 1 \
python -u benchmark.py --deepspeed \
  --config /ccs/home/lfsm/code/magma/configs/benchmark_mbs2.yml

