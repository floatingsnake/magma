#!/bin/bash
export WANDB_DIR=/gpfs/alpine/scratch/lfsm/csc499/wandb
export WANDB_MODE=dryrun

source ~/scripts/setup.sh
source ~/scratch/miniconda3/bin/activate
conda activate gpt-neox-3.9 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/scratch/miniconda3/envs/gpt-neox-3.9/lib
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/scratch/lfsm/csc499/mycache/torch_extensions/
export OMP_NUM_THREADS=1
module load ninja

NNODE=4

export WORLD_SIZE=$(($NNODE*6))

jsrun -n 4 -a 6 -c 6 -g 6 \
python -u benchmark.py --deepspeed \
  --config /ccs/home/lfsm/code/magma/configs/benchmark_n4.yml

