#!/bin/bash
#BSUB -P CSC499
#BSUB -W 1:00
#BSUB -nnodes 1
#BSUB -J magma
#BSUB -o magma.out.%J
#BSUB -e n2.out

export PATH=$HOME/.local/bin:$PATH
export CPATH=$HOME/.local/include:$CPATH
export LIBRARY_PATH=$HOME/.local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
export WANDB_DIR=/gpfs/alpine/scratch/lfsm/csc499/wandb
export WANDB_MODE=dryrun

source ~/scripts/setup.sh
source ~/scratch/miniconda3/bin/activate
conda activate gpt-neox-3.9 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/scratch/miniconda3/envs/gpt-neox-3.9/lib

export TORCH_EXTENSIONS_DIR=/gpfs/alpine/scratch/lfsm/csc499/mycache/torch_extensions/
export WORLD_SIZE=4
export OMP_NUM_THREADS=1
	
jsrun -n 1 -a 4 -c 4 -g 4 \
python -u my_train.py --deepspeed \
  --config /ccs/home/lfsm/code/magma/configs/profile_bs4.yml\
  --batch-size 1
#deepspeed --master_addr=localhost --master_port=8888 train_ds.py \
#  --config /ccs/home/lfsm/code/magma/configs/profile_bs4.yml

