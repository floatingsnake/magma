#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
export CPATH=$HOME/.local/include:$CPATH
export LIBRARY_PATH=$HOME/.local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

source ~/scripts/setup.sh
source ~/scratch/miniconda3/bin/activate
conda activate open-ce

export TORCH_EXTENSIONS_DIR=/gpfs/alpine/scratch/lfsm/csc499/mycache/torch_extensions/
export WANDB_DIR=/gpfs/alpine/scratch/lfsm/csc499/wandb

export OMP_NUM_THREADS=1

cat $LSB_DJOB_HOSTFILE | sort | uniq | grep -v batch | sed -e 's/$/ slots=6/' > myhostfile 
deepspeed --launcher jsrun --hostfile myhostfile \
  train_ds.py \
  --config /ccs/home/lfsm/code/magma/configs/profile_bs4.yml

