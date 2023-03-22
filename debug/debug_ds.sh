#!/bin/bash
source ~/scratch/miniconda3/bin/activate
conda activate gpt-ds-jsrun
source ~/scripts/setup.sh
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/scratch/lfsm/csc499/mycache/torch_extensions/
export WANDB_DIR=/gpfs/alpine/scratch/lfsm/csc499/wandb

export OMP_NUM_THREADS=1

cat $LSB_DJOB_HOSTFILE | sort | uniq | tail -n +2 | sed -e 's/$/ slots=6/' > myhostfile 
deepspeed --launcher jsrun --hostfile myhostfile \
  train_ds.py \
  --config /ccs/home/lfsm/code/magma/configs/profile_bs16.yml

