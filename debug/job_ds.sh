#!/bin/bash
#BSUB -P CSC499
#BSUB -W 1:00
#BSUB -nnodes 2
#BSUB -J magma
#BSUB -o magma.out.%J
#BSUB -e n2.out

source ~/scratch/miniconda4/bin/activate
conda activate gpt-ds-jsrun
source ~/scripts/setup.sh
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/scratch/lfsm/csc499/mycache/torch_extensions/
export WANDB_DIR=/gpfs/alpine/scratch/lfsm/csc499/wandb

export MASTER_PORT=12138
export MASTER_ADDR=$(cat $LSB_DJOB_HOSTFILE | sort | uniq | grep -v batch | grep -v login | head -1)

export OMP_NUM_THREADS=1

cat $LSB_DJOB_HOSTFILE | sort | uniq | tail -n +2 | sed -e 's/$/ slots=6/' > myhostfile 
deepspeed --launcher jsrun --hostfile myhostfile \
  --master_port=${MASTER_PORT} --master_addr=${MASTER_ADDR}\
  train_ds.py \
  --config /ccs/home/lfsm/code/magma/configs/profile_bs16.yml

