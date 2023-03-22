#!/bin/bash
#BSUB -P CSC499
#BSUB -W 1:00
#BSUB -nnodes 2
#BSUB -J magma
#BSUB -o magma.out.%J
#BSUB -e n2.out

export PATH=$PATH:$HOME/.local/bin
export CPATH=$CPATH:$HOME/.local/include
export LIBRARY_PATH=$LIBRARY_PATH:$HOME/.local/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib

source ~/scripts/setup.sh
source ~/scratch/miniconda3/bin/activate
conda activate open-ce

export LD_LIBRARY_PATH=/lib64/:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/scratch/lfsm/csc499/mycache/torch_extensions/
export WANDB_DIR=/gpfs/alpine/scratch/lfsm/csc499/wandb

export OMP_NUM_THREADS=1

cat $LSB_DJOB_HOSTFILE | sort | uniq | grep -v batch | sed -e 's/$/ slots=6/' > myhostfile 
deepspeed --launcher jsrun --hostfile myhostfile \
  train_ds.py \
  --config /ccs/home/lfsm/code/magma/configs/profile_bs16.yml

