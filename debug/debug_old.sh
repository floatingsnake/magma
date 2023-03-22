#!/bin/bash
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

export OMP_NUM_THREADS=1
	
cat $LSB_DJOB_HOSTFILE | sort | uniq | grep -v batch | sed -e 's/$/ slots=6/' > myhostfile 

module load ninja

#jsrun -n 1 -a 6 -c 6 -g 6 \
jsrun -n 1 -a 4 -c 4 -g 4 \ 
python -u train_ds.py --deepspeed --config /ccs/home/lfsm/code/magma/configs/MAGMA_v1.yml

