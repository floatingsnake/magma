#!/bin/bash

source ~/scratch/miniconda3/bin/activate
source ~/scripts/setup.sh
conda activate gpt-ds-jsrun

export NUM_NODES=1
export NGPU_PER_NODE=6
export NGPU=$((NGPU_PER_NODE*NUM_NODES))
export MASTER_PORT=12138
export MASTER_ADDR=$(cat $LSB_DJOB_HOSTFILE | sort | uniq | grep -v batch | grep -v login | head -1)

export OMP_NUM_THREADS=1
export WORLD_SIZE=${NGPU}

cat $LSB_DJOB_HOSTFILE | sort | uniq | tail -n +2 | sed -e 's/$/ slots=6/' > myhostfile 

jsrun -n ${NUM_NODES}  -a 6 -c 6 -g 6 --smpiargs="-disable_gpu_hooks"\
	python train_ds.py --deepspeed \
	 --config /ccs/home/lfsm/code/magma/configs/profile_bs16.yml

