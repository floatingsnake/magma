#!/bin/bash

export TORCH_EXTENSIONS_DIR=/gpfs/alpine/scratch/lfsm/csc499/mycache/torch_extensions/

#jsrun -n 1 -a 6 -c 6 -g 6 \
#	python test_device_num.py
#jsrun -n 6 -a 3 -c 3 -g 3 \
#	python test_device_num.py


jsrun -n 8 -a 6 -c 6 -g 6 \
python test_model --deepspeed --config ../configs/benchmark_20b_mbs1.yml
#deepspeed --launcher jsrun --hostfile myhostfile \
#	train_ds.py \
#	--config configs/profile_bs16.yml > n2_oom.txt
