#!/bin/bash

jsrun -n 2 -a 6 -c 6 -g 6 \
	python test_model.py

#deepspeed --launcher jsrun --hostfile myhostfile \
#	train_ds.py \
#	--config configs/profile_bs16.yml > n2_oom.txt
