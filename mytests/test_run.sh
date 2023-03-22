#!/bin/bash

deepspeed --launcher jsrun --hostfile myhostfile \
	train_ds.py \
	--config configs/profile_bs16.yml > n2_oom.txt
