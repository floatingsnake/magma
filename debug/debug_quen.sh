#/bin/bash

source /gpfs/alpine/csc499/proj-shared/env_setup/setup.sh
cat $LSB_DJOB_HOSTFILE | sort | uniq | tail -n +2 | sed -e 's/$/ slots=6/' > myhostfile
deepspeed --launcher jsrun --hostfile myhostfile \
  train_ds.py \
  --config /ccs/home/lfsm/code/magma/configs/profile_bs16.yml
