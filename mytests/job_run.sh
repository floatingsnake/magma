#!/bin/bash
#BSUB -P CSC499
#BSUB -W 1:00
#BSUB -nnodes 32 
#BSUB -J 20b
#BSUB -e 20b.out.%J
#BSUB -o n32.out

# compute node don't have write permisson and ability to connect internet
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/scratch/lfsm/csc499/mycache/torch_extensions/

# set up module and other env variable
# this env and shell are built following 
# https://docs.google.com/document/d/1gyMVFonqgTRToaHckEPYjGTX-06cZcO04VoT-GHoXAc/edit#heading=h.zh8lrk889uso
source ~/scripts/setup.sh
source ~/scratch/miniconda3/bin/activate
conda activate gpt-neox-3.9 

# first do
# ln -s /minoconda3/lib/libstdc++.so.6 /current_env_path/lib
# this is to ensure system can use new libstdc++.so.6 
# while don't use incompatiable openssl in /minoconda3/lib/ 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/scratch/miniconda3/envs/gpt-neox-3.9/lib

jsrun -n 32 -a 6 -c 6 -g 6 \
python test_model.py --deepspeed --config ../configs/benchmark_20b_mbs1.yml
#deepspeed --launcher jsrun --hostfile myhostfile \
#	train_ds.py \
#	--config configs/profile_bs16.yml > n2_oom.txt
