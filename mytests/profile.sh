#!/bin/bash
#YBATCH -r a100_4
#SBATCH -N 1
#SBATCH -o /home/lfsm/code/result/logs/pf%j.out
#SBATCH --time=72:00:00
#SBATCH -J pf
#SBATCH --error /home/lfsm/code/result/logs/pf%j.err
source activate
conda deactivate
conda activate deepspeed
export LOG_PATH='/home/lfsm/code/magma/pf_1b_1gpus'
deepspeed --num_gpus 1 --master_port=8888 ./profile_model_train.py --config ../configs/benchmark_1b_s3.yml
export LOG_PATH='/home/lfsm/code/magma/pf_1b_2gpus'
deepspeed --num_gpus 2 --master_port=8888 ./profile_model_train.py --config ../configs/benchmark_1b_s3.yml
export LOG_PATH='/home/lfsm/code/magma/pf_1b_4gpus'
deepspeed --num_gpus 4 --master_port=8888 ./profile_model_train.py --config ../configs/benchmark_1b_s3.yml
