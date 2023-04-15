#!/bin/bash
#YBATCH -r a100_8
#SBATCH -N 1
#SBATCH -o ./12b%j.out
#SBATCH --time=72:00:00
#SBATCH -J vit_t
#SBATCH --error ./12b%j.err

source activate
conda deactivate
conda activate megatron

deepspeed test_model_train.py --config ../configs/benchmark_12b_s3.yml
