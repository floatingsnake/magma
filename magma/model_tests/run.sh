#!/bin/bash
source activate
conda activate /home/lfsm/anaconda3/envs/megatron
module load cuda/11.6
# export NCCL_IB_DISABLE=1
deepspeed --master_port 3333 --num_gpus 2 model_tests/test_magma_forward.py
# deepspeed --master_port 8888 --num_gpus 2 model_tests/test_model_forward.py
# deepspeed --master_port 8888 --num_gpus 1 ../mytests/test_model_train.py --config ../configs/summit_clipH_pythia70m_web.yml 