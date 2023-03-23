import sys
sys.path.append('..')
import argparse
from magma.magma import Magma
from magma.webdataset import get_wds_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=False, help="path to your training config",
	default='/ccs/home/lfsm/code/magma/configs/benchmark_mbs1.yml')
    args = parser.parse_args()
    return args

args=parse_args() 
args.world_size=1
model = Magma(
    args.config,
)  # for finetuning one might want to load the model via Magma.from_checkpoint(...) here
import pdb;pdb.set_trace()
print('Hello,World')
