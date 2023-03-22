import sys
sys.path.append('..')
import argparse
from magma.magma import Magma
from magma.webdataset import get_wds_dataset
from magma.utils import parse_args


args=parse_args() 
args.world_size=1
model = Magma(
    args.config,
)  # for finetuning one might want to load the model via Magma.from_checkpoint(...) here

tokenizer, config, transforms = model.tokenizer, model.config, model.transforms
