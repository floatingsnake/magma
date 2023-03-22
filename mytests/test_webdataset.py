import sys
sys.path.append('..')
import argparse
from magma.magma import Magma
from magma.webdataset import get_wds_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=False, help="path to your training config",
	default='/ccs/home/lfsm/code/magma/configs/benchmark_n2.yml')
    parser.add_argument(
        "--train-data",
        type=str,
        default="/gpfs/alpine/csc499/proj-shared/LAION-400m-webdataset/data/{00000..41455}.tar",
        help="Path to csv filewith training data",
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=407332084,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of dataloader workers per GPU."
    )
    args = parser.parse_args()
    return args

args=parse_args() 
args.world_size=1
model = Magma(
    args.config,
)  # for finetuning one might want to load the model via Magma.from_checkpoint(...) here

tokenizer, config, transforms = model.tokenizer, model.config, model.transforms
def preprocess_text(text):
    return tokenizer.encode(
		captions,
                return_tensors="pt",
                max_length=2048,
                padding="max_length",
                truncation=True,)
data = get_wds_dataset(args, transforms, preprocess_text, is_train=True)
data.set_epoch(0)
dataloader = data.dataloader
i=0
for batch in dataloader:
    i += 1
    import pdb;pdb.set_trace()
    print(batch[0].shape)
    if i == 1000:
        print("sample 1000 times done")
        break

