import sys
sys.path.append('..')
import argparse
from magma.magma import Magma
from magma.webdataset import get_wds_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=False, help="path to your training config",
	default='/ccs/home/lfsm/code/magma/configs/summit_clipH_pythia70m_web.yml')
    parser.add_argument(
        "--val-data",
        type=str,
        default="/gpfs/alpine/csc499/proj-shared/LAION-400m-webdataset/data/{41400..41401}.tar",
        help="Path to csv filewith training data",
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        #default=407332084,
        default=1084,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--micro-batch-size", type=int, default=128, help="Batch size per GPU."
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
		        text,
                return_tensors="pt",
                max_length=2048,
                padding="max_length",
                truncation=True,)
data = get_wds_dataset(args, preprocess_img=transforms, preprocess_text=preprocess_text, is_train=False)
data.set_epoch(0)
dataloader = data.dataloader
i=0
print('start get data')
for batch in dataloader:
    i += 1
    print(i)
    print(batch[0].shape)
    print(batch[1].shape)
    if i % 100==0:
        print(f"sample {i} times done")
print(f'total sample number is {i}')


def cycle(loader):
    while True:
        for data in loader:
            yield data

dataloader = cycle(dataloader)
print('start over one epoch simmulation')
for i in range(407332084):
    batch = next(dataloader)
    i += 1
    print(i)
    if i % 1000000==0:
        print(f"sample {i} times done")
print(f'total sample number is {i}')




