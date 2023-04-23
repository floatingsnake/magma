import webdataset as wds
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

def get_wds_dataset(args, preprocess_img, preprocess_text, is_train, epoch=0, floor=False):
    input_shards = args.train_data if is_train else args.val_data
