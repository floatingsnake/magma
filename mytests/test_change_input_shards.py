import webdataset as wds

input_shards=r'/gpfs/alpine/csc499/proj-shared/LAION-400m-webdataset/data/{00000..41455}.tar'
urls = wds.shardlists.expand_urls(input_shards)
b = wds.SimpleShardList(urls)
import pdb; pdb.set_trace()
a = 0
print(urls)

