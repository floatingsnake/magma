import sys
sys.path.append('..')
import torch
import os
import deepspeed
from magma import Magma
from magma.image_input import ImageInput

model = Magma(
    config = "../configs/benchmark_mbs1.yml",
    device = 'cuda:0'
)

model = model.to('cuda:0')

inputs =[
    ## supports urls and path/to/image
    ImageInput('https://www.art-prints-on-demand.com/kunst/thomas_cole/woods_hi.jpg'),
    # 'Describe the painting:'
]
import pdb;pdb.set_trace()
## returns a tensor of shape: (1, 149, 4096)
embeddings = model.preprocess_inputs(inputs).to('cuda:0')  

## returns a list of length embeddings.shape[0] (batch size)
output = model.generate(
    embeddings = embeddings,
    max_steps = 6,
    temperature = 0.7,
    top_k = 0,
)  

print(output[0]) ##  A cabin on a lake
