from magma import Magma
from magma.image_input import ImageInput

model = Magma(
    config = "configs/summit_clipH_pythia19m.yml",
    device = 'cuda:0'
)

inputs =[
    ## supports urls and path/to/image
    ImageInput('https://www.art-prints-on-demand.com/kunst/thomas_cole/woods_hi.jpg'),
    'Describe the painting:'
]

## returns a tensor of shape: (1, 149, 4096)
embeddings = model.preprocess_inputs(inputs)  

## returns a list of length embeddings.shape[0] (batch size)
output = model.generate(
    embeddings = embeddings,
    max_steps = 6,
    temperature = 0.7,
    top_k = 0,
)  

print(output[0]) ##  A cabin on a lake
