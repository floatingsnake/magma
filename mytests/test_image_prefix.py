device = "cuda:0"

from torchsummary import summary
import torch
from magma.image_encoders import get_image_encoder

enc = get_image_encoder(
            'openclip-H',
            # device=self.device,
            pretrained=False,
            ).to(device)
print(enc)
input = torch.Tensor(4,3,224,224).to(device)
output = enc(input)
print(output.shape)

summary(enc,(4,3,224,224))
