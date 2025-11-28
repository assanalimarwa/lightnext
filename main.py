import torch 
import torch.nn as nn 
from convnextblock import ConvNext


img = torch.rand(1, 96, 224, 224)

model = ConvNext(dim = 96)

out = model(img)

print(out)

