import torch 
import torch.nn as nn 
from convnextblock import ConvNext


img = torch.rand(1, 3, 224, 224)

out = ConvNext(img)

print(out)

