import torch 
import torch.nn as nn 
from convnext.convnextblock import ConvNextBlock
from convnext.convnext import ConvNeXtModel



img = torch.rand(1, 96, 224, 224)

model = ConvNextBlock(dim = 96)

out = model(img)

print(out.shape)

def convnext_tiny(**kwargs):
    model = ConvNeXtModel(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

model = convnext_tiny()

print(model)







