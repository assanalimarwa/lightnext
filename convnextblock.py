import torch
import torch.nn as nn 
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from layernorm import LayerNorm

class ConvNext(nn.Module):
    def __init__(self, dim, drop_path = 0., layer_scale_init_value= 1e-6):
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.ln = LayerNorm(dim, eps = 1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.gelu = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    

    def forward(self, x):

        input = x

        x = self.depthwise(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.ln(x)
        x = self.pwconv1(x)
        x = self.gelu(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x) 

        return x 



