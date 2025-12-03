import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from layernorm import LayerNorm
from convnext.convnextblock import ConvNextBlock





class ConvNeXtUNet(nn.Module):
    def __init__(self, in_chans=1, num_classes=4, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        
        # ============ ENCODER ============
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        # Stochastic depth decay rule
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Encoder Stage 1
        cur = 0
        self.enc_stage1 = nn.Sequential(
            *[ConvNextBlock(dim=dims[0], drop_path=dp_rates[cur + j], 
                           layer_scale_init_value=layer_scale_init_value) 
              for j in range(depths[0])]
        )
        
        # Downsample 1
        self.downsample1 = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)
        )
        
        # Encoder Stage 2
        cur += depths[0]
        self.enc_stage2 = nn.Sequential(
            *[ConvNextBlock(dim=dims[1], drop_path=dp_rates[cur + j], 
                           layer_scale_init_value=layer_scale_init_value) 
              for j in range(depths[1])]
        )
        
        # Downsample 2
        self.downsample2 = nn.Sequential(
            LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2)
        )
        
        # Encoder Stage 3
        cur += depths[1]
        self.enc_stage3 = nn.Sequential(
            *[ConvNextBlock(dim=dims[2], drop_path=dp_rates[cur + j], 
                           layer_scale_init_value=layer_scale_init_value) 
              for j in range(depths[2])]
        )
        
        # Downsample 3
        self.downsample3 = nn.Sequential(
            LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2)
        )
        
        # Bottleneck (Stage 4)
        cur += depths[2]
        self.bottleneck = nn.Sequential(
            *[ConvNextBlock(dim=dims[3], drop_path=dp_rates[cur + j], 
                           layer_scale_init_value=layer_scale_init_value) 
              for j in range(depths[3])]
        )
        
        # ============ DECODER ============
        # Upsample 3
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(dims[3], dims[2], kernel_size=2, stride=2),
            LayerNorm(dims[2], eps=1e-6, data_format="channels_first")
        )
        
        # Decoder Stage 3 (with skip connection)
        self.dec_stage3 = nn.Sequential(
            *[ConvNextBlock(dim=dims[2]*2, drop_path=0., 
                           layer_scale_init_value=layer_scale_init_value) 
              for j in range(depths[2])]
        )
        self.skip_conv3 = nn.Conv2d(dims[2]*2, dims[2], kernel_size=1)
        
        # Upsample 2
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(dims[2], dims[1], kernel_size=2, stride=2),
            LayerNorm(dims[1], eps=1e-6, data_format="channels_first")
        )
        
        # Decoder Stage 2 (with skip connection)
        self.dec_stage2 = nn.Sequential(
            *[ConvNextBlock(dim=dims[1]*2, drop_path=0., 
                           layer_scale_init_value=layer_scale_init_value) 
              for j in range(depths[1])]
        )
        self.skip_conv2 = nn.Conv2d(dims[1]*2, dims[1], kernel_size=1)
        
        # Upsample 1
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(dims[1], dims[0], kernel_size=2, stride=2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        # Decoder Stage 1 (with skip connection)
        self.dec_stage1 = nn.Sequential(
            *[ConvNextBlock(dim=dims[0]*2, drop_path=0., 
                           layer_scale_init_value=layer_scale_init_value) 
              for j in range(depths[0])]
        )
        self.skip_conv1 = nn.Conv2d(dims[0]*2, dims[0], kernel_size=1)
        
        # Final upsampling to original resolution
        self.final_upsample = nn.ConvTranspose2d(dims[0], dims[0], kernel_size=4, stride=4)
        
        # Segmentation head
        self.seg_head = nn.Conv2d(dims[0], num_classes, kernel_size=1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # ============ ENCODER ============
        x = self.stem(x)
        
        # Stage 1 (save for skip)
        skip1 = self.enc_stage1(x)
        x = self.downsample1(skip1)
        
        # Stage 2 (save for skip)
        skip2 = self.enc_stage2(x)
        x = self.downsample2(skip2)
        
        # Stage 3 (save for skip)
        skip3 = self.enc_stage3(x)
        x = self.downsample3(skip3)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # ============ DECODER ============
        # Upsample and concatenate with skip3
        x = self.upsample3(x)
        x = torch.cat([x, skip3], dim=1)  # Concatenate along channel dimension
        x = self.dec_stage3(x)
        x = self.skip_conv3(x)
        
        # Upsample and concatenate with skip2
        x = self.upsample2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec_stage2(x)
        x = self.skip_conv2(x)
        
        # Upsample and concatenate with skip1
        x = self.upsample1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec_stage1(x)
        x = self.skip_conv1(x)
        
        # Final upsampling to original resolution
        x = self.final_upsample(x)
        
        # Segmentation head
        x = self.seg_head(x)
        
        return x