import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from layernorm import LayerNorm
from convnext.convnextblock import ConvNextBlock


class MedNeXtBlock(nn.Module):
    """Base MedNeXt block"""
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, 
                 do_res=True, norm_type='group', dim='2d', grn=False):
        super().__init__()
        
        self.do_res = do_res
        self.dim = dim
        
        # Convolution type based on dimension
        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
            
        # Normalization
        if norm_type == 'group':
            self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        elif norm_type == 'layer':
            self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        
        # Depthwise convolution
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            groups=in_channels
        )
        
        # Expansion
        self.conv2 = conv(in_channels, exp_r * in_channels, kernel_size=1)
        self.act = nn.GELU()
        
        # Contraction
        self.conv3 = conv(exp_r * in_channels, out_channels, kernel_size=1)
        
        # Residual connection
        if do_res:
            if in_channels != out_channels:
                self.res_conv = conv(in_channels, out_channels, kernel_size=1)
            else:
                self.res_conv = nn.Identity()
    
    def forward(self, x):
        x1 = self.norm(x)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.act(x1)
        x1 = self.conv3(x1)
        
        if self.do_res:
            res = self.res_conv(x)
            x1 = x1 + res
            
        return x1


class MedNeXtDownBlock(MedNeXtBlock):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, 
                 do_res=False, norm_type='group', dim='2d', grn=False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size, 
                        do_res=False, norm_type=norm_type, dim=dim, grn=grn)
        
        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
            
        self.resample_do_res = do_res
        
        if do_res:
            self.res_conv_downsample = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )
        
        # Downsampling convolution
        self.downsample = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size//2,
            groups=in_channels
        )
    
    def forward(self, x):
        # Save input for residual (before downsampling)
        x_input = x
        
        # Downsample
        x = self.downsample(x)
        
        # Apply MedNeXt block processing
        x1 = super().forward(x)
        
        # Add residual connection if enabled
        if self.resample_do_res:
            res = self.res_conv_downsample(x_input)  # Downsample the original input
            x1 = x1 + res
        
        return x1


class MedNeXtUpBlock(MedNeXtBlock):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, 
                 do_res=False, norm_type='group', dim='2d', grn=False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False, norm_type=norm_type, dim=dim, grn=grn)
        
        self.resample_do_res = do_res
        self.dim = dim
        
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
            
        if do_res:
            self.res_conv_upsample = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )
        
        # Upsampling convolution
        self.upsample = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size//2,
            groups=in_channels,
            output_padding=1  # Important for matching dimensions
        )
    
    def forward(self, x):
        # Save input for residual (before upsampling)
        x_input = x
        
        # Upsample
        x = self.upsample(x)
        
        # Apply MedNeXt block processing
        x1 = super().forward(x)
        
        # Add residual connection if enabled
        if self.resample_do_res:
            res = self.res_conv_upsample(x_input)  # Upsample the original input
            
            # Match dimensions if needed
            if x1.shape != res.shape:
                if self.dim == '2d':
                    # Pad to match if there's still a size mismatch
                    diff_h = x1.shape[2] - res.shape[2]
                    diff_w = x1.shape[3] - res.shape[3]
                    res = torch.nn.functional.pad(res, (0, diff_w, 0, diff_h))
            
            x1 = x1 + res
        
        return x1


class ConvNextBlock(nn.Module):
    """ConvNeXt Block"""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class LightNextv1(nn.Module):
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
        
        # Downsample 1 - NOW WITH PROPER PARAMETERS
        self.downsample1 = MedNeXtDownBlock(
            in_channels=dims[0], 
            out_channels=dims[1], 
            exp_r=2, 
            kernel_size=3, 
            do_res=True,
            norm_type='group',
            dim='2d'
        )
        
        # Encoder Stage 2
        cur += depths[0]
        self.enc_stage2 = nn.Sequential(
            *[ConvNextBlock(dim=dims[1], drop_path=dp_rates[cur + j], 
                           layer_scale_init_value=layer_scale_init_value) 
              for j in range(depths[1])]
        )
        
        # Downsample 2
        self.downsample2 = MedNeXtDownBlock(
            in_channels=dims[1], 
            out_channels=dims[2], 
            exp_r=2, 
            kernel_size=3, 
            do_res=True,
            norm_type='group',
            dim='2d'
        )
        
        # Encoder Stage 3
        cur += depths[1]
        self.enc_stage3 = nn.Sequential(
            *[ConvNextBlock(dim=dims[2], drop_path=dp_rates[cur + j], 
                           layer_scale_init_value=layer_scale_init_value) 
              for j in range(depths[2])]
        )
        
        # Downsample 3
        self.downsample3 = MedNeXtDownBlock(
            in_channels=dims[2], 
            out_channels=dims[3], 
            exp_r=2, 
            kernel_size=3, 
            do_res=True,
            norm_type='group',
            dim='2d'
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
        self.upsample3 = MedNeXtUpBlock(
            in_channels=dims[3], 
            out_channels=dims[2], 
            exp_r=2, 
            kernel_size=3, 
            do_res=True,
            norm_type='group',
            dim='2d'
        )
        
        # Decoder Stage 3 (with skip connection)
        self.dec_stage3 = nn.Sequential(
            *[ConvNextBlock(dim=dims[2]*2, drop_path=0., 
                           layer_scale_init_value=layer_scale_init_value) 
              for j in range(depths[2])]
        )
        self.skip_conv3 = nn.Conv2d(dims[2]*2, dims[2], kernel_size=1)
        
        # Upsample 2
        self.upsample2 = MedNeXtUpBlock(
            in_channels=dims[2], 
            out_channels=dims[1], 
            exp_r=2, 
            kernel_size=3, 
            do_res=True,
            norm_type='group',
            dim='2d'
        )
        
        # Decoder Stage 2 (with skip connection)
        self.dec_stage2 = nn.Sequential(
            *[ConvNextBlock(dim=dims[1]*2, drop_path=0., 
                           layer_scale_init_value=layer_scale_init_value) 
              for j in range(depths[1])]
        )
        self.skip_conv2 = nn.Conv2d(dims[1]*2, dims[1], kernel_size=1)
        
        # Upsample 1
        self.upsample1 = MedNeXtUpBlock(
            in_channels=dims[1], 
            out_channels=dims[0], 
            exp_r=2, 
            kernel_size=3, 
            do_res=True,
            norm_type='group',
            dim='2d'
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
        x = torch.cat([x, skip3], dim=1)
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