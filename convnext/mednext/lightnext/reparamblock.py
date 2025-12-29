import torch
import torch.nn as nn
import torch.nn.functional as F



class LargeKernelReparam(nn.Module):
    """
    Large Kernel Reparameterization (adapted from RepLKNet for PyTorch)
    
    Training: Multiple branches (7×7 + 3×3 + 5×5)
    Inference: Single merged conv
    """
    
    def __init__(self, 
                 channels: int,
                 kernel_size: int = 7,
                 small_kernels: tuple = (5, 3),
                 stride: int = 1,
                 groups: int = None,
                 deploy: bool = False):
        """
        Args:
            channels: Number of channels (in = out for depthwise)
            kernel_size: Size of large kernel (e.g., 7)
            small_kernels: Tuple of small kernel sizes (e.g., (5, 3))
            stride: Convolution stride
            groups: Number of groups (channels for depthwise)
            deploy: If True, use single merged conv
        """
        super().__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.small_kernels = small_kernels
        self.stride = stride
        self.groups = groups if groups is not None else channels  # Depthwise by default
        self.deploy = deploy
        
        if deploy:
            # ============================================
            # INFERENCE MODE: Single convolution
            # ============================================
            self.dw_reparam = nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=self.groups,
                bias=True
            )
        else:
            # ============================================
            # TRAINING MODE: Multiple branches
            # ============================================
            
            # Large kernel branch
            self.dw_large = nn.Sequential(
                nn.Conv2d(
                    channels, channels, kernel_size,
                    stride=stride, padding=kernel_size // 2,
                    groups=self.groups, bias=False
                ),
                nn.BatchNorm2d(channels)
            )
            
            # Small kernel branches
            for k in small_kernels:
                branch = nn.Sequential(
                    nn.Conv2d(
                        channels, channels, k,
                        stride=stride, padding=k // 2,
                        groups=self.groups, bias=False
                    ),
                    nn.BatchNorm2d(channels)
                )
                setattr(self, f'dw_small_{k}', branch)
    
    def forward(self, x):
        if self.deploy:
            # Inference: single conv
            return self.dw_reparam(x)
        
        # Training: sum all branches
        out = self.dw_large(x)
        
        for k in self.small_kernels:
            out += getattr(self, f'dw_small_{k}')(x)
        
        return out
    
    def _fuse_conv_bn(self, conv, bn):
        """Fuse Conv2d + BatchNorm2d into single Conv2d"""
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        
        std = torch.sqrt(running_var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)
        
        fused_kernel = kernel * t
        fused_bias = beta - running_mean * gamma / std
        
        return fused_kernel, fused_bias
    
    def merge_kernels(self):
        """Merge all branches into single kernel"""
        if self.deploy:
            return
        
        # Fuse large branch (conv + bn)
        large_kernel, large_bias = self._fuse_conv_bn(
            self.dw_large[0], self.dw_large[1]
        )
        
        # Start with large kernel
        merged_kernel = large_kernel
        merged_bias = large_bias
        
        # Add small kernels (padded to large kernel size)
        for k in self.small_kernels:
            branch = getattr(self, f'dw_small_{k}')
            small_kernel, small_bias = self._fuse_conv_bn(branch[0], branch[1])
            
            # Pad small kernel to match large kernel size
            pad_size = (self.kernel_size - k) // 2
            small_kernel_padded = F.pad(
                small_kernel,
                [pad_size, pad_size, pad_size, pad_size]
            )
            
            merged_kernel += small_kernel_padded
            merged_bias += small_bias
        
        # Create deployed conv
        self.dw_reparam = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size // 2,
            groups=self.groups,
            bias=True
        )
        
        self.dw_reparam.weight.data = merged_kernel
        self.dw_reparam.bias.data = merged_bias
        
        # Delete training branches
        self.__delattr__('dw_large')
        for k in self.small_kernels:
            self.__delattr__(f'dw_small_{k}')
        
        self.deploy = True


def merge_all_large_kernel_reparam(model):
    """Merge all LargeKernelReparam in model"""
    for module in model.modules():
        if isinstance(module, LargeKernelReparam):
            module.merge_kernels()




class RepMedNeXtBlock(nn.Module):
    """
    MedNeXt Block with Large Kernel Reparameterization
    
    Changes from original:
    - conv1: Regular Conv2d → LargeKernelReparam
    - Training: Uses multiple kernel branches
    - Inference: Single merged kernel (faster)
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_r: int = 4,
                 kernel_size: int = 7,
                 small_kernels: tuple = (5, 3),  # ← NEW parameter
                 do_res: bool = True,
                 norm_type: str = 'group',
                 n_groups: int = None,
                 dim: str = '2d',
                 grn: bool = False,
                 deploy: bool = False):  # ← NEW parameter
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            exp_r: Expansion ratio
            kernel_size: Large kernel size (e.g., 7)
            small_kernels: Small kernel sizes for rep branches (e.g., (5, 3))
            do_res: Use residual connection
            norm_type: 'group' or 'layer'
            n_groups: Groups for GroupNorm
            dim: '2d' or '3d' (currently only 2d supported)
            grn: Use Global Response Normalization
            deploy: Deployment mode (merged kernels)
        """
        super().__init__()
        
        self.do_res = do_res
        self.dim = dim
        self.deploy = deploy
        
        assert dim == '2d', "RepMedNeXt currently only supports 2D"
        
        # ============================================
        # 1. REPARAMETERIZABLE DEPTHWISE CONVOLUTION
        # ============================================
        self.conv1 = LargeKernelReparam(
            channels=in_channels,
            kernel_size=kernel_size,
            small_kernels=small_kernels,
            stride=1,
            groups=in_channels if n_groups is None else n_groups,
            deploy=deploy
        )
        
        # ============================================
        # 2. NORMALIZATION (unchanged)
        # ============================================
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels if n_groups is None else n_groups,
                num_channels=in_channels
            )
        elif norm_type == 'layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels,
                data_format='channels_first'
            )
        
        # ============================================
        # 3. EXPANSION (unchanged)
        # ============================================
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # ============================================
        # 4. ACTIVATION (unchanged)
        # ============================================
        self.act = nn.GELU()
        
        # ============================================
        # 5. COMPRESSION (unchanged)
        # ============================================
        self.conv3 = nn.Conv2d(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # ============================================
        # 6. GRN (unchanged)
        # ============================================
        self.grn = grn
        if grn:
            self.grn_beta = nn.Parameter(
                torch.zeros(1, exp_r * in_channels, 1, 1),
                requires_grad=True
            )
            self.grn_gamma = nn.Parameter(
                torch.zeros(1, exp_r * in_channels, 1, 1),
                requires_grad=True
            )
    
    def forward(self, x, dummy_tensor=None):
        """Forward pass (same as original)"""
        
        x1 = x
        
        # Reparameterizable depthwise conv
        x1 = self.conv1(x1)
        
        # Rest is unchanged
        x1 = self.norm(x1)
        x1 = self.conv2(x1)
        x1 = self.act(x1)
        
        # GRN (if enabled)
        if self.grn:
            gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        
        x1 = self.conv3(x1)
        
        # Residual connection
        if self.do_res:
            x1 = x + x1
        
        return x1
    
    def merge_kernels(self):
        """Merge reparameterizable conv"""
        if hasattr(self.conv1, 'merge_kernels'):
            self.conv1.merge_kernels()
        self.deploy = True





class MedNeXtDownBlock(RepMedNeXtBlock):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, 
                 do_res=False, norm_type='group', dim='2d', grn=False):
        
        # Call parent with do_res=False (original does this)
        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                        do_res=False, norm_type=norm_type, dim=dim,
                        grn=grn)
        
        self.resample_do_res = do_res
        self.dim = dim
        
        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
            
        # Residual convolution for downsampling path
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )
        
        # Override conv1 with stride=2 for downsampling
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )
    
    def forward(self, x, dummy_tensor=None):
        # Call parent's forward (which uses the overridden conv1 with stride=2)
        x1 = super().forward(x)
        
        # Add residual connection if enabled
        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res
        
        return x1


class MedNeXtUpBlock(RepMedNeXtBlock):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, 
                 do_res=False, norm_type='group', dim='2d', grn=False):
        
        # Call parent with do_res=False
        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False, norm_type=norm_type, dim=dim,
                         grn=grn)
        
        self.resample_do_res = do_res
        self.dim = dim
        
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
            
        # Residual transposed convolution
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )
        
        # Override conv1 with transposed convolution for upsampling
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )
    
    def forward(self, x, dummy_tensor=None):
        # Call parent's forward (which uses the overridden conv1 with transposed conv)
        x1 = super().forward(x)
        
        # Padding to fix asymmetry (as in original)
        if self.dim == '2d':
            x1 = F.pad(x1, (1, 0, 1, 0))
        elif self.dim == '3d':
            x1 = F.pad(x1, (1, 0, 1, 0, 1, 0))
        
        # Add residual connection if enabled
        if self.resample_do_res:
            res = self.res_conv(x)
            if self.dim == '2d':
                res = F.pad(res, (1, 0, 1, 0))
            elif self.dim == '3d':
                res = F.pad(res, (1, 0, 1, 0, 1, 0))
            x1 = x1 + res
        
        return x1


class OutBlock(nn.Module):
    def __init__(self, in_channels, n_classes, dim='2d'):
        super().__init__()
        
        if dim == '2d':
            conv = nn.ConvTranspose2d  # NOTE: Changed to ConvTranspose to match original!
        elif dim == '3d':
            conv = nn.ConvTranspose3d  # NOTE: Changed to ConvTranspose to match original!
        
        self.conv_out = conv(in_channels, n_classes, kernel_size=1)
    
    def forward(self, x, dummy_tensor=None): 
        return self.conv_out(x)


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. """
    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x, dummy_tensor=None):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            
            # Handle both 2D and 3D cases
            if x.dim() == 4:  # 2D: [B, C, H, W]
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            elif x.dim() == 5:  # 3D: [B, C, D, H, W]
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            else:
                raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
            return x