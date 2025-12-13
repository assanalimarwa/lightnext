import torch
import torch.nn as nn
import torch.nn.functional as F



class RepMedNeXtBlock(nn.Module):
    """MedNeXt block with reparameterizable depthwise conv"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        small_kernel: int = 3,
        do_res: bool = True,
        norm_type: str = 'group',
        grn: bool = False,
        deployed: bool = False
    ):
        super().__init__()
        
        self.do_res = do_res
        
        # Use ReparamBlock for depthwise conv
        self.conv1 = ReparamBlock(
            channels=in_channels,
            large_kernel=kernel_size,
            small_kernel=small_kernel,
            deployed=deployed
        )
        
        # Rest of MedNeXt block (same as before)
        if norm_type == 'group':
            self.norm = nn.GroupNorm(in_channels, in_channels)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(in_channels)
        
        self.conv2 = nn.Conv2d(in_channels, exp_r * in_channels, 1)
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(exp_r * in_channels, out_channels, 1)
        
        self.grn = grn
        if grn:
            self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1))
            self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1))
    
    def forward(self, x):
        identity = x
        
        # Reparameterizable depthwise conv
        x = self.conv1(x)
        
        # Standard MedNeXt operations
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)
        
        if self.grn:
            gx = torch.norm(x, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x = self.grn_gamma * (x * nx) + self.grn_beta + x
        
        x = self.conv3(x)
        
        if self.do_res:
            x = x + identity
        
        return x
    
    def merge_kernels(self):
        """Merge reparameterizable conv for deployment"""
        self.conv1.merge_kernels()


class ReparamBlock(nn.Module):
    """
    Simplified reparameterization block for medical imaging.
    Train with [7×7 + 3×3], deploy with merged [7×7].
    """
    def __init__(self, channels, large_kernel=7, small_kernel=3, deployed=False):
        super().__init__()
        
        self.channels = channels
        self.large_kernel = large_kernel
        self.small_kernel = small_kernel
        self.deployed = deployed
        
        if deployed:
            # Inference mode: single conv
            self.reparam_conv = nn.Conv2d(
                channels, 
                channels,
                kernel_size=large_kernel,
                padding=large_kernel // 2,
                groups=channels,
                bias=True
            )
        else:
            # Training mode: two branches
            self.large_conv = nn.Conv2d(
                channels, 
                channels,
                kernel_size=large_kernel,
                padding=large_kernel // 2,
                groups=channels,
                bias=False
            )
            self.large_bn = nn.BatchNorm2d(channels)
            
            self.small_conv = nn.Conv2d(
                channels, 
                channels,
                kernel_size=small_kernel,
                padding=small_kernel // 2,
                groups=channels,
                bias=False
            )
            self.small_bn = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        if self.deployed:
            # Inference: single convolution
            return self.reparam_conv(x)
        else:
            # Training: two branches
            large_out = self.large_bn(self.large_conv(x))
            small_out = self.small_bn(self.small_conv(x))
            return large_out + small_out
    
    def fuse_conv_bn(self, conv, bn):
        """Fuse convolution and batch norm into single conv."""
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        
        # Safety check
        if running_mean is None or running_var is None:
            raise RuntimeError(
                "BatchNorm running statistics not initialized. "
                "Run forward passes before calling merge_kernels()."
            )
        
        # Compute fused weights and bias
        std = torch.sqrt(running_var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)
        fused_weight = kernel * t
        fused_bias = beta - running_mean * gamma / std
        
        return fused_weight, fused_bias
    
    def merge_kernels(self):
        """Merge training branches into single inference conv."""
        if self.deployed:
            return  # Already merged
        
        # Set to eval mode to finalize BN statistics
        self.eval()
        
        # Fuse large conv + BN
        large_weight, large_bias = self.fuse_conv_bn(self.large_conv, self.large_bn)
        
        # Fuse small conv + BN
        small_weight, small_bias = self.fuse_conv_bn(self.small_conv, self.small_bn)
        
        # Pad small kernel to match large kernel size
        padding = (self.large_kernel - self.small_kernel) // 2
        small_weight_padded = F.pad(small_weight, [padding] * 4)
        
        # Merge by addition
        merged_weight = large_weight + small_weight_padded
        merged_bias = large_bias + small_bias
        
        # Create deployed conv
        self.reparam_conv = nn.Conv2d(
            self.channels, 
            self.channels,
            kernel_size=self.large_kernel,
            padding=self.large_kernel // 2,
            groups=self.channels,
            bias=True
        )
        
        # Set merged parameters
        self.reparam_conv.weight.data = merged_weight
        self.reparam_conv.bias.data = merged_bias
        
        # Delete training branches to save memory
        self.__delattr__('large_conv')
        self.__delattr__('large_bn')
        self.__delattr__('small_conv')
        self.__delattr__('small_bn')
        
        self.deployed = True
        
        print(f"✅ Reparameterization complete: {self.channels} channels, "
              f"{self.large_kernel}×{self.large_kernel} kernel")