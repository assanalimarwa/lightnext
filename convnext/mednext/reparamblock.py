import torch
import torch.nn as nn
import torch.nn.functional as F


class ReparamMedNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, exp_r, kernel_size, do_res, norm_type, n_groups, 
                 dim, grn):
        super().__init__()


        self.conv1 = nn.Conv2d(in_channels= in_channels, 
                              out_channels= out_channels,
                              kernel_size=kernel_size, 
                              stride = 1, 
                              padding=kernel_size //2, 
                              groups= in_channels if n_groups is None else n_groups) 
        

        # Normalization
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels
            )
        elif norm_type == 'layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels,
                data_format='channels_first'
            )
        
        # Expansion (1x1 conv)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,  # ✓ Expand from in_channels
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # Activation
        self.act = nn.GELU()
        
        # Compression (1x1 conv)
        self.conv3 = nn.Conv2d(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # GRN parameters
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
    
    def forward(self, x, dummy_tensor = None):


        
        x1 = x
        
        # Depthwise conv
        x1 = self.conv1(x1)
        
        # Norm → Expand → Activation
        x1 = self.act(self.conv2(self.norm(x1)))
        
        # GRN (if enabled)
        if self.grn:
            gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        
        # Compress
        x1 = self.conv3(x1)
        
        # Residual connection (if enabled)
        if self.do_res:
            x1 = x + x1
        
        return x1
        





class MedNeXtDownBlock(nn.Module):  # Inherit from nn.Module, not nothing!
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, 
                 do_res=False, norm_type='group', dim ='2d', grn=False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size, do_res= False, norm_type = norm_type, dim = dim)
        
        self.do_res = do_res  # This is for the MAIN residual connection
        self.resample_do_res = do_res  # For downsampled residual
        
        # 1. Depthwise convolution with stride=2 for downsampling
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,  # ✓ Keep same (depthwise)
            kernel_size=kernel_size,
            stride=2,  # ✓ Downsampling
            padding=kernel_size // 2,
            groups=in_channels
        )
        
        # 2. Normalization
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels
            )
        elif norm_type == 'layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels,
                data_format='channels_first'
            )
        
        # 3. Expansion (1x1 conv)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # 4. Activation
        self.act = nn.GELU()
        
        # 5. Compression (1x1 conv)
        self.conv3 = nn.Conv2d(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # 6. GRN parameters (optional)
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
        
        # 7. Residual convolution for downsampling path (optional)
        if do_res:
            self.res_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )
    
    def forward(self, x, dummy_tensor = None):
        # Save input for residual
        identity = x
        
        # Main path
        # 1. Depthwise conv (downsamples here)
        x = self.conv1(x)
        
        # 2. Norm → Expand → Activate
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)
        
        # 3. GRN (if enabled)
        if self.grn:
            gx = torch.norm(x, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x = self.grn_gamma * (x * nx) + self.grn_beta + x
        
        # 4. Compress
        x = self.conv3(x)
        
        # 5. Add downsampled residual (if enabled)
        if self.resample_do_res:
            identity = self.res_conv(identity)
            x = x + identity
        
        return x


class MedNeXtUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, 
                 do_res=False, norm_type='group', dim = '2d', grn=False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size, do_res = False, norm_type = norm_type, dim = dim, grn = grn)
        
        self.do_res = do_res
        self.resample_do_res = do_res
        
        # 1. Depthwise TRANSPOSED convolution for UPSAMPLING
        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,  # ✓ Upsamples by 2x
            padding=kernel_size // 2,
            groups=in_channels
        )
        
        # 2. Normalization
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels
            )
        elif norm_type == 'layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels,
                data_format='channels_first'
            )
        
        # 3. Expansion (1x1 conv)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # 4. Activation
        self.act = nn.GELU()
        
        # 5. Compression (1x1 conv)
        self.conv3 = nn.Conv2d(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # 6. GRN parameters (optional)
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
        
        # 7. Residual TRANSPOSED convolution for upsampling path
        if do_res:
            self.res_conv = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2  # ✓ Upsamples by 2x
            )
    
    def forward(self, x, dummy_tensor = None):
        # Save original input for residual
        identity = x
        
        # Main path
        # 1. Depthwise transposed conv (upsamples)
        x = self.conv1(x)
        
        # 2. Norm → Expand → Activate
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)
        
        # 3. GRN (if enabled)
        if self.grn:
            gx = torch.norm(x, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x = self.grn_gamma * (x * nx) + self.grn_beta + x
        
        # 4. Compress
        x = self.conv3(x)
        
        # 5. Pad to fix asymmetry from transposed conv
        x = F.pad(x, (1, 0, 1, 0))  # Pad right and bottom by 1
        
        # 6. Add upsampled residual (if enabled)
        if self.resample_do_res:
            res = self.res_conv(identity)  # ✓ Use ORIGINAL input
            res = F.pad(res, (1, 0, 1, 0))
            x = x + res
        
        return x


class OutBlock(nn.Module):

    def __init__(self, in_channels, n_classes, dim):
        super().__init__()
        
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        self.conv_out = conv(in_channels, n_classes, kernel_size=1)
    
    def forward(self, x, dummy_tensor=None): 
        return self.conv_out(x)


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))        # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape))         # gamma
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x, dummy_tensor=False):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x