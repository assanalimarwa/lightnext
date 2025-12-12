import torch
import torch.nn as nn
import torch.nn.functional as F




# class MedNeXtBlock(nn.Module):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  exp_r: int = 4,
#                  kernel_size: int = 7,
#                  do_res: bool = True,
#                  norm_type: str = 'group',
#                  n_groups: int or None = None,
#                  dim = '2d', 
#                  grn: bool = False):
#         super().__init__()
        
#         self.do_res = do_res
        
#         # Depthwise convolution (keeps channels same)
#         self.conv1 = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=in_channels,
#             kernel_size=kernel_size,
#             stride=1,
#             padding=kernel_size // 2,
#             groups=in_channels if n_groups is None else n_groups
#         )
        
#         # Normalization
#         if norm_type == 'group':
#             self.norm = nn.GroupNorm(
#                 num_groups=in_channels,
#                 num_channels=in_channels
#             )
#         elif norm_type == 'layer':
#             self.norm = nn.LayerNorm(in_channels)
        
#         # Expansion (1x1 conv)
#         self.conv2 = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=exp_r * in_channels,
#             kernel_size=1,
#             stride=1,
#             padding=0
#         )
        
#         # Activation
#         self.act = nn.GELU()
        
#         # Compression (1x1 conv)
#         self.conv3 = nn.Conv2d(
#             in_channels=exp_r * in_channels,
#             out_channels=out_channels,
#             kernel_size=1,
#             stride=1,
#             padding=0
#         )
        
#         # GRN parameters
#         self.grn = grn
#         if grn:
#             self.grn_beta = nn.Parameter(
#                 torch.zeros(1, exp_r * in_channels, 1, 1), 
#                 requires_grad=True
#             )
#             self.grn_gamma = nn.Parameter(
#                 torch.zeros(1, exp_r * in_channels, 1, 1), 
#                 requires_grad=True
#             )
    
#     def forward(self, x, dummy_tensor=None):
#         x1 = x
        
#         # Depthwise conv
#         x1 = self.conv1(x1)
        
#         # Norm → Expand → Activation
#         x1 = self.act(self.conv2(self.norm(x1)))
        
#         # GRN (if enabled)
#         if self.grn:
#             gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
#             nx = gx / (gx.mean(dim=1, keepdim=True)+1e-6)
#             x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        
#         # Compress
#         x1 = self.conv3(x1)
        
#         # Residual connection (if enabled)
#         if self.do_res:
#             x1 = x + x1
        
#         return x1


# class MedNeXtDownBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, 
#                  do_res=False, norm_type='group', dim='2d', grn=False):
#         super().__init__()  # ✓ Fixed: No parameters to nn.Module.__init__
        
#         self.do_res = do_res
#         self.resample_do_res = do_res
        
#         # 1. Depthwise convolution with stride=2 for downsampling
#         self.conv1 = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=in_channels,
#             kernel_size=kernel_size,
#             stride=2,
#             padding=kernel_size // 2,
#             groups=in_channels
#         )
        
#         # 2. Normalization
#         if norm_type == 'group':
#             self.norm = nn.GroupNorm(
#                 num_groups=in_channels,
#                 num_channels=in_channels
#             )
#         elif norm_type == 'layer':
#             self.norm = nn.LayerNorm(in_channels)
        
#         # 3. Expansion (1x1 conv)
#         self.conv2 = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=exp_r * in_channels,
#             kernel_size=1,
#             stride=1,
#             padding=0
#         )
        
#         # 4. Activation
#         self.act = nn.GELU()
        
#         # 5. Compression (1x1 conv)
#         self.conv3 = nn.Conv2d(
#             in_channels=exp_r * in_channels,
#             out_channels=out_channels,
#             kernel_size=1,
#             stride=1,
#             padding=0
#         )
        
#         # 6. GRN parameters (optional)
#         self.grn = grn
#         if grn:
#             self.grn_beta = nn.Parameter(
#                 torch.zeros(1, exp_r * in_channels, 1, 1), 
#                 requires_grad=True
#             )
#             self.grn_gamma = nn.Parameter(
#                 torch.zeros(1, exp_r * in_channels, 1, 1), 
#                 requires_grad=True
#             )
        
#         # 7. Residual convolution for downsampling path (optional)
#         if do_res:
#             self.res_conv = nn.Conv2d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=1,
#                 stride=2
#             )
    
#     def forward(self, x, dummy_tensor=None):
#         # Save input for residual
#         identity = x
        
#         # Main path
#         # 1. Depthwise conv (downsamples here)
#         x = self.conv1(x)
        
#         # 2. Norm → Expand → Activate
#         x = self.norm(x)
#         x = self.conv2(x)
#         x = self.act(x)
        
#         # 3. GRN (if enabled)
#         if self.grn:
#             gx = torch.norm(x, p=2, dim=(-2, -1), keepdim=True)
#             nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
#             x = self.grn_gamma * (x * nx) + self.grn_beta + x
        
#         # 4. Compress
#         x = self.conv3(x)
        
#         # 5. Add downsampled residual (if enabled)
#         if self.resample_do_res:
#             identity = self.res_conv(identity)
#             x = x + identity
        
#         return x


# class MedNeXtUpBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, 
#                  do_res=False, norm_type='group', dim='2d', grn=False):
#         super().__init__()  # ✓ Fixed: No parameters to nn.Module.__init__
        
#         self.do_res = do_res
#         self.resample_do_res = do_res
        
#         # 1. Depthwise TRANSPOSED convolution for UPSAMPLING
#         self.conv1 = nn.ConvTranspose2d(
#             in_channels=in_channels,
#             out_channels=in_channels,
#             kernel_size=kernel_size,
#             stride=2,
#             padding=kernel_size // 2,
#             groups=in_channels
#         )
        
#         # 2. Normalization
#         if norm_type == 'group':
#             self.norm = nn.GroupNorm(
#                 num_groups=in_channels,
#                 num_channels=in_channels
#             )
#         elif norm_type == 'layer':
#             self.norm = nn.LayerNorm(in_channels)
        
#         # 3. Expansion (1x1 conv)
#         self.conv2 = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=exp_r * in_channels,
#             kernel_size=1,
#             stride=1,
#             padding=0
#         )
        
#         # 4. Activation
#         self.act = nn.GELU()
        
#         # 5. Compression (1x1 conv)
#         self.conv3 = nn.Conv2d(
#             in_channels=exp_r * in_channels,
#             out_channels=out_channels,
#             kernel_size=1,
#             stride=1,
#             padding=0
#         )
        
#         # 6. GRN parameters (optional)
#         self.grn = grn
#         if grn:
#             self.grn_beta = nn.Parameter(
#                 torch.zeros(1, exp_r * in_channels, 1, 1), 
#                 requires_grad=True
#             )
#             self.grn_gamma = nn.Parameter(
#                 torch.zeros(1, exp_r * in_channels, 1, 1), 
#                 requires_grad=True
#             )
        
#         # 7. Residual TRANSPOSED convolution for upsampling path
#         if do_res:
#             self.res_conv = nn.ConvTranspose2d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=1,
#                 stride=2
#             )
    
#     def forward(self, x, dummy_tensor=None):
#         # Save original input for residual
#         identity = x
        
#         # Main path
#         # 1. Depthwise transposed conv (upsamples)
#         x = self.conv1(x)
        
#         # 2. Norm → Expand → Activate
#         x = self.norm(x)
#         x = self.conv2(x)
#         x = self.act(x)
        
#         # 3. GRN (if enabled)
#         if self.grn:
#             gx = torch.norm(x, p=2, dim=(-2, -1), keepdim=True)
#             nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
#             x = self.grn_gamma * (x * nx) + self.grn_beta + x
        
#         # 4. Compress
#         x = self.conv3(x)
        
#         # 5. Pad to fix asymmetry from transposed conv
#         x = F.pad(x, (1, 0, 1, 0))
        
#         # 6. Add upsampled residual (if enabled)
#         if self.resample_do_res:
#             res = self.res_conv(identity)
#             res = F.pad(res, (1, 0, 1, 0))
#             x = x + res
        
#         return x


# class OutBlock(nn.Module):
#     def __init__(self, in_channels, n_classes, dim='2d'):
#         super().__init__()
        
#         if dim == '2d':
#             conv = nn.Conv2d  # ✓ Fixed: Should be Conv2d, not ConvTranspose2d
#         elif dim == '3d':
#             conv = nn.Conv3d
#         self.conv_out = conv(in_channels, n_classes, kernel_size=1)
    
#     def forward(self, x, dummy_tensor=None): 
#         return self.conv_out(x)

# class LayerNorm(nn.Module):
#     """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
#     The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
#     shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
#     with shape (batch_size, channels, height, width).
#     """
#     def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))        # beta
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))         # gamma
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError 
#         self.normalized_shape = (normalized_shape, )
    
#     def forward(self, x, dummy_tensor=False):
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         elif self.data_format == "channels_first":
#             u = x.mean(1, keepdim=True)
#             s = (x - u).pow(2).mean(1, keepdim=True)
#             x = (x - u) / torch.sqrt(s + self.eps)
#             x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
#             return x





class MedNeXtBlock(nn.Module):

    def __init__(self, 
                in_channels:int, 
                out_channels:int, 
                exp_r:int=4, 
                kernel_size:int=7, 
                do_res:int=True,
                norm_type:str = 'group',
                n_groups:int or None = None,
                dim = '3d',
                grn = False
                ):

        super().__init__()

        self.do_res = do_res

        assert dim in ['2d', '3d']
        self.dim = dim
        if self.dim == '2d':
            conv = nn.Conv2d
        elif self.dim == '3d':
            conv = nn.Conv3d
            
        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 1,
            padding = kernel_size//2,
            groups = in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type=='group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels, 
                num_channels=in_channels
                )
        elif norm_type=='layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels, 
                data_format='channels_first'
                )

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv(
            in_channels = in_channels,
            out_channels = exp_r*in_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        
        # GeLU activations
        self.act = nn.GELU()
        
        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels = exp_r*in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

        self.grn = grn
        if grn:
            if dim == '3d':
                self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1,1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1,1), requires_grad=True)
            elif dim == '2d':
                self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)

 
    def forward(self, x, dummy_tensor=None):
        
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            if self.dim == '3d':
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            elif self.dim == '2d':
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True)+1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1  
        return x1


class MedNeXtDownBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, 
                do_res=False, norm_type = 'group', dim='3d', grn=False):

        super().__init__(in_channels, out_channels, exp_r, kernel_size, 
                        do_res = False, norm_type = norm_type, dim=dim,
                        grn=grn)

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2
            )

        self.conv1 = conv(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 2,
            padding = kernel_size//2,
            groups = in_channels,
        )

    def forward(self, x, dummy_tensor=None):
        
        x1 = super().forward(x)
        
        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class MedNeXtUpBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, 
                do_res=False, norm_type = 'group', dim='3d', grn = False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False, norm_type = norm_type, dim=dim,
                         grn=grn)

        self.resample_do_res = do_res
        
        self.dim = dim
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        if do_res:            
            self.res_conv = conv(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2
                )

        self.conv1 = conv(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 2,
            padding = kernel_size//2,
            groups = in_channels,
        )


    def forward(self, x, dummy_tensor=None):
        
        x1 = super().forward(x)
        # Asymmetry but necessary to match shape
        
        if self.dim == '2d':
            x1 = torch.nn.functional.pad(x1, (1,0,1,0))
        elif self.dim == '3d':
            x1 = torch.nn.functional.pad(x1, (1,0,1,0,1,0))
        
        if self.resample_do_res:
            res = self.res_conv(x)
            if self.dim == '2d':
                res = torch.nn.functional.pad(res, (1,0,1,0))
            elif self.dim == '3d':
                res = torch.nn.functional.pad(res, (1,0,1,0,1,0))
            x1 = x1 + res

        return x1


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