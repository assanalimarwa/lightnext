import torch
import sys


import torch.nn as nn


sys.path.append('/path/to/original')
sys.path.append('/path/to/new')

from mednext.mednextorig.mednextorig import MedNeXt as MedNeXtOrig
from mednext.newmednext.mednextnew import MedNeXtNew as MedNeXtNew

# Create models with same parameters
orig_model = MedNeXtOrig(
    in_channels=3,
    n_channels=32,
    n_classes=2,
    exp_r=4,
    kernel_size=3,
    deep_supervision=False,
    do_res=True,
    do_res_up_down=True,
    checkpoint_style=None,
    block_counts=[1,1,1,1,1,1,1,1,1],  # Reduced for testing
    norm_type='group',
    dim='2d',
    grn=False
)

new_model = MedNeXtNew(
    in_channels=3,
    n_channels=32,
    n_classes=2,
    exp_r=4,
    kernel_size=3,
    deep_supervision=False,
    do_res=True,
    do_res_up_down=True,
    checkpoint_style=None,
    block_counts=[1,1,1,1,1,1,1,1,1],  # Reduced for testing
    norm_type='group',
    dim='2d',
    grn=False
)

# Initialize weights the same way
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GroupNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

orig_model.apply(init_weights)
new_model.apply(init_weights)

# Copy weights from original to new
state_dict_orig = orig_model.state_dict()
new_model.load_state_dict(state_dict_orig)

# Test with same input
torch.manual_seed(42)
dummy_input = torch.randn(1, 3, 64, 64)

with torch.no_grad():
    orig_output = orig_model(dummy_input)
    new_output = new_model(dummy_input)
    
    print(f"Original output shape: {orig_output.shape}")
    print(f"New output shape: {new_output.shape}")
    
    if orig_output.shape == new_output.shape:
        diff = torch.abs(orig_output - new_output)
        print(f"Max absolute difference: {diff.max().item():.6f}")
        print(f"Mean absolute difference: {diff.mean().item():.6f}")
        
        if diff.max() < 1e-5:
            print("✓ Models produce identical outputs!")
        else:
            print("✗ Models produce different outputs")
    else:
        print("✗ Output shapes don't match!")




# import torch

# # Test both implementations
# from mednext.mednextorig.mednextorig import MedNeXt as MedNeXtOrig
# from mednext.newmednext.mednextnew import MedNeXtNew

# # Create both models with same config
# config = {
#     'in_channels': 1,
#     'n_channels': 32,
#     'n_classes': 4,
#     'exp_r': [2,3,4,4,4,4,4,3,2],
#     'kernel_size': 3,
#     'deep_supervision': False,
#     'do_res': True,
#     'do_res_up_down': True,
#     'block_counts': [2,2,2,2,2,2,2,2,2],
#     'norm_type': 'group',
#     'dim': '2d',
#     'grn': False
# }

# model_orig = MedNeXtOrig(**config)
# model_new = MedNeXtNew(**config)

# # Count parameters
# params_orig = sum(p.numel() for p in model_orig.parameters())
# params_new = sum(p.numel() for p in model_new.parameters())

# print(f"Original parameters: {params_orig:,}")
# print(f"New parameters: {params_new:,}")
# print(f"Difference: {abs(params_orig - params_new):,}")

# # Test forward pass with same input
# torch.manual_seed(42)
# x = torch.randn(1, 1, 224, 224)

# with torch.no_grad():
#     out_orig = model_orig(x)
#     out_new = model_new(x)

# print(f"\nOutput shapes:")
# print(f"Original: {out_orig.shape}")
# print(f"New: {out_new.shape}")

# print(f"\nOutput statistics:")
# print(f"Original - mean: {out_orig.mean():.6f}, std: {out_orig.std():.6f}")
# print(f"New - mean: {out_new.mean():.6f}, std: {out_new.std():.6f}")