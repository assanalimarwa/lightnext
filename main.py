import torch 
import torch.nn as nn 
from convnext.convnextblock import ConvNextBlock
from convnext.lightnextv1 import LightNextv1
from datasets.trainer import trainer_acdc
from datasets.dataset_acdc import BaseDataSets, RandomGenerator
from torch.utils.data import DataLoader
from torchvision import transforms
from convnext.mednext.mednextorig import MedNeXt
from convnext.mednext.newmednext.mednextnew import MedNeXtNew

from thop import profile



# img = torch.rand(1, 96, 224, 224)

def create_mednextv1_base_orig(num_input_channels, num_classes, kernel_size=3, ds=False):

    return MedNeXt(
        in_channels = num_input_channels, 
        n_channels = 32,
        n_classes = num_classes, 
        exp_r=[2,3,4,4,4,4,4,3,2],       
        kernel_size=kernel_size,         
        deep_supervision=ds,             
        do_res=True,                     
        do_res_up_down = True,
        block_counts = [2,2,2,2,2,2,2,2,2]

        
    )


model_orig = create_mednextv1_base_orig(1, 4)

img = torch.rand(1, 1, 224, 224)

print(model_orig(img).shape)



#Model Calculation
total_params = sum(p.numel() for p in model_orig.parameters())
print(total_params)
dummy = torch.randn(1, 1, 224, 224)  # adjust input size
macs, params = profile(model_orig, inputs=(dummy,))

# Convert MACs → FLOPs (roughly FLOPs ≈ 2 × MACs)
flops = 2 * macs

print(f"MACs:  {macs / 1e9:.3f} G")
print(f"FLOPs: {flops / 1e9:.3f} G")
print(f"Params: {params / 1e6:.3f} M")


train = trainer_acdc(model_orig, 'orig')






def create_mednextv1_base_new(num_input_channels, num_classes, kernel_size=3, ds=False):

    return MedNeXtNew(
        in_channels = num_input_channels, 
        n_channels = 32,
        n_classes = num_classes, 
        exp_r=[2,3,4,4,4,4,4,3,2],       
        kernel_size=kernel_size,         
        deep_supervision=ds,             
        do_res=True,                     
        do_res_up_down = True,
        block_counts = [2,2,2,2,2,2,2,2,2]

        
    )


model_new = create_mednextv1_base_new(1, 4)

img = torch.rand(1, 1, 224, 224)

print(model_new(img).shape)


#Model Calculation
total_params = sum(p.numel() for p in model_new.parameters())
print(total_params)
dummy = torch.randn(1, 1, 224, 224)  # adjust input size
macs, params = profile(model_new, inputs=(dummy,))

# Convert MACs → FLOPs (roughly FLOPs ≈ 2 × MACs)
flops = 2 * macs

print(f"MACs:  {macs / 1e9:.3f} G")
print(f"FLOPs: {flops / 1e9:.3f} G")
print(f"Params: {params / 1e6:.3f} M")


train = trainer_acdc(model_new, 'new')










# out = model(img)

# print(out.shape)

# def convnext_tiny(**kwargs):
#     model = ConvNeXtModel(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
#     return model

# model = convnext_tiny()

# print(model)

# print(torch.cuda.is_available())


# db_train = BaseDataSets(base_dir='/home/user/lightnext/datasets/ACDC', split="train", transform=transforms.Compose([
#         RandomGenerator([224, 224])]))

# trainloader = DataLoader(db_train, batch_size=4, shuffle=True,
#                              num_workers=8, pin_memory=True)

# train = trainer_acdc(model)

# train()

# batch = next(iter(trainloader))  # batch is a list of dicts
# print(len(batch))  # batch size

# # check the first item
# first_sample = batch[0]
# print(first_sample.keys())  # dict keys: image, label, idx, case_name

# # shapes
# print("Image shape:", first_sample['image'].shape)
# print("Label shape:", first_sample['label'].shape)

#new











