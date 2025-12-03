import torch 
import torch.nn as nn 
from convnext.convnextblock import ConvNextBlock
from convnext.lightnext import LightNext
from datasets.trainer import trainer_acdc
from datasets.dataset_acdc import BaseDataSets, RandomGenerator
from torch.utils.data import DataLoader
from torchvision import transforms






# img = torch.rand(1, 96, 224, 224)

model = LightNext()


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

train = trainer_acdc(model)

train()

# batch = next(iter(trainloader))  # batch is a list of dicts
# print(len(batch))  # batch size

# # check the first item
# first_sample = batch[0]
# print(first_sample.keys())  # dict keys: image, label, idx, case_name

# # shapes
# print("Image shape:", first_sample['image'].shape)
# print("Label shape:", first_sample['label'].shape)

#new






