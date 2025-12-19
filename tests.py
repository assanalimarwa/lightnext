import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from convnext.lightnext import LightNext
from utils.utilities import test_single_volume
from convnext.mednext.mednextorig import MedNeXt
from convnext.mednext.newmednext.mednextnew import MedNeXtNew




# def inference_acdc(model, model_path='best_model.pth'):
#     from datasets.dataset_acdc import BaseDataSets
    
#     # Setup logging
#     logging.basicConfig(level=logging.INFO,
#                        format='%(asctime)s - %(levelname)s - %(message)s')
    
#     # Parameters
#     num_classes = 4
#     img_size = 224
#     base_dir = '/home/user/lightnext/datasets/ACDC'
#     test_save_path = '/home/user/lightnext/predictions/'
    
#     # Create test save directory
#     import os
#     os.makedirs(test_save_path, exist_ok=True)
    
#     # Device setup
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
    
#     # Load model
#     logging.info(f"Loading model from {model_path}")
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()
    
#     # Load test dataset
#     db_test = BaseDataSets(base_dir=base_dir, split="test_vol")
#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
#     logging.info("{} test iterations per epoch".format(len(testloader)))
    
#     metric_list = 0.0
    
#     with torch.no_grad():
#         for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
#             image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            
#             metric_i = test_single_volume(
#                 image, label, model, 
#                 classes=num_classes, 
#                 patch_size=[img_size, img_size],
#                 test_save_path=test_save_path, 
#                 case=case_name, 
#                 z_spacing=1.0  # Adjust if needed
#             )
            
#             metric_list += np.array(metric_i)
            
#             logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % 
#                         (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    
#     # Calculate average metrics
#     metric_list = metric_list / len(db_test)
    
#     # Log per-class metrics
#     for i in range(1, num_classes):
#         logging.info('Mean class %d mean_dice %f mean_hd95 %f' % 
#                     (i, metric_list[i-1][0], metric_list[i-1][1]))
    
#     # Overall performance
#     performance = np.mean(metric_list, axis=0)[0]
#     mean_hd95 = np.mean(metric_list, axis=0)[1]
    
#     logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % 
#                 (performance, mean_hd95))
    
#     return performance, mean_hd95





# def create_mednextv1_base_orig(num_input_channels, num_classes, kernel_size=3, ds=False):

#     return MedNeXt(
#         in_channels = num_input_channels, 
#         n_channels = 32,
#         n_classes = num_classes, 
#         exp_r=[2,3,4,4,4,4,4,3,2],       
#         kernel_size=kernel_size,         
#         deep_supervision=ds,             
#         do_res=True,                     
#         do_res_up_down = True,
#         block_counts = [2,2,2,2,2,2,2,2,2]

        
#     )


# model_orig = create_mednextv1_base_orig(1, 4)



# # Run inference
# model_path = '/home/user/lightnext/snapshot/orig_model1.pth'
# dice_score, hd95_score = inference_acdc(model_orig, model_path)

# print(f"Final Results - Dice: {dice_score:.4f}, HD95: {hd95_score:.4f}")
# print(123)



import torch
import numpy as np
import logging
from convnext.mednext.mednextorig import MedNeXt
from torch.utils.data import DataLoader
from tqdm import tqdm

def inference_acdc(model, model_path='best_model.pth'):
    from datasets.dataset_acdc import BaseDataSets
    from utils.utilities import test_single_volume
    import os
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    num_classes = 4
    img_size = 224
    base_dir = '/home/user/lightnext/datasets/ACDC'
    test_save_path = '/home/user/lightnext/predictions/'
    os.makedirs(test_save_path, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    logging.info(f"Loading model from {model_path}")
    checkpoint = torch.load(
        model_path, 
        map_location=device, 
        weights_only=False  # ← FIX: Allow non-tensor objects
    )
    
    # Handle both formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Your new format (with config)
            state_dict = checkpoint['model_state_dict']
            logging.info(f"Loaded checkpoint from iteration {checkpoint.get('iteration', 'unknown')}")
            logging.info(f"Validation Dice: {checkpoint.get('best_dice', 'unknown'):.4f}")
        else:
            # Old format (just state_dict)
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load test dataset
    db_test = BaseDataSets(base_dir=base_dir, split="test_vol")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
    logging.info(f"{len(testloader)} test cases")
    
    metric_list = 0.0
    
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
            image = sampled_batch["image"]
            label = sampled_batch["label"]
            case_name = sampled_batch['case_name'][0]
            
            metric_i = test_single_volume(
                image, label, model, 
                classes=num_classes, 
                patch_size=[img_size, img_size],
                test_save_path=test_save_path, 
                case=case_name, 
                z_spacing=1.0
            )
            
            metric_list += np.array(metric_i)
            
            logging.info(
                f'Case {i_batch} ({case_name}): '
                f'Dice={np.mean(metric_i, axis=0)[0]:.4f}, '
                f'HD95={np.mean(metric_i, axis=0)[1]:.4f}'
            )
    
    # Average metrics
    metric_list = metric_list / len(db_test)
    
    # Per-class results
    for i in range(1, num_classes):
        logging.info(
            f'Class {i}: Dice={metric_list[i-1][0]:.4f}, '
            f'HD95={metric_list[i-1][1]:.4f}'
        )
    
    # Overall performance
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    
    logging.info('='*60)
    logging.info(f'FINAL TEST RESULTS:')
    logging.info(f'  Mean Dice: {performance:.4f}')
    logging.info(f'  Mean HD95: {mean_hd95:.4f}')
    logging.info('='*60)
    
    return performance, mean_hd95


# ============================================
# USAGE
# ============================================

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


# Run inference
model_path = '/home/user/lightnext/snapshot/new_model1.pth'
dice_score, hd95_score = inference_acdc(model_new, model_path)

print("\n" + "="*60)
print(f"Final Test Results:")
print(f"  Dice Score: {dice_score:.4f}")
print(f"  HD95 Score: {hd95_score:.4f}")
print("="*60)
