import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from convnext.lightnext import LightNext
from utils.utilities import test_single_volume
from convnext.lightnextv1 import LightNextv1




def inference_acdc(model, model_path='best_model.pth'):
    from datasets.dataset_acdc import BaseDataSets
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parameters
    num_classes = 4
    img_size = 224
    base_dir = '/home/user/lightnext/datasets/ACDC'
    test_save_path = '/home/user/lightnext/predictions/'
    
    # Create test save directory
    import os
    os.makedirs(test_save_path, exist_ok=True)
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load model
    logging.info(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load test dataset
    db_test = BaseDataSets(base_dir=base_dir, split="test_vol")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
    logging.info("{} test iterations per epoch".format(len(testloader)))
    
    metric_list = 0.0
    
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            
            metric_i = test_single_volume(
                image, label, model, 
                classes=num_classes, 
                patch_size=[img_size, img_size],
                test_save_path=test_save_path, 
                case=case_name, 
                z_spacing=1.0  # Adjust if needed
            )
            
            metric_list += np.array(metric_i)
            
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % 
                        (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    
    # Calculate average metrics
    metric_list = metric_list / len(db_test)
    
    # Log per-class metrics
    for i in range(1, num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % 
                    (i, metric_list[i-1][0], metric_list[i-1][1]))
    
    # Overall performance
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % 
                (performance, mean_hd95))
    
    return performance, mean_hd95


# Usage:

    
    # Initialize model with same parameters as training
model = LightNextv1(
    in_chans=1, 
    num_classes=4, 
    depths=[3, 3, 9, 3], 
    dims=[96, 192, 384, 768]
)

# Run inference
model_path = '/home/user/lightnext/snapshot/best_model.pth'
dice_score, hd95_score = inference_acdc(model, model_path)

print(f"Final Results - Dice: {dice_score:.4f}, HD95: {hd95_score:.4f}")
print(123)

