import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import test_single_volume
import matplotlib.pyplot as plt
import os

def inference_with_visualization(model, model_path='best_model.pth'):
    from datasets.dataset_acdc import BaseDataSets
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parameters
    num_classes = 4
    img_size = 224
    base_dir = '/home/user/lightnext/datasets/ACDC'
    test_save_path = '/home/user/lightnext/predictions/'
    vis_save_path = '/home/user/lightnext/visualizations/'
    
    # Create directories
    os.makedirs(test_save_path, exist_ok=True)
    os.makedirs(vis_save_path, exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    logging.info(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load test dataset
    db_test = BaseDataSets(base_dir=base_dir, split="test_vol")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
    logging.info("{} test iterations".format(len(testloader)))
    
    metric_list = 0.0
    
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
            image = sampled_batch["image"]
            label = sampled_batch["label"]
            case_name = sampled_batch['case_name'][0]
            
            # Move to device and get prediction
            image_gpu = image.to(device)
            outputs = model(image_gpu)
            
            # Get predicted mask
            prediction = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            
            # Move back to CPU for visualization
            image_np = image.cpu().numpy()[0, 0]  # Shape: (H, W)
            label_np = label.cpu().numpy()[0]     # Shape: (H, W)
            pred_np = prediction.cpu().numpy()[0] # Shape: (H, W)
            
            # Normalize image for display
            image_display = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
            
            # Create visualization
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # 1. Original Image
            axes[0].imshow(image_display, cmap='gray')
            axes[0].set_title('Original Image', fontsize=14)
            axes[0].axis('off')
            
            # 2. Ground Truth Overlay
            axes[1].imshow(image_display, cmap='gray')
            masked_gt = np.ma.masked_where(label_np == 0, label_np)
            axes[1].imshow(masked_gt, cmap='jet', alpha=0.6, vmin=0, vmax=3)
            axes[1].set_title('Ground Truth', fontsize=14)
            axes[1].axis('off')
            
            # 3. Prediction Overlay
            axes[2].imshow(image_display, cmap='gray')
            masked_pred = np.ma.masked_where(pred_np == 0, pred_np)
            axes[2].imshow(masked_pred, cmap='jet', alpha=0.6, vmin=0, vmax=3)
            axes[2].set_title('Prediction', fontsize=14)
            axes[2].axis('off')
            
            # 4. Side by side masks only
            axes[3].imshow(label_np, cmap='jet', vmin=0, vmax=3)
            axes[3].set_title('GT (left) vs Pred (right)', fontsize=14)
            axes[3].axis('off')
            
            # Add colorbar legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='black', label='Background'),
                Patch(facecolor='red', label='RV (Class 1)'),
                Patch(facecolor='yellow', label='MYO (Class 2)'),
                Patch(facecolor='blue', label='LV (Class 3)')
            ]
            fig.legend(handles=legend_elements, loc='upper center', 
                      bbox_to_anchor=(0.5, 0.95), ncol=4, fontsize=12)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            
            # Save figure
            save_filename = os.path.join(vis_save_path, f'{case_name}_comparison.png')
            plt.savefig(save_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Also save just the masks side by side
            fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
            axes2[0].imshow(label_np, cmap='jet', vmin=0, vmax=3)
            axes2[0].set_title('Ground Truth Mask', fontsize=14)
            axes2[0].axis('off')
            
            axes2[1].imshow(pred_np, cmap='jet', vmin=0, vmax=3)
            axes2[1].set_title('Predicted Mask', fontsize=14)
            axes2[1].axis('off')
            
            plt.tight_layout()
            mask_filename = os.path.join(vis_save_path, f'{case_name}_masks_only.png')
            plt.savefig(mask_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Calculate metrics
            metric_i = test_single_volume(
                image, label, model, 
                classes=num_classes, 
                patch_size=[img_size, img_size],
                test_save_path=test_save_path, 
                case=case_name, 
                z_spacing=1.0
            )
            
            metric_list += np.array(metric_i)
            
            dice_score = np.mean(metric_i, axis=0)[0]
            hd95_score = np.mean(metric_i, axis=0)[1]
            
            logging.info(f'Case {i_batch}: {case_name} | Dice: {dice_score:.4f} | HD95: {hd95_score:.4f}')
    
    # Calculate final metrics
    metric_list = metric_list / len(db_test)
    
    # Log per-class metrics
    logging.info("\n" + "="*60)
    logging.info("PER-CLASS RESULTS:")
    logging.info("="*60)
    class_names = ['RV', 'MYO', 'LV']
    for i in range(1, num_classes):
        logging.info(f'{class_names[i-1]:3s} (Class {i}) | Dice: {metric_list[i-1][0]:.4f} | HD95: {metric_list[i-1][1]:.4f}')
    
    # Overall performance
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    
    logging.info("="*60)
    logging.info(f'OVERALL | Mean Dice: {performance:.4f} | Mean HD95: {mean_hd95:.4f}')
    logging.info("="*60)
    logging.info(f'\nVisualizations saved to: {vis_save_path}')
    
    return performance, mean_hd95


# Standalone script to run
if __name__ == "__main__":
    from models.convnext_unet import ConvNeXtUNet  # Adjust import based on your file
    
    # Initialize model
    model = ConvNeXtUNet(
        in_chans=1, 
        num_classes=4, 
        depths=[3, 3, 9, 3], 
        dims=[96, 192, 384, 768]
    )
    
    # Run inference with visualization
    model_path = '/home/user/lightnext/snapshot/best_model.pth'
    dice, hd95 = inference_with_visualization(model, model_path)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"Mean Dice Score: {dice:.4f}")
    print(f"Mean HD95: {hd95:.4f}")
    print(f"{'='*60}")