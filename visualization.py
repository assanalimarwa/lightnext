import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
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
    vis_save_path = '/home/user/lightnext/visualizations/'
    
    # Create directories
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
    
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
            image = sampled_batch["image"]  # Shape: (1, 1, H, W) or (1, 1, D, H, W)
            label = sampled_batch["label"]  # Shape: (1, H, W) or (1, D, H, W)
            case_name = sampled_batch['case_name'][0]
            
            print(f"Image shape: {image.shape}, Label shape: {label.shape}")
            
            # Handle 3D volumes - take middle slice for visualization
            if len(image.shape) == 5:  # (B, C, D, H, W)
                mid_slice = image.shape[2] // 2
                image_slice = image[:, :, mid_slice, :, :]  # (1, 1, H, W)
                label_slice = label[:, mid_slice, :, :]     # (1, H, W)
            elif len(image.shape) == 4:  # (B, C, H, W)
                image_slice = image
                label_slice = label
            else:
                logging.warning(f"Unexpected image shape: {image.shape}")
                continue
            
            # Resize if needed
            if image_slice.shape[2] != img_size or image_slice.shape[3] != img_size:
                image_slice = torch.nn.functional.interpolate(
                    image_slice, 
                    size=(img_size, img_size), 
                    mode='bilinear', 
                    align_corners=False
                )
                label_slice = torch.nn.functional.interpolate(
                    label_slice.unsqueeze(1).float(), 
                    size=(img_size, img_size), 
                    mode='nearest'
                ).squeeze(1).long()
            
            # Move to device and get prediction
            image_gpu = image_slice.to(device)
            outputs = model(image_gpu)
            
            # Get predicted mask
            prediction = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            
            # Move back to CPU for visualization
            image_np = image_slice.cpu().numpy()[0, 0]  # Shape: (H, W)
            label_np = label_slice.cpu().numpy()[0]     # Shape: (H, W)
            pred_np = prediction.cpu().numpy()[0]       # Shape: (H, W)
            
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
            
            # 4. Comparison
            axes[3].imshow(image_display, cmap='gray')
            # Show overlap: GT in green, Pred in red, overlap in yellow
            overlap = np.zeros((*label_np.shape, 3))
            overlap[label_np > 0] = [0, 1, 0]  # GT green
            overlap[pred_np > 0] = [1, 0, 0]   # Pred red
            overlap[(label_np > 0) & (pred_np > 0)] = [1, 1, 0]  # Overlap yellow
            axes[3].imshow(overlap, alpha=0.6)
            axes[3].set_title('Overlap (GT:Green, Pred:Red, Both:Yellow)', fontsize=14)
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
            save_filename = os.path.join(vis_save_path, f'{case_name}_visualization.png')
            plt.savefig(save_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            logging.info(f'Saved visualization for case {i_batch}: {case_name}')
            
            # Only visualize first 20 cases to save time
            if i_batch >= 19:
                break
    
    logging.info(f'\nVisualizations saved to: {vis_save_path}')
    return "Visualization complete!"


# Run
if __name__ == "__main__":
    from convnext.lightnext import LightNext # Adjust import
    
    # Initialize model
    model = LightNext(
        in_chans=1, 
        num_classes=4, 
        depths=[3, 3, 9, 3], 
        dims=[96, 192, 384, 768]
    )
    
    # Run visualization
    model_path = '/home/user/lightnext/snapshot/best_model.pth'
    inference_with_visualization(model, model_path)