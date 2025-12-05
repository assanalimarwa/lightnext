import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets.dataset_acdc import BaseDataSets
from torch.utils.data import DataLoader
from convnext.lightnext import LightNext

def quick_visualize(model, model_path, num_samples=5):
    """
    Quickly visualize a few predictions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load test data
    db_test = BaseDataSets(base_dir='/home/user/lightnext/datasets/ACDC', split="test_vol")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for i, sampled_batch in enumerate(testloader):
            if i >= num_samples:
                break
            
            image = sampled_batch["image"].to(device)
            label = sampled_batch["label"].cpu().numpy()[0]
            case_name = sampled_batch['case_name'][0]
            
            # Get prediction
            output = model(image)
            prediction = torch.argmax(torch.softmax(output, dim=1), dim=1).cpu().numpy()[0]
            
            # Visualize
            img = image.cpu().numpy()[0, 0]
            img = (img - img.min()) / (img.max() - img.min())
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title(f'{case_name} - Original')
            axes[0].axis('off')
            
            axes[1].imshow(img, cmap='gray')
            axes[1].imshow(label, alpha=0.5, cmap='jet', vmin=0, vmax=3)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            axes[2].imshow(img, cmap='gray')
            axes[2].imshow(prediction, alpha=0.5, cmap='jet', vmin=0, vmax=3)
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'/home/user/lightnext/visualizations/{case_name}.png', dpi=150)
            plt.show()
            plt.close()

# Usage




model = LightNext(in_chans=1, num_classes=4)
quick_visualize(model, '/home/user/lightnext/snapshot/best_model.pth')