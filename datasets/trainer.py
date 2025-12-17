import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utilities import DiceLoss
from torchvision import transforms
from utils.utilities import DiceLoss, test_single_volume, calculate_metric_percase




def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ All seeds set to {seed}")


def trainer_acdc(model, model_type: str, seed=42, patience=20):
    """
    Train model with comprehensive metrics logging and early stopping.
    
    Args:
        model: PyTorch model to train
        model_type: Model identifier for saving checkpoints
        seed: Random seed for reproducibility
        patience: Number of epochs without improvement before early stopping
    
    Returns:
        best_performance: Best validation Dice score achieved
        best_hd95: Best validation HD95 score achieved
    """
    from datasets.dataset_acdc import BaseDataSets, RandomGenerator
    
    # ============================================
    # INITIALIZATION
    # ============================================
    set_seed(seed)
    
    # Training hyperparameters
    base_lr = 0.001
    num_classes = 4
    batch_size = 4
    img_size = 224
    max_epoch = 100
    max_iterations = max_epoch * 1500 // batch_size

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshot_path = '/home/user/lightnext/snapshot/'

    # ============================================
    # DATASET & DATALOADER
    # ============================================
    db_train = BaseDataSets(
        base_dir='/home/user/lightnext/datasets/ACDC', 
        split="train", 
        transform=transforms.Compose([RandomGenerator([224, 224])])
    )
    db_val = BaseDataSets(
        base_dir='/home/user/lightnext/datasets/ACDC', 
        split="val"
    )
    
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    trainloader = DataLoader(
        db_train, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=8, 
        pin_memory=True, 
        worker_init_fn=worker_init_fn
    )
    
    valloader = DataLoader(
        db_val, 
        batch_size=1, 
        shuffle=False,
        num_workers=1
    )
    
    # ============================================
    # MODEL, OPTIMIZER, LOSS
    # ============================================
    model.to(device)
    model.train()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )
    
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = DiceLoss(num_classes)

    # ============================================
    # LOGGING & TRACKING
    # ============================================
    writer = SummaryWriter(snapshot_path + '/log')
    
    logging.info("")
    logging.info("="*70)
    logging.info("TRAINING CONFIGURATION")
    logging.info("="*70)
    logging.info(f"Model Type:              {model_type}")
    logging.info(f"Random Seed:             {seed}")
    logging.info(f"Device:                  {device}")
    logging.info(f"Optimizer:               AdamW")
    logging.info(f"Learning Rate:           {base_lr}")
    logging.info(f"Batch Size:              {batch_size}")
    logging.info(f"Max Epochs:              {max_epoch}")
    logging.info(f"Early Stopping Patience: {patience}")
    logging.info(f"Train Samples:           {len(db_train)}")
    logging.info(f"Val Samples:             {len(db_val)}")
    logging.info(f"Train Batches/Epoch:     {len(trainloader)}")
    logging.info(f"Val Volumes:             {len(valloader)}")
    logging.info("="*70)
    logging.info("")

    # Tracking variables
    iter_num = 0
    best_performance = 0.0  # Best validation Dice
    best_hd95 = 0.0
    best_iteration = 0
    patience_counter = 0
    
    # For epoch-level averaging
    epoch_train_loss = []
    epoch_train_ce = []
    epoch_train_dice_loss = []
    
    # History tracking
    train_loss_history = []
    val_dice_history = []
    val_hd95_history = []
    
    iterator = tqdm(range(max_epoch), ncols=70, desc="Training")
    
    # ============================================
    # TRAINING LOOP
    # ============================================
    for epoch_num in iterator:
        
        # Reset epoch metrics
        epoch_train_loss.clear()
        epoch_train_ce.clear()
        epoch_train_dice_loss.clear()
        
        # ============================================
        # TRAINING PHASE
        # ============================================
        model.train()
        
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
            
            # Forward pass
            outputs = model(volume_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice_val = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice_val
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Learning rate decay (polynomial)
            lr_ = float(base_lr * (1.0 - float(iter_num) / float(max_iterations)) ** 0.9)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            
            # Track losses for epoch averaging
            epoch_train_loss.append(loss.item())
            epoch_train_ce.append(loss_ce.item())
            epoch_train_dice_loss.append(loss_dice_val.item())
            
            # Log to tensorboard
            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/total_loss', loss.item(), iter_num)
            writer.add_scalar('train/loss_ce', loss_ce.item(), iter_num)
            writer.add_scalar('train/loss_dice', loss_dice_val.item(), iter_num)

            # Periodic detailed logging
            if iter_num % 100 == 0:
                logging.info(
                    f'[Epoch {epoch_num+1:3d} | Iter {iter_num:5d}] '
                    f'Loss: {loss.item():.4f} | '
                    f'CE: {loss_ce.item():.4f} | '
                    f'Dice Loss: {loss_dice_val.item():.4f} | '
                    f'LR: {lr_:.6f}'
                )

            # Visualization
            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs_vis = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_vis[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # ============================================
        # EPOCH-LEVEL TRAINING METRICS
        # ============================================
        avg_train_loss = np.mean(epoch_train_loss)
        avg_train_ce = np.mean(epoch_train_ce)
        avg_train_dice_loss = np.mean(epoch_train_dice_loss)
        
        train_loss_history.append(avg_train_loss)
        
        writer.add_scalar('epoch/train_loss', avg_train_loss, epoch_num)
        writer.add_scalar('epoch/train_ce', avg_train_ce, epoch_num)
        writer.add_scalar('epoch/train_dice_loss', avg_train_dice_loss, epoch_num)
        
        # ============================================
        # VALIDATION PHASE
        # ============================================
        model.eval()
        metric_list = 0.0
        
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                image, label = sampled_batch["image"], sampled_batch["label"]
                
                # Use test_single_volume (handles 3D volumes correctly)
                metric_i = test_single_volume(
                    image, label, model, 
                    classes=num_classes,
                    patch_size=[img_size, img_size]
                )
                metric_list += np.array(metric_i)
        
        # Average validation metrics
        metric_list = metric_list / len(db_val)
        
        # Per-class metrics (RV, MYO, LV)
        class_names = ['RV', 'MYO', 'LV']
        for class_i in range(num_classes - 1):
            class_dice = metric_list[class_i, 0]
            class_hd95 = metric_list[class_i, 1]
            
            writer.add_scalar(
                f'val/class_{class_i+1}_dice',
                class_dice, 
                epoch_num
            )
            writer.add_scalar(
                f'val/class_{class_i+1}_hd95',
                class_hd95, 
                epoch_num
            )

        # Overall validation performance
        performance = np.mean(metric_list, axis=0)[0]  # Mean Dice
        mean_hd95 = np.mean(metric_list, axis=0)[1]     # Mean HD95
        
        val_dice_history.append(performance)
        val_hd95_history.append(mean_hd95)
        
        writer.add_scalar('epoch/val_mean_dice', performance, epoch_num)
        writer.add_scalar('epoch/val_mean_hd95', mean_hd95, epoch_num)
        
        # ============================================
        # COMPREHENSIVE EPOCH SUMMARY
        # ============================================
        logging.info("")
        logging.info("="*70)
        logging.info(f"EPOCH {epoch_num + 1}/{max_epoch} SUMMARY")
        logging.info("="*70)
        
        # Training metrics
        logging.info("TRAINING METRICS:")
        logging.info(f"  Avg Total Loss:     {avg_train_loss:.4f}")
        logging.info(f"  Avg CE Loss:        {avg_train_ce:.4f}")
        logging.info(f"  Avg Dice Loss:      {avg_train_dice_loss:.4f}")
        logging.info(f"  Learning Rate:      {lr_:.6f}")
        logging.info("")
        
        # Validation metrics
        logging.info("VALIDATION METRICS:")
        logging.info(f"  Mean Dice Score:    {performance:.4f}")
        logging.info(f"  Mean HD95:          {mean_hd95:.2f}")
        logging.info("")
        logging.info("  Per-Class Dice Scores:")
        for class_i in range(num_classes - 1):
            logging.info(
                f"    {class_names[class_i]:3s}: {metric_list[class_i, 0]:.4f} "
                f"(HD95: {metric_list[class_i, 1]:.2f})"
            )
        
        # ============================================
        # MODEL CHECKPOINT & EARLY STOPPING
        # ============================================
        
        improved = False
        if performance > best_performance:
            improved = True
            best_iteration = iter_num
            best_performance = performance
            best_hd95 = mean_hd95
            patience_counter = 0
            
            # Save checkpoint
            save_best = os.path.join(
                snapshot_path, 
                f'{model_type}_model_best2.pth'
            )
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'in_channels': 1,
                    'n_channels': 32,
                    'n_classes': 4,
                    'exp_r': [2,3,4,4,4,4,4,3,2],
                    'kernel_size': 7,
                    'deep_supervision': False,
                    'do_res': True,
                    'do_res_up_down': True,
                    'block_counts': [2,2,2,2,2,2,2,2,2]
                },
                'seed': seed,
                'epoch': epoch_num,
                'iteration': iter_num,
                'best_dice': best_performance,
                'best_hd95': best_hd95,
                'train_loss_history': train_loss_history,
                'val_dice_history': val_dice_history,
                'val_hd95_history': val_hd95_history,
                'optimizer_state': optimizer.state_dict()
            }, save_best)
            
            logging.info("")
            logging.info("🎉 NEW BEST MODEL SAVED!")
            logging.info(f"  Checkpoint:      {save_best}")
            logging.info(f"  Iteration:       {iter_num}")
            logging.info(f"  Val Dice:        {performance:.4f} ↑")
            logging.info(f"  Val HD95:        {mean_hd95:.2f}")
            logging.info(f"  Improvement:     +{(performance - val_dice_history[-2] if len(val_dice_history) > 1 else 0):.4f}")
            
        else:
            patience_counter += 1
            logging.info("")
            logging.info(f"⚠️  NO IMPROVEMENT")
            logging.info(f"  Patience Counter:  {patience_counter}/{patience}")
            logging.info(f"  Current Val Dice:  {performance:.4f}")
            logging.info(f"  Best Val Dice:     {best_performance:.4f}")
            logging.info(f"  Gap:               -{(best_performance - performance):.4f}")
        
        # Progress bar update
        iterator.set_postfix({
            'Loss': f'{avg_train_loss:.3f}',
            'Dice': f'{performance:.3f}',
            'Best': f'{best_performance:.3f}',
            'Pat': f'{patience_counter}/{patience}'
        })
        
        logging.info("="*70)
        logging.info("")
        
        # ============================================
        # EARLY STOPPING CHECK
        # ============================================
        if patience_counter >= patience:
            logging.info("")
            logging.info("🛑 "*15)
            logging.info("EARLY STOPPING TRIGGERED!")
            logging.info("🛑 "*15)
            logging.info(f"No improvement for {patience} consecutive epochs")
            logging.info(f"Training stopped at epoch {epoch_num + 1}/{max_epoch}")
            logging.info("")
            logging.info("BEST MODEL STATISTICS:")
            logging.info(f"  Best Epoch:       {best_iteration // len(trainloader)}")
            logging.info(f"  Best Iteration:   {best_iteration}")
            logging.info(f"  Best Val Dice:    {best_performance:.4f}")
            logging.info(f"  Best Val HD95:    {best_hd95:.2f}")
            logging.info("🛑 "*15)
            logging.info("")
            break
    
    writer.close()
    
    # ============================================
    # FINAL TRAINING SUMMARY
    # ============================================
    logging.info("")
    logging.info("="*70)
    logging.info("🎓 TRAINING COMPLETE!")
    logging.info("="*70)
    logging.info(f"Model Type:           {model_type}")
    logging.info(f"Total Epochs:         {epoch_num + 1}/{max_epoch}")
    logging.info(f"Total Iterations:     {iter_num}")
    logging.info(f"Early Stopped:        {'Yes' if patience_counter >= patience else 'No'}")
    logging.info("")
    logging.info("BEST MODEL PERFORMANCE:")
    logging.info(f"  Best Epoch:         {best_iteration // len(trainloader)}")
    logging.info(f"  Best Iteration:     {best_iteration}")
    logging.info(f"  Best Val Dice:      {best_performance:.4f}")
    logging.info(f"  Best Val HD95:      {best_hd95:.2f}")
    logging.info("")
    logging.info("TRAINING HISTORY:")
    logging.info(f"  Initial Train Loss: {train_loss_history[0]:.4f}")
    logging.info(f"  Final Train Loss:   {train_loss_history[-1]:.4f}")
    logging.info(f"  Initial Val Dice:   {val_dice_history[0]:.4f}")
    logging.info(f"  Final Val Dice:     {val_dice_history[-1]:.4f}")
    logging.info(f"  Best Val Dice:      {max(val_dice_history):.4f}")
    logging.info("")
    logging.info(f"Model saved to: {snapshot_path}{model_type}_model_best.pth")
    logging.info("="*70)
    logging.info("")
    
    return best_performance, best_hd95



# def set_seed(seed=42):
#     """Set all random seeds for reproducibility"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
    
#     # Make CUDA deterministic
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
    
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     print(f"✅ All seeds set to {seed}")


# def trainer_acdc(model, model_type: str, seed=42):
#     from datasets.dataset_acdc import BaseDataSets, RandomGenerator
    
#     # ============================================
#     # SET SEED FIRST (BEFORE ANY RANDOM OPERATIONS)
#     # ============================================
#     set_seed(seed)
    
#     base_lr = 0.001
#     num_classes = 4
#     batch_size = 4
#     img_size = 224
#     max_epoch = 100
#     max_iterations = max_epoch * 1500 // batch_size

#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")

#     snapshot_path = '/home/user/lightnext/snapshot/'

#     db_train = BaseDataSets(
#         base_dir='/home/user/lightnext/datasets/ACDC', 
#         split="train", 
#         transform=transforms.Compose([RandomGenerator([224, 224])])
#     )
#     db_val = BaseDataSets(
#         base_dir='/home/user/lightnext/datasets/ACDC', 
#         split="val"
#     )
    
#     def worker_init_fn(worker_id):
#         # Seed each worker with base seed + worker_id
#         worker_seed = seed + worker_id
#         random.seed(worker_seed)
#         np.random.seed(worker_seed)
#         torch.manual_seed(worker_seed)
    
#     trainloader = DataLoader(
#         db_train, 
#         batch_size=batch_size, 
#         shuffle=True,
#         num_workers=8, 
#         pin_memory=True, 
#         worker_init_fn=worker_init_fn
#     )
    
#     valloader = DataLoader(
#         db_val, 
#         batch_size=1, 
#         shuffle=False,
#         num_workers=1
#     )
    
#     model.to(device)
#     model.train()
    
#     optimizer = optim.SGD(
#         model.parameters(), 
#         lr=base_lr,
#         momentum=0.9, 
#         weight_decay=0.0001
#     )
    
#     ce_loss = CrossEntropyLoss(ignore_index=4)
#     dice_loss = DiceLoss(num_classes)

#     writer = SummaryWriter(snapshot_path + '/log')
#     logging.info("{} iterations per epoch".format(len(trainloader)))
#     logging.info("{} val iterations per epoch".format(len(valloader)))

#     iter_num = 0
#     max_epoch = 100
#     best_performance = 0.0
#     iterator = tqdm(range(max_epoch), ncols=70)
    
#     for epoch_num in iterator:
#         for i_batch, sampled_batch in enumerate(trainloader):
#             volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#             volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
            
#             outputs = model(volume_batch)
#             loss_ce = ce_loss(outputs, label_batch[:].long())
#             loss_dice = dice_loss(outputs, label_batch, softmax=True)
#             loss = 0.5 * loss_ce + 0.5 * loss_dice
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             # Learning rate decay
#             lr_ = float(base_lr * (1.0 - float(iter_num) / float(max_iterations)) ** 0.9)
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr_

#             iter_num = iter_num + 1
#             writer.add_scalar('info/lr', lr_, iter_num)
#             writer.add_scalar('info/total_loss', loss, iter_num)
#             writer.add_scalar('info/loss_ce', loss_ce, iter_num)

#             logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

#             # Visualization
#             if iter_num % 20 == 0:
#                 image = volume_batch[1, 0:1, :, :]
#                 image = (image - image.min()) / (image.max() - image.min())
#                 writer.add_image('train/Image', image, iter_num)
#                 outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
#                 writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
#                 labs = label_batch[1, ...].unsqueeze(0) * 50
#                 writer.add_image('train/GroundTruth', labs, iter_num)

#             # Validation
#             if iter_num > 0 and iter_num % 500 == 0:
#                 model.eval()
#                 metric_list = 0.0
                
#                 for i_batch, sampled_batch in enumerate(valloader):
#                     image, label = sampled_batch["image"], sampled_batch["label"]
#                     metric_i = test_single_volume(
#                         image, label, model, 
#                         classes=num_classes,
#                         patch_size=[img_size, img_size]
#                     )
#                     metric_list += np.array(metric_i)
                
#                 metric_list = metric_list / len(db_val)
                
#                 for class_i in range(num_classes - 1):
#                     writer.add_scalar(
#                         'info/val_{}_dice'.format(class_i + 1),
#                         metric_list[class_i, 0], 
#                         iter_num
#                     )
#                     writer.add_scalar(
#                         'info/val_{}_hd95'.format(class_i + 1),
#                         metric_list[class_i, 1], 
#                         iter_num
#                     )

#                 performance = np.mean(metric_list, axis=0)[0]
#                 mean_hd95 = np.mean(metric_list, axis=0)[1]
                
#                 writer.add_scalar('info/val_mean_dice', performance, iter_num)
#                 writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

#                 if performance > best_performance:
#                     best_iteration = iter_num
#                     best_performance = performance
#                     best_hd95 = mean_hd95
                    
#                     save_best = os.path.join(
#                         snapshot_path, 
#                         'orig_modelrep1.pth' if model_type == 'orig' else 'new_modelrep1.pth'
#                     )
                    
#                     # Save with seed info
#                     torch.save({
#                         'model_state_dict': model.state_dict(),
#                         'config': {
#                             'in_channels': 1,
#                             'n_channels': 32,
#                             'n_classes': 4,
#                             'exp_r': [2,3,4,4,4,4,4,3,2],
#                             'kernel_size': 5,
#                             'deep_supervision': False,
#                             'do_res': True,
#                             'do_res_up_down': True,
#                             'block_counts': [2,2,2,2,2,2,2,2,2]
#                         },
#                         'seed': seed,
#                         'best_dice': performance,
#                         'best_hd95': mean_hd95,
#                         'iteration': iter_num
#                     }, save_best)
                    
#                     logging.info(
#                         'Best model | iteration %d : mean_dice : %f mean_hd95 : %f' % 
#                         (iter_num, performance, mean_hd95)
#                     )

#                 logging.info(
#                     'iteration %d : mean_dice : %f mean_hd95 : %f' % 
#                     (iter_num, performance, mean_hd95)
#                 )
#                 model.train()
    
#     writer.close()
#     return best_performance, best_hd95



# def trainer_acdc(model, model_type: str):
#     from datasets.dataset_acdc import BaseDataSets, RandomGenerator
#     base_lr = 0.001
#     num_classes = 4
#     batch_size = 4
#     img_size = 224
#     max_epoch = 100
#     max_iterations = max_epoch * 1500 // batch_size

#     if torch.cuda.is_available():
#     # If available, set the device to 'cuda' (GPU)
#         device = torch.device("cuda")

#     snapshot_path = '/home/user/lightnext/snapshot/'

#     db_train = BaseDataSets(base_dir='/home/user/lightnext/datasets/ACDC', split="train", transform=transforms.Compose([
#         RandomGenerator([224, 224])]))
#     db_val = BaseDataSets(base_dir='/home/user/lightnext/datasets/ACDC', split="val")
#     def worker_init_fn(worker_id):
#         random.seed(1234 + worker_id)
#     trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
#                              num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
#     valloader = DataLoader(db_val, batch_size=1, shuffle=False,
#                            num_workers=1)
#     model.to(device)
#     model.train()
#     optimizer = optim.SGD(model.parameters(), lr=base_lr,
#                           momentum=0.9, weight_decay=0.0001)
#     ce_loss = CrossEntropyLoss(ignore_index=4)
#     dice_loss = DiceLoss(num_classes)

#     writer = SummaryWriter(snapshot_path + '/log')
#     logging.info("{} iterations per epoch".format(len(trainloader)))
#     logging.info("{} val iterations per epoch".format(len(valloader)))
#     # logging.info("{} test iterations per epoch".format(len(testloader)))

#     iter_num = 0
#     max_epoch = 100
#     best_performance = 0.0
#     iterator = tqdm(range(max_epoch), ncols=70)
#     for epoch_num in iterator:
#         for i_batch, sampled_batch in enumerate(trainloader):
#             volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#             volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
#             outputs = model(volume_batch)
#             loss_ce = ce_loss(outputs, label_batch[:].long())
#             loss_dice = dice_loss(outputs, label_batch, softmax=True)
#             loss = 0.5 * loss_ce + 0.5 * loss_dice
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             # to change in near future
#             lr_ = float(base_lr * (1.0 - float(iter_num) / float(max_iterations)) ** 0.9)
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr_

#             iter_num = iter_num + 1
#             writer.add_scalar('info/lr', lr_, iter_num)
#             writer.add_scalar('info/total_loss', loss, iter_num)
#             writer.add_scalar('info/loss_ce', loss_ce, iter_num)

#             logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

#             if iter_num % 20 == 0:
#                 image = volume_batch[1, 0:1, :, :]
#                 image = (image - image.min()) / (image.max() - image.min())
#                 writer.add_image('train/Image', image, iter_num)
#                 outputs = torch.argmax(torch.softmax(
#                     outputs, dim=1), dim=1, keepdim=True)
#                 writer.add_image('train/Prediction',
#                                  outputs[1, ...] * 50, iter_num)
#                 labs = label_batch[1, ...].unsqueeze(0) * 50
#                 writer.add_image('train/GroundTruth', labs, iter_num)

#             if iter_num > 0 and iter_num % 500 == 0:  # 500
#                 model.eval()
#                 metric_list = 0.0
#                 for i_batch, sampled_batch in enumerate(valloader):
#                     image, label = sampled_batch["image"], sampled_batch["label"]
#                     metric_i = test_single_volume(image, label, model, classes=num_classes,
#                                                   patch_size=[img_size, img_size])
#                     metric_list += np.array(metric_i)
#                 metric_list = metric_list / len(db_val)
#                 for class_i in range(num_classes - 1):
#                     writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
#                                       metric_list[class_i, 0], iter_num)
#                     writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
#                                       metric_list[class_i, 1], iter_num)

#                 performance = np.mean(metric_list, axis=0)[0]

#                 mean_hd95 = np.mean(metric_list, axis=0)[1]
#                 writer.add_scalar('info/val_mean_dice', performance, iter_num)
#                 writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

#                 if performance > best_performance:
#                     best_iteration, best_performance, best_hd95 = iter_num, performance, mean_hd95
#                     save_best = os.path.join(snapshot_path, 'orig_model.pth' if model_type == 'orig' else 'new_model.pth')
#                     torch.save(model.state_dict(), save_best)
#                     logging.info('Best model | iteration %d : mean_dice : %f mean_hd95 : %f' % (
#                     iter_num, performance, mean_hd95))

#                 logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
#                 model.train()

            


