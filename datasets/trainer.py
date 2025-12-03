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
from utils import DiceLoss
from torchvision import transforms
from utils import DiceLoss, test_single_volume, calculate_metric_percase

def trainer_acdc(model):
    from datasets.dataset_acdc import BaseDataSets, RandomGenerator
    
    # Training parameters
    base_lr = 0.01
    num_classes = 4
    batch_size = 4
    max_epoch = 100
    img_size = 224
    
    # Calculate max iterations for LR scheduler
    # Assuming ~1500 training samples, this gives ~37500 iterations
    max_iterations = max_epoch * 1500 // batch_size
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    snapshot_path = '/home/user/lightnext/snapshot/'
    os.makedirs(snapshot_path, exist_ok=True)
    
    # Datasets
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
        random.seed(1234 + worker_id)
    
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
    
    # Model setup
    model.to(device)
    model.train()
    
    # Optimizer and loss
    optimizer = optim.SGD(
        model.parameters(), 
        lr=base_lr,
        momentum=0.9, 
        weight_decay=0.0001
    )
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = DiceLoss(num_classes)
    
    # TensorBoard writer
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    logging.info("{} val iterations per epoch".format(len(valloader)))
    logging.info("Max iterations: {}".format(max_iterations))
    
    iter_num = 0
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
            
            # Forward pass
            outputs = model(volume_batch)
            
            # Calculate losses
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Learning rate scheduling (polynomial decay)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
            iter_num = iter_num + 1
            
            # Logging
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            
            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % 
                        (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            
            # Visualization every 20 iterations
            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs_vis = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_vis[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
            
            # Validation every 500 iterations
            if iter_num > 0 and iter_num % 500 == 0:
                model.eval()
                metric_list = 0.0
                
                with torch.no_grad():
                    for i_batch, sampled_batch in enumerate(valloader):
                        image, label = sampled_batch["image"], sampled_batch["label"]
                        metric_i = test_single_volume(
                            image, label, model, 
                            classes=num_classes,
                            patch_size=[img_size, img_size]
                        )
                        metric_list += np.array(metric_i)
                
                metric_list = metric_list / len(db_val)
                
                # Log per-class metrics
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                    metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                                    metric_list[class_i, 1], iter_num)
                
                # Calculate mean metrics
                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                
                # Save best model
                if performance > best_performance:
                    best_iteration = iter_num
                    best_performance = performance
                    best_hd95 = mean_hd95
                    save_best = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_best)
                    logging.info('Best model saved | iteration %d : mean_dice : %f mean_hd95 : %f' % 
                               (iter_num, performance, mean_hd95))
                
                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % 
                           (iter_num, performance, mean_hd95))
                
                model.train()
    
    # Save final model
    save_final = os.path.join(snapshot_path, 'final_model.pth')
    torch.save(model.state_dict(), save_final)
    logging.info('Training completed. Best performance: %f at iteration %d' % 
                (best_performance, best_iteration))
    
    writer.close()
    return best_performance