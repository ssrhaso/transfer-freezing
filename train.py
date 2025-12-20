import argparse
from networkx import config
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# CUSTOM MODULES
from scripts.preprocess_data import GTZANDataset
from scripts.models import AudioResNet50, AudioViT

def load_config(
    config_path: str,
):
    """ LOAD HYPERPARAMETERS FROM YAML FILE """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(
    seed : int,
):
    """ SET RANDOM SEED FOR REPRODUCIBILITY """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    
def train_one(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    """TRAIN FOR ONE EPOCH"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() 
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100.*correct/total

def evaluate(
    model,
    loader,
    criterion,
    device,
):
    """ EVALUATE MODEL ON VALIDATION/TEST SET """
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() 
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100.*correct/total


def main():
    # CL ARGUMENT PARSING
    
    parser = argparse.ArgumentParser(description = "Train GTZAN Genre Classifier")
    parser.add_argument('--config', type = str, required = True, default = 'config.yaml', help = 'Path to config YAML file')
    parser.add_argument('--model', type = str, required = True, choices = ['resnet50', 'vit_b_16'], help = 'Model architecture')
    parser.add_argument('--freeze_mode', type = str, help = 'Layer freezing strategy')
    args = parser.parse_args()
    
    # LOAD CONFIG
    cfg = load_config(args.config)
    
    # OVERRIDE WITH CL ARGS
    model_name = args.model if args.model else cfg['experiment']['model_arch']
    freeze_mode = args.freeze_mode if args.freeze_mode else cfg['experiment']['freeze_mode']
    
    # SETUP ENV
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nEXPERIMENT: \n          MODEL :{model_name} | FREEZE MODE: {freeze_mode} | DEVICE: {device}\n")
    
    set_seed(cfg['project']['seed'])
    os.makedirs(cfg['project']['output_dir'], exist_ok = True)
    
    # DATA LOADERS
    print(" LOADING DATSETS")
    train_ds = GTZANDataset(cfg['data']['root'], split = 'train', argument = cfg['augmentation']['enabled'])
    val_ds = GTZANDataset(cfg['data']['root'], split = 'val', argument = False)
    test_ds = GTZANDataset(cfg['data']['root'], split = 'test', argument = False)
    
    train_loader = DataLoader(
        train_ds,
        batch_size = cfg['data']['batch_size'],
        shuffle = True,
        num_workers = cfg['data']['num_workers'],
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size = cfg['data']['batch_size'],
        shuffle = False,
        num_workers = cfg['data']['num_workers'],
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size = cfg['data']['batch_size'],
        shuffle = False,
        num_workers = cfg['data']['num_workers'],
    )
    
    print(" LOADED DATASETS: ")
    print(f"TRAIN : {len(train_ds)} | VAL : {len(val_ds)} | TEST : {len(test_ds)}\n")
    
    
    
    
    

    

if __name__ == "__main__":
    main()