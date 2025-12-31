import argparse
from networkx import config
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# CUSTOM MODULES
from dataset import GTZANDataset
from scripts.models import AudioResNet50, AudioViT


def load_config(config_path: str):
    """ LOAD HYPERPARAMETERS FROM YAML FILE """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """ SET RANDOM SEED FOR REPRODUCIBILITY """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one(model, loader, criterion, optimizer, device):
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


def evaluate(model, loader, criterion, device):
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
    parser = argparse.ArgumentParser(description="Train GTZAN Genre Classifier")
    parser.add_argument('--seed', type=int, default=None, help='Override random seed from config')
    parser.add_argument('--config', type=str, required=True, default='config.yaml', help='Path to config YAML file')
    parser.add_argument('--model', type=str, required=True, help='Model label (e.g. vit_b_16, vit_b_16_aug)')
    
    parser.add_argument('--freeze_mode', type=str, help='Layer freezing strategy')
    args = parser.parse_args()
    
    # LOAD CONFIG
    cfg = load_config(args.config)
    
    # OVERRIDE WITH CL ARGS
    model_name = args.model if args.model else cfg['experiment']['model_arch']
    freeze_mode = args.freeze_mode if args.freeze_mode else cfg['experiment']['freeze_mode']
    seed = args.seed if args.seed is not None else cfg['project']['seed']
    set_seed(seed)
    
    # Determine base architecture from the name string
    # This allows 'vit_b_16_aug' to still load the ViT class
    if 'resnet' in model_name:
        base_arch = 'resnet50'
    elif 'vit' in model_name:
        base_arch = 'vit_b_16'
    else:
        raise ValueError(f"Could not determine architecture from model name: {model_name}")

    # SETUP ENV
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nEXPERIMENT: \n MODEL LABEL: {model_name} (Arch: {base_arch}) | FREEZE MODE: {freeze_mode} | SEED: {seed}\n")
    os.makedirs(cfg['project']['output_dir'], exist_ok=True)
    
    # Pass SpecAugment params to Dataset (connects to your dataset.py)
    print(" LOADING DATASETS")
    
    train_ds = GTZANDataset(
        cfg['data']['root'], 
        split='train', 
        augment=cfg['augmentation']['enabled'], # This reads true/false from config
        time_mask=cfg['augmentation'].get('spec_augment', {}).get('time_mask', 0),
        freq_mask=cfg['augmentation'].get('spec_augment', {}).get('freq_mask', 0),
        time_shift=cfg['augmentation'].get('time_shift', 0)
    )
    
    val_ds = GTZANDataset(cfg['data']['root'], split='val', augment=False)
    test_ds = GTZANDataset(cfg['data']['root'], split='test', augment=False)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['data']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['data']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg['data']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
    )
    
    print("LOADED DATASETS : ")
    print(f"TRAIN : {len(train_ds)} | VAL : {len(val_ds)} | TEST : {len(test_ds)}\n")
    
    print("\n BUILDING MODEL...")
    sys.stdout.flush()

    # Initialize based on detected base_arch
    if base_arch == 'resnet50':
        print("   - Loading ResNet50 architecture...")
        sys.stdout.flush()
        model = AudioResNet50(freeze_mode=freeze_mode, num_classes=cfg['data']['num_classes'])
    elif base_arch == 'vit_b_16':
        print("   - Loading ViT-B/16 architecture...")
        sys.stdout.flush()
        model = AudioViT(freeze_mode=freeze_mode, num_classes=cfg['data']['num_classes'])
    
    print("   MODEL LOADED")
    sys.stdout.flush()
    
    # Move model to GPU
    model = model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TRAINABLE PARAMETERS {trainable_params:,}\n")
    
    # Setup Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['training']['learning_rate'],
        momentum=cfg['training']['momentum']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['training']['epochs']
    )

    # Training Loop
    print(f"TRAINING FOR {cfg['training']['epochs']} epochs...\n")
    best_val_acc = 0.0
    
    for epoch in range(cfg['training']['epochs']):
        train_loss, train_acc = train_one(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        
        print(f"Epoch [{epoch+1:2d}/{cfg['training']['epochs']}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Uses 'model_name' (e.g. vit_b_16_aug) for directory
            ckpt_dir = os.path.join(cfg['project']['output_dir'], 'checkpoints', model_name, freeze_mode)
            os.makedirs(ckpt_dir, exist_ok=True)
            
            save_path = os.path.join(ckpt_dir, f"best_{model_name}_seed{seed}_freeze{freeze_mode}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"   BEST MODEL SAVED (Val Acc: {val_acc:.2f}%)")

    # Test evaluation
    print("\nTEST SET EVALUATION:")
    best_ckpt_path = os.path.join(cfg['project']['output_dir'], 'checkpoints', model_name, freeze_mode, f"best_{model_name}_seed{seed}_freeze{freeze_mode}.pth")
    model.load_state_dict(torch.load(best_ckpt_path))

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
      
    print(f"\nRESULTS:")
    print(f"   Model Label: {model_name}")
    print(f"   Freeze Mode: {freeze_mode}")
    print(f"   Test Accuracy: {test_acc:.2f}%\n")
        
    # Log results
    import csv
    from datetime import datetime
    
    # Create results directory
    results_dir = os.path.join(cfg['project']['output_dir'], "results")
    os.makedirs(results_dir, exist_ok=True)
    summary_csv = os.path.join(results_dir, "results_summary.csv")

    # Write header if file doesn't exist
    file_exists = os.path.isfile(summary_csv)
    with open(summary_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model", "freeze_mode", "seed", "best_val_acc", "test_acc"])
        
        writer.writerow([
            datetime.now().isoformat(), 
            model_name, # Saves as 'vit_b_16_aug'
            freeze_mode, 
            seed, 
            f"{best_val_acc:.2f}", 
            f"{test_acc:.2f}"
        ])
    print(f"Results saved to {summary_csv}")

if __name__ == "__main__":
    main()
