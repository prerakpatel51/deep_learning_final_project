import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import yaml
import time
import pandas as pd
from data_and_train import get_model, BalancedDataset, train_model, set_seed, download_and_organize_data

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Fine-tune Best Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the best model checkpoint')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for fine-tuning')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to fine-tune')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config['training']['seed'])

    base_path = os.getcwd()
    # Ensure data is ready (should be if we are fine-tuning)
    # We can just get paths directly since we assume data_and_train ran
    processed_path = os.path.join(base_path, config['data']['processed_dir'])
    train_dir = os.path.join(processed_path, 'train')
    val_dir = os.path.join(processed_path, 'val')

    if not os.path.exists(train_dir):
        print("Data not found. Please run data_and_train.py first.")
        return

    img_size = tuple(config['data']['image_size'])

    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = BalancedDataset(train_dir, transform=train_transforms, target_per_class=config['data']['target_per_class'])
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    if args.debug:
        print("DEBUG MODE: Using small subset")
        train_dataset.samples = train_dataset.samples[:100]
        val_dataset.samples = val_dataset.samples[:50]
        args.epochs = 2

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['data']['num_workers']),
        'val': DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parse model info from checkpoint filename to reconstruct model
    # Filename format expected: {model_name}_{pt_str}_config{config_id}_BEST.pth
    ckpt_filename = os.path.basename(args.checkpoint)
    parts = ckpt_filename.replace('_BEST.pth', '').split('_')
    
    model_name = parts[0]
    pt_str = parts[1]
    config_id = parts[2].replace('config', '')
    pretrained = (pt_str == 'pretrained')
    
    # Get dropout from config
    model_config = next((c for c in config['model_configs'] if str(c['id']) == config_id), None)
    dropout = model_config['dropout'] if model_config else 0.5

    print(f"Loading {model_name} (Pretrained={pretrained}) from {args.checkpoint}")
    
    model = get_model(model_name, pretrained, num_classes=len(train_dataset.classes), dropout_rate=dropout)
    
    # Load weights
    state_dict = torch.load(args.checkpoint)
    # Handle DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer for fine-tuning (lower LR)
    # We use Adam by default for fine-tuning or we could stick to the original optimizer.
    # Let's use Adam with the specified low LR.
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    checkpoint_prefix = f"checkpoints/{model_name}_{pt_str}_config{config_id}_finetuned"
    log_file = f"logs/{model_name}_{pt_str}_config{config_id}_finetuned_training_log.csv"

    print(f"Starting fine-tuning for {args.epochs} epochs with LR={args.lr}...")
    
    train_model(
        model, dataloaders, criterion, optimizer, device,
        num_epochs=args.epochs,
        patience=config['training']['patience'],
        checkpoint_prefix=checkpoint_prefix,
        log_file=log_file
    )

if __name__ == "__main__":
    main()
