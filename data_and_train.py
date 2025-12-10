import os
import shutil
import random
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from PIL import Image
import kagglehub
from tqdm import tqdm
import time

# =============================================================================
# CONFIGURATION & REPRODUCIBILITY
# =============================================================================

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# DATA PREPARATION
# =============================================================================

def download_and_organize_data(base_path, config):
    """
    Downloads dataset via kagglehub and organizes it into train/val/test splits.
    Uses HAM10000_metadata.csv to map images to classes.
    """
    print("Downloading dataset...")
    path = kagglehub.dataset_download(config['data']['raw_dataset_name'])
    print(f"Dataset downloaded to: {path}")

    # ... (path finding logic remains same)
    
    csv_path = None
    image_dir = None
    
    for root, dirs, files in os.walk(path):
        if 'HAM10000_metadata.csv' in files:
            csv_path = os.path.join(root, 'HAM10000_metadata.csv')
        
        images_in_dir = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(images_in_dir) > 1000:
            image_dir = root

    if not csv_path or not image_dir:
        raise ValueError(f"Could not locate metadata CSV or image directory in {path}")

    print(f"Found CSV at: {csv_path}")
    print(f"Found images at: {image_dir}")

    processed_path = os.path.join(base_path, config['data']['processed_dir'])
    train_dir = os.path.join(processed_path, 'train')
    val_dir = os.path.join(processed_path, 'val')
    test_dir = os.path.join(processed_path, 'test')

    if os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0:
        print("Processed data already exists. Skipping organization.")
        return train_dir, val_dir, test_dir

    # ... (dx_to_class map remains same)
    dx_to_class = {
        'nv': 'Nevus',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratosis',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    df = pd.read_csv(csv_path)
    
    for cls in dx_to_class.values():
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    # Split data using config ratios
    test_size = config['data']['test_split']
    val_size = config['data']['val_split'] / (1 - test_size) # Adjust val split relative to remaining
    
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=config['training']['seed'], stratify=df['dx'])
    val_df, test_df = train_test_split(temp_df, test_size=val_size, random_state=config['training']['seed'], stratify=temp_df['dx'])

    def copy_images(dataframe, destination_dir):
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Copying to {os.path.basename(destination_dir)}"):
            img_id = row['image_id']
            dx = row['dx']
            cls_name = dx_to_class[dx]
            
            src_file = os.path.join(image_dir, f"{img_id}.jpg")
            if not os.path.exists(src_file):
                src_file = os.path.join(image_dir, f"{img_id}.jpeg")
            if not os.path.exists(src_file):
                src_file = os.path.join(image_dir, f"{img_id}.png")
                
            if os.path.exists(src_file):
                shutil.copy(src_file, os.path.join(destination_dir, cls_name, os.path.basename(src_file)))
            else:
                print(f"Warning: Image {img_id} not found.")

    copy_images(train_df, train_dir)
    copy_images(val_df, val_dir)
    copy_images(test_df, test_dir)
            
    print("Data organization complete.")
    return train_dir, val_dir, test_dir

class BalancedDataset(Dataset):
    """
    A dataset wrapper that balances the classes to a target number of samples.
    - If count > target: Randomly subsample.
    - If count < target: Randomly oversample.
    """
    def __init__(self, root_dir, transform=None, target_per_class=400):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        self.targets = []

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            images = [os.path.join(cls_path, img) for img in os.listdir(cls_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            count = len(images)
            if count == 0:
                continue
                
            if count > target_per_class:
                # Subsample
                selected = random.sample(images, target_per_class)
            else:
                # Oversample
                selected = images + random.choices(images, k=target_per_class - count)
            
            for img_path in selected:
                self.samples.append((img_path, self.class_to_idx[cls]))
                self.targets.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, target

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def get_model(model_name, pretrained, num_classes, dropout_rate=0.5):
    if model_name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_name == 'vgg16':
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)
        num_ftrs = model.classifier[6].in_features
        # Replace the last layer
        model.classifier[6] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs, patience, checkpoint_prefix, log_file):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    epochs_no_improve = 0
    
    log_data = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=phase, leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Collect predictions and labels for F1 calculation
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average='macro')

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')

            if phase == 'train':
                train_loss = epoch_loss
                train_acc = epoch_acc.item()
                train_f1 = epoch_f1
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc.item()
                val_f1 = epoch_f1

        # Logging
        log_data.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1
        })
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, f"{checkpoint_prefix}_epoch{epoch}.pth")

        # Deep copy the model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            torch.save(model.state_dict(), f"{checkpoint_prefix}_BEST.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Save logs
    df = pd.DataFrame(log_data)
    df.to_csv(log_file, index=False)

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Skin Cancer Classification Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (fast)')
    parser.add_argument('--target_per_class', type=int, help='Target samples per class (overrides config)')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs (overrides config)')
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Override config with CLI args if provided
    if args.target_per_class:
        config['data']['target_per_class'] = args.target_per_class
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs

    set_seed(config['training']['seed'])
    
    base_path = os.getcwd()
    train_dir, val_dir, test_dir = download_and_organize_data(base_path, config)

    # Ensure checkpoints and logs directories exist
    os.makedirs(os.path.join(base_path, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'logs'), exist_ok=True)

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
        print("DEBUG MODE: Using small subset of data")
        train_dataset.samples = train_dataset.samples[:100]
        val_dataset.samples = val_dataset.samples[:50]
        config['training']['num_epochs'] = 2

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['data']['num_workers']),
        'val': DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    configs = config['model_configs']
    models_to_train = config['models']

    for model_cfg in models_to_train:
        model_name = model_cfg['name']
        pretrained = model_cfg['pretrained']
        
        for run_config in configs:
            print(f"\nTraining {model_name} (Pretrained={pretrained}) - Config {run_config['id']}")
            
            curr_dataloaders = {
                'train': DataLoader(train_dataset, batch_size=run_config['batch_size'], shuffle=True, num_workers=config['data']['num_workers']),
                'val': DataLoader(val_dataset, batch_size=run_config['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
            }

            model = get_model(model_name, pretrained, num_classes=7, dropout_rate=run_config['dropout'])
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(device)

            if run_config['optimizer'] == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=run_config['lr'], weight_decay=run_config['weight_decay'])
            elif run_config['optimizer'] == 'RMSprop':
                optimizer = optim.RMSprop(model.parameters(), lr=run_config['lr'], weight_decay=run_config['weight_decay'])
            
            criterion = nn.CrossEntropyLoss()

            pt_str = "pretrained" if pretrained else "scratch"
            checkpoint_prefix = f"checkpoints/{model_name}_{pt_str}_config{run_config['id']}"
            log_file = f"logs/{model_name}_{pt_str}_config{run_config['id']}_training_log.csv"

            train_model(
                model, curr_dataloaders, criterion, optimizer, device,
                num_epochs=config['training']['num_epochs'],
                patience=config['training']['patience'],
                checkpoint_prefix=checkpoint_prefix,
                log_file=log_file
            )

if __name__ == "__main__":
    main()
