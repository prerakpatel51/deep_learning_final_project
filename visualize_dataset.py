import yaml
import argparse

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import random

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_class_distribution(data_dir, title, save_path):
    # ... (remains same)
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    counts = []
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        counts.append(count)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=classes, y=counts)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved distribution plot to {save_path}")

def plot_samples(data_dir, save_dir, img_size, augment=False):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    if augment:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
        ])
        suffix = "after_aug"
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        suffix = "before_aug"

    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        images = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            continue

        selected = random.sample(images, min(16, len(images)))
        
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle(f"Class: {cls} ({suffix})")
        
        for i, ax in enumerate(axes.flat):
            if i < len(selected):
                img = Image.open(selected[i]).convert('RGB')
                img_t = transform(img)
                # Convert back to PIL for display
                img_np = img_t.permute(1, 2, 0).numpy()
                if augment: 
                    pass
                ax.imshow(np.clip(img_np, 0, 1))
                ax.axis('off')
            else:
                ax.axis('off')
        
        save_path = os.path.join(save_dir, f"samples_{suffix}_class_{cls}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved sample grid to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize Dataset')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    
    base_path = os.getcwd()
    data_path = os.path.join(base_path, config['data']['processed_dir'])
    train_dir = os.path.join(data_path, 'train')
    viz_dir = os.path.join(base_path, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    if not os.path.exists(train_dir):
        print("Processed data not found. Please run data_and_train.py first.")
        return

    # 1. Class distribution
    plot_class_distribution(train_dir, "Class Distribution (On Disk / Before Balancing)", 
                            os.path.join(viz_dir, "class_distribution_before.png"))

    # Simulate balanced distribution
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    plt.figure(figsize=(10, 6))
    sns.barplot(x=classes, y=[config['data']['target_per_class']]*len(classes))
    plt.title(f"Class Distribution (After Balancing - Target {config['data']['target_per_class']})")
    plt.xticks(rotation=45)
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "class_distribution_after.png"))
    print("Saved balanced distribution plot.")

    # 2. Sample grids
    img_size = tuple(config['data']['image_size'])
    print("Generating sample grids...")
    plot_samples(train_dir, viz_dir, img_size, augment=False)
    plot_samples(train_dir, viz_dir, img_size, augment=True)

if __name__ == "__main__":
    main()
