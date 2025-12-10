import yaml
import argparse

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from data_and_train import get_model

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_probs, classes, save_path):
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    n_classes = len(classes)
    
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_history(log_file, save_path):
    if not os.path.exists(log_file):
        print(f"Warning: Log file {log_file} not found. Skipping history plot.")
        return

    df = pd.read_csv(log_file)

    # Check if F1 scores are available in the log file
    has_f1 = 'train_f1' in df.columns and 'val_f1' in df.columns

    if has_f1:
        plt.figure(figsize=(18, 5))

        # Loss
        plt.subplot(1, 3, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Accuracy
        plt.subplot(1, 3, 2)
        plt.plot(df['epoch'], df['train_acc'], label='Train Acc')
        plt.plot(df['epoch'], df['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        # F1 Score
        plt.subplot(1, 3, 3)
        plt.plot(df['epoch'], df['train_f1'], label='Train F1')
        plt.plot(df['epoch'], df['val_f1'], label='Val F1')
        plt.xlabel('Epoch')
        plt.ylabel('Macro F1 Score')
        plt.title('Training and Validation F1 Score')
        plt.legend()
    else:
        plt.figure(figsize=(12, 5))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(df['epoch'], df['train_acc'], label='Train Acc')
        plt.plot(df['epoch'], df['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_model_comparison(results_df, save_dir):
    # Bar chart for F1 scores
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='model_name', y='macro_f1', hue='config_id')
    plt.title('Model Comparison - F1 Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison_f1.png'))
    plt.close()

    # Bar chart for Accuracy
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='model_name', y='test_accuracy', hue='config_id')
    plt.title('Model Comparison - Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison_acc.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate Models')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)

    base_path = os.getcwd()
    test_dir = os.path.join(base_path, config['data']['processed_dir'], 'test')
    checkpoints_dir = os.path.join(base_path, 'checkpoints')
    logs_dir = os.path.join(base_path, 'logs')
    results_dir = os.path.join(base_path, 'results')
    viz_dir = os.path.join(base_path, 'visualizations')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    if not os.path.exists(test_dir):
        print("Test data not found. Please run data_and_train.py first.")
        return

    img_size = tuple(config['data']['image_size'])
    
    # Transforms (same as validation)
    test_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=config['data']['num_workers'])
    classes = test_dataset.classes
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Find all BEST checkpoints
    checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith('_BEST.pth')]
    
    results = []

    # Create a map from config ID to dropout rate
    config_map = {str(c['id']): c['dropout'] for c in config['model_configs']}
    
    best_f1 = -1.0
    best_model_info = None

    for ckpt in checkpoints:
        print(f"Evaluating {ckpt}...")
        # Parse model info from filename
        # Format: {model_name}_{pt_str}_config{config_id}_BEST.pth
        parts = ckpt.replace('_BEST.pth', '').split('_')
        
        model_name = parts[0]
        pt_str = parts[1]
        config_id = parts[2].replace('config', '')
        
        pretrained = (pt_str == 'pretrained')
        
        dropout = config_map.get(config_id, 0.5)

        model = get_model(model_name, pretrained, num_classes=len(classes), dropout_rate=dropout)
        
        # Load weights
        ckpt_path = os.path.join(checkpoints_dir, ckpt)
        # Handle DataParallel saving (keys might start with 'module.')
        state_dict = torch.load(ckpt_path)
        
        # If saved with DataParallel, strip 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict)
        model = model.to(device)
        
        y_true, y_pred, y_probs = evaluate_model(model, test_loader, device, len(classes))
        
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        
        try:
            roc_auc = roc_auc_score(label_binarize(y_true, classes=range(len(classes))), y_probs, average='macro', multi_class='ovr')
        except ValueError:
            roc_auc = 0.0 # Handle edge cases
            
        print(f"  Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
        
        results.append({
            'model_name': model_name,
            'pretrained': pretrained,
            'config_id': config_id,
            'test_accuracy': acc,
            'macro_precision': prec,
            'macro_recall': rec,
            'macro_f1': f1,
            'roc_auc_mean': roc_auc
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_info = {
                'model_name': model_name,
                'pretrained': pretrained,
                'config_id': config_id,
                'metrics': {'acc': acc, 'f1': f1, 'auc': roc_auc}
            }
        
        # Plots
        base_name = f"{model_name}_{pt_str}_config{config_id}"
        plot_confusion_matrix(y_true, y_pred, classes, os.path.join(viz_dir, f"cm_{base_name}.png"))
        plot_roc_curve(y_true, y_probs, classes, os.path.join(viz_dir, f"roc_{base_name}.png"))
        
        # Plot training history
        log_file = os.path.join(logs_dir, f"{model_name}_{pt_str}_config{config_id}_training_log.csv")
        plot_training_history(log_file, os.path.join(viz_dir, f"history_{base_name}.png"))

    # Save summary
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, 'model_evaluation_summary.csv'), index=False)
    
    # Comparison plots
    plot_model_comparison(df, viz_dir)
    
    print(f"Evaluation complete. Results saved to {results_dir}")
    
    if best_model_info:
        print("\n" + "="*30)
        print("BEST MODEL CONFIGURATION")
        print("="*30)
        print(f"Model: {best_model_info['model_name']}")
        print(f"Pretrained: {best_model_info['pretrained']}")
        print(f"Config ID: {best_model_info['config_id']}")
        print(f"Metrics: {best_model_info['metrics']}")
        print("="*30)
        
        # Save best model config to file
        with open(os.path.join(results_dir, 'best_model_config.txt'), 'w') as f:
            f.write("BEST MODEL CONFIGURATION\n")
            f.write(f"Model: {best_model_info['model_name']}\n")
            f.write(f"Pretrained: {best_model_info['pretrained']}\n")
            f.write(f"Config ID: {best_model_info['config_id']}\n")
            f.write(f"Metrics: {best_model_info['metrics']}\n")

if __name__ == "__main__":
    main()
