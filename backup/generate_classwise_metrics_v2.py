#!/usr/bin/env python
"""
Generate class-wise metrics CSV for all models.
This version uses delayed imports to avoid torchvision initialization issues.
"""

import yaml
import argparse
import os
import sys

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(model, dataloader, device):
    """
    Evaluate model and return predictions and true labels.
    """
    import torch
    import numpy as np
    from tqdm import tqdm

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

def compute_classwise_metrics(y_true, y_pred, class_names):
    """
    Compute per-class F1 scores and accuracy.

    Returns:
        metrics_dict: Dictionary with class-wise metrics
    """
    import numpy as np
    from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

    # Per-class F1 scores
    f1_per_class = f1_score(y_true, y_pred, labels=range(len(class_names)), average=None, zero_division=0)

    # Per-class precision, recall, F1, and support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=range(len(class_names)),
        average=None,
        zero_division=0
    )

    # Compute per-class accuracy using confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    # Handle division by zero (classes with no samples)
    per_class_accuracy = np.nan_to_num(per_class_accuracy, nan=0.0)

    metrics_dict = {}
    for i, class_name in enumerate(class_names):
        metrics_dict[class_name] = {
            'f1_score': f1_per_class[i],
            'accuracy': per_class_accuracy[i],
            'precision': precision[i],
            'recall': recall[i],
            'support': support[i]
        }

    return metrics_dict

def main():
    # Import heavy libraries only when needed
    import torch
    from torch.utils.data import DataLoader
    import pandas as pd

    # Import torchvision components separately to isolate issues
    try:
        from torchvision import transforms
        from torchvision import datasets
    except RuntimeError as e:
        print(f"Error importing torchvision: {e}")
        print("\nTrying alternative import method...")
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets

    # Import get_model from the training script
    sys.path.insert(0, os.getcwd())
    from data_and_train import get_model

    parser = argparse.ArgumentParser(description='Generate class-wise metrics CSV for all models')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--output_val', type=str, default='classwise_metrics_val.csv', help='Output CSV filename for validation set')
    parser.add_argument('--output_test', type=str, default='classwise_metrics_test.csv', help='Output CSV filename for test set')
    args = parser.parse_args()

    config = load_config(args.config)

    base_path = os.getcwd()
    processed_dir = os.path.join(base_path, config['data']['processed_dir'])
    val_dir = os.path.join(processed_dir, 'val')
    test_dir = os.path.join(processed_dir, 'test')
    checkpoints_dir = os.path.join(base_path, 'checkpoints')
    results_dir = os.path.join(base_path, 'results')

    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(test_dir) or not os.path.exists(val_dir):
        print("Test or validation data not found. Please run data_and_train.py first.")
        return

    img_size = tuple(config['data']['image_size'])

    # Transforms (same as validation)
    eval_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load both validation and test datasets
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transforms)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=config['data']['num_workers'])

    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=config['data']['num_workers'])

    class_names = test_dataset.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Classes: {class_names}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")

    # Find all BEST checkpoints
    checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith('_BEST.pth')]

    if not checkpoints:
        print("No checkpoint files found in checkpoints directory.")
        return

    # Create a map from config ID to dropout rate
    config_map = {str(c['id']): c['dropout'] for c in config['model_configs']}

    # Lists to store results for validation and test sets
    val_results = []
    test_results = []

    for ckpt in checkpoints:
        print(f"Processing {ckpt}...")

        # Parse model info from filename
        # Format: {model_name}_{pt_str}_config{config_id}_BEST.pth
        parts = ckpt.replace('_BEST.pth', '').split('_')

        model_name = parts[0]
        pt_str = parts[1]
        config_id = parts[2].replace('config', '')

        pretrained = (pt_str == 'pretrained')
        pretrained_str = 'Yes' if pretrained else 'No'

        dropout = config_map.get(config_id, 0.5)

        # Load model
        model = get_model(model_name, pretrained, num_classes=len(class_names), dropout_rate=dropout)

        # Load weights
        ckpt_path = os.path.join(checkpoints_dir, ckpt)
        state_dict = torch.load(ckpt_path, map_location=device)

        # Handle DataParallel saving (keys might start with 'module.')
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model = model.to(device)

        # Evaluate on VALIDATION set
        print("  Evaluating on validation set...")
        y_true_val, y_pred_val = evaluate_model(model, val_loader, device)
        classwise_metrics_val = compute_classwise_metrics(y_true_val, y_pred_val, class_names)

        for class_name in class_names:
            metrics = classwise_metrics_val[class_name]
            row = {
                'model_name': model_name.upper(),
                'pretrained': pretrained_str,
                'config_id': config_id,
                'class': class_name,
                'f1_score': round(metrics['f1_score'], 4),
                'accuracy': round(metrics['accuracy'], 4),
                'precision': round(metrics['precision'], 4),
                'recall': round(metrics['recall'], 4),
                'support': int(metrics['support'])
            }
            val_results.append(row)

        # Evaluate on TEST set
        print("  Evaluating on test set...")
        y_true_test, y_pred_test = evaluate_model(model, test_loader, device)
        classwise_metrics_test = compute_classwise_metrics(y_true_test, y_pred_test, class_names)

        for class_name in class_names:
            metrics = classwise_metrics_test[class_name]
            row = {
                'model_name': model_name.upper(),
                'pretrained': pretrained_str,
                'config_id': config_id,
                'class': class_name,
                'f1_score': round(metrics['f1_score'], 4),
                'accuracy': round(metrics['accuracy'], 4),
                'precision': round(metrics['precision'], 4),
                'recall': round(metrics['recall'], 4),
                'support': int(metrics['support'])
            }
            test_results.append(row)

        print(f"  Completed {model_name} (pretrained={pretrained_str}, config={config_id})\n")

    # Create DataFrames and save to CSV
    df_val = pd.DataFrame(val_results)
    df_test = pd.DataFrame(test_results)

    # Sort by model, config, and class for better readability
    df_val = df_val.sort_values(['model_name', 'pretrained', 'config_id', 'class'])
    df_test = df_test.sort_values(['model_name', 'pretrained', 'config_id', 'class'])

    output_path_val = os.path.join(results_dir, args.output_val)
    output_path_test = os.path.join(results_dir, args.output_test)

    df_val.to_csv(output_path_val, index=False)
    df_test.to_csv(output_path_test, index=False)

    print("="*60)
    print("RESULTS SAVED")
    print("="*60)
    print(f"Validation metrics saved to: {output_path_val}")
    print(f"Test metrics saved to: {output_path_test}")
    print(f"Total rows per file: {len(df_val)}")
    print(f"Models evaluated: {len(checkpoints)}")
    print(f"Classes: {len(class_names)}")

    # Print summary statistics for TEST SET
    print("\n" + "="*60)
    print("TEST SET SUMMARY STATISTICS")
    print("="*60)

    # Average F1 score per class across all models
    print("\nAverage F1 Score per Class (across all models):")
    avg_f1_per_class = df_test.groupby('class')['f1_score'].mean().sort_values(ascending=False)
    for class_name, avg_f1 in avg_f1_per_class.items():
        print(f"  {class_name}: {avg_f1:.4f}")

    # Best model per class
    print("\nBest Model for Each Class (by F1 Score):")
    for class_name in class_names:
        class_df = df_test[df_test['class'] == class_name]
        best_row = class_df.loc[class_df['f1_score'].idxmax()]
        print(f"  {class_name}: {best_row['model_name']} (pretrained={best_row['pretrained']}, config={best_row['config_id']}, F1={best_row['f1_score']:.4f})")

    # Print summary statistics for VALIDATION SET
    print("\n" + "="*60)
    print("VALIDATION SET SUMMARY STATISTICS")
    print("="*60)

    # Average F1 score per class across all models
    print("\nAverage F1 Score per Class (across all models):")
    avg_f1_per_class_val = df_val.groupby('class')['f1_score'].mean().sort_values(ascending=False)
    for class_name, avg_f1 in avg_f1_per_class_val.items():
        print(f"  {class_name}: {avg_f1:.4f}")

    # Best model per class
    print("\nBest Model for Each Class (by F1 Score):")
    for class_name in class_names:
        class_df = df_val[df_val['class'] == class_name]
        best_row = class_df.loc[class_df['f1_score'].idxmax()]
        print(f"  {class_name}: {best_row['model_name']} (pretrained={best_row['pretrained']}, config={best_row['config_id']}, F1={best_row['f1_score']:.4f})")

    print("="*60)

if __name__ == "__main__":
    main()
