import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# Directory containing the log files
logs_dir = '/home1/ppatel2025/deep_learning_final_project/final_project_deeplearining/logs'

# Get all CSV files
log_files = glob.glob(os.path.join(logs_dir, '*_training_log.csv'))

# Filter out the finetuned file to avoid duplication
log_files = [f for f in log_files if 'finetuned' not in f]

# Sort files for consistent ordering
log_files.sort()

print(f"Found {len(log_files)} model training logs")

# ========== PLOT 1: Loss Curves (16 subplots) ==========
fig1, axes1 = plt.subplots(4, 4, figsize=(20, 16))
axes1 = axes1.flatten()

for idx, log_file in enumerate(log_files):
    model_name = os.path.basename(log_file).replace('_training_log.csv', '')

    try:
        df = pd.read_csv(log_file)

        # Plot train and val loss
        axes1[idx].plot(df['epoch'], df['train_loss'],
                       label='Train Loss',
                       color='blue',
                       linewidth=2)
        axes1[idx].plot(df['epoch'], df['val_loss'],
                       label='Val Loss',
                       color='red',
                       linewidth=2)

        axes1[idx].set_title(model_name, fontsize=10, fontweight='bold')
        axes1[idx].set_xlabel('Epoch', fontsize=9)
        axes1[idx].set_ylabel('Loss', fontsize=9)
        axes1[idx].legend(fontsize=8)
        axes1[idx].grid(True, alpha=0.3)

        print(f"Plotted loss for: {model_name}")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

fig1.suptitle('Training and Validation Loss - All Models', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save loss plot
loss_output_path = os.path.join(logs_dir, 'all_models_loss_curves.png')
plt.savefig(loss_output_path, dpi=300, bbox_inches='tight')
print(f"\nLoss plot saved to: {loss_output_path}")

# ========== PLOT 2: Accuracy Curves (16 subplots) ==========
fig2, axes2 = plt.subplots(4, 4, figsize=(20, 16))
axes2 = axes2.flatten()

for idx, log_file in enumerate(log_files):
    model_name = os.path.basename(log_file).replace('_training_log.csv', '')

    try:
        df = pd.read_csv(log_file)

        # Plot train and val accuracy
        axes2[idx].plot(df['epoch'], df['train_acc'],
                       label='Train Acc',
                       color='green',
                       linewidth=2)
        axes2[idx].plot(df['epoch'], df['val_acc'],
                       label='Val Acc',
                       color='orange',
                       linewidth=2)

        axes2[idx].set_title(model_name, fontsize=10, fontweight='bold')
        axes2[idx].set_xlabel('Epoch', fontsize=9)
        axes2[idx].set_ylabel('Accuracy', fontsize=9)
        axes2[idx].legend(fontsize=8)
        axes2[idx].grid(True, alpha=0.3)

        print(f"Plotted accuracy for: {model_name}")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

fig2.suptitle('Training and Validation Accuracy - All Models', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save accuracy plot
acc_output_path = os.path.join(logs_dir, 'all_models_accuracy_curves.png')
plt.savefig(acc_output_path, dpi=300, bbox_inches='tight')
print(f"Accuracy plot saved to: {acc_output_path}")

# ========== PLOT 3: F1 Score Curves (16 subplots) - If Available ==========
# Check if any file has F1 scores
sample_df = pd.read_csv(log_files[0])
has_f1 = 'train_f1' in sample_df.columns and 'val_f1' in sample_df.columns

if has_f1:
    fig3, axes3 = plt.subplots(4, 4, figsize=(20, 16))
    axes3 = axes3.flatten()

    for idx, log_file in enumerate(log_files):
        model_name = os.path.basename(log_file).replace('_training_log.csv', '')

        try:
            df = pd.read_csv(log_file)

            # Check if this specific file has F1 scores
            if 'train_f1' in df.columns and 'val_f1' in df.columns:
                # Plot train and val F1 score
                axes3[idx].plot(df['epoch'], df['train_f1'],
                               label='Train F1',
                               color='purple',
                               linewidth=2)
                axes3[idx].plot(df['epoch'], df['val_f1'],
                               label='Val F1',
                               color='magenta',
                               linewidth=2)

                axes3[idx].set_title(model_name, fontsize=10, fontweight='bold')
                axes3[idx].set_xlabel('Epoch', fontsize=9)
                axes3[idx].set_ylabel('Macro F1 Score', fontsize=9)
                axes3[idx].legend(fontsize=8)
                axes3[idx].grid(True, alpha=0.3)

                print(f"Plotted F1 for: {model_name}")
            else:
                # If F1 not available for this file, show empty plot with message
                axes3[idx].text(0.5, 0.5, 'F1 data not available',
                               ha='center', va='center', fontsize=10)
                axes3[idx].set_title(model_name, fontsize=10, fontweight='bold')
        except Exception as e:
            print(f"Error loading {model_name}: {e}")

    fig3.suptitle('Training and Validation Macro F1 Score - All Models', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save F1 plot
    f1_output_path = os.path.join(logs_dir, 'all_models_f1_curves.png')
    plt.savefig(f1_output_path, dpi=300, bbox_inches='tight')
    print(f"F1 plot saved to: {f1_output_path}")
else:
    print("\nF1 scores not found in log files. Skipping F1 plot generation.")

plt.show()
