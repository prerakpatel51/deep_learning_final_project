import pandas as pd
import matplotlib.pyplot as plt
import os

# Directory containing the log files
logs_dir = '/home1/ppatel2025/deep_learning_final_project/final_project_deeplearining/logs'

# Best model according to best_model_config.txt
best_model_file = 'resnet50_pretrained_config2_training_log.csv'

# Also plot config 3 which has highest test accuracy
best_test_model_file = 'resnet50_pretrained_config3_training_log.csv'

# Configuration details
config2_details = {
    'lr': '1e-4',
    'batch_size': 32,
    'optimizer': 'Adam',
    'dropout': 0.5,
    'weight_decay': '1e-2'
}

config3_details = {
    'lr': '1e-4',
    'batch_size': 64,
    'optimizer': 'RMSprop',
    'dropout': 0.3,
    'weight_decay': '1e-4'
}

# Read the best model data
best_model_path = os.path.join(logs_dir, best_model_file)
df_best = pd.read_csv(best_model_path)

# Read the best test accuracy model data
best_test_path = os.path.join(logs_dir, best_test_model_file)
df_best_test = pd.read_csv(best_test_path)

# Check if F1 scores are available
has_f1_best = 'train_f1' in df_best.columns and 'val_f1' in df_best.columns
has_f1_test = 'train_f1' in df_best_test.columns and 'val_f1' in df_best_test.columns

# ========== PLOT 1: Loss Curves for Best Model (Config 2) ==========
fig1, ax1 = plt.subplots(figsize=(12, 7))

ax1.plot(df_best['epoch'], df_best['train_loss'],
         label='Train Loss',
         color='blue',
         linewidth=2.5,
         marker='o',
         markersize=4)
ax1.plot(df_best['epoch'], df_best['val_loss'],
         label='Validation Loss',
         color='red',
         linewidth=2.5,
         marker='s',
         markersize=4)

ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('Loss', fontsize=14)

# Add configuration details as text box
# config_text = f"Config 2: LR={config2_details['lr']}, Batch={config2_details['batch_size']}, " \
#               f"Optimizer={config2_details['optimizer']}, Dropout={config2_details['dropout']}, " \
#               f"Weight Decay={config2_details['weight_decay']}"
# ax1.text(0.5, 1.08, config_text, transform=ax1.transAxes,
#          fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax1.set_title(f'ResNet50 Pretrained Config 2 - Loss Curves\n(Best Overall Model) Config 2: LR={config2_details['lr']}, Batch={config2_details['batch_size']}, Optimizer={config2_details['optimizer']}, Dropout={config2_details['dropout']},Weight Decay={config2_details['weight_decay']}',
              fontsize=16, fontweight='bold', pad=30)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
plt.tight_layout()

# Save loss plot
loss_output = os.path.join(logs_dir, 'best_model_loss_curve.png')
plt.savefig(loss_output, dpi=300, bbox_inches='tight')
print(f"Best model loss plot saved to: {loss_output}")

# ========== PLOT 2: Accuracy Curves for Best Model (Config 2) ==========
fig2, ax2 = plt.subplots(figsize=(12, 7))

ax2.plot(df_best['epoch'], df_best['train_acc'],
         label='Train Accuracy',
         color='green',
         linewidth=2.5,
         marker='o',
         markersize=4)
ax2.plot(df_best['epoch'], df_best['val_acc'],
         label='Validation Accuracy',
         color='orange',
         linewidth=2.5,
         marker='s',
         markersize=4)

ax2.set_xlabel('Epoch', fontsize=14)
ax2.set_ylabel('Accuracy', fontsize=14)

# Add configuration details as text box
# config_text = f"Config 2: LR={config2_details['lr']}, Batch={config2_details['batch_size']}, " \
#               f"Optimizer={config2_details['optimizer']}, Dropout={config2_details['dropout']}, " \
#               f"Weight Decay={config2_details['weight_decay']}"
# ax2.text(0.5, 1.08, config_text, transform=ax2.transAxes,
#          fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

ax2.set_title(f'ResNet50 Pretrained Config 2 - Accuracy Curves\n(Best Overall Model) Config 2: LR={config2_details['lr']}, Batch={config2_details['batch_size']},Optimizer={config2_details['optimizer']}, Dropout={config2_details['dropout']},Weight Decay={config2_details['weight_decay']}',
              fontsize=16, fontweight='bold', pad=30)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)
plt.tight_layout()

# Save accuracy plot
acc_output = os.path.join(logs_dir, 'best_model_accuracy_curve.png')
plt.savefig(acc_output, dpi=300, bbox_inches='tight')
print(f"Best model accuracy plot saved to: {acc_output}")

# ========== BONUS: PLOT 3: Loss Curves for Highest Test Accuracy Model (Config 3) ==========
fig3, ax3 = plt.subplots(figsize=(12, 7))

ax3.plot(df_best_test['epoch'], df_best_test['train_loss'],
         label='Train Loss',
         color='blue',
         linewidth=2.5,
         marker='o',
         markersize=4)
ax3.plot(df_best_test['epoch'], df_best_test['val_loss'],
         label='Validation Loss',
         color='red',
         linewidth=2.5,
         marker='s',
         markersize=4)

ax3.set_xlabel('Epoch', fontsize=14)
ax3.set_ylabel('Loss', fontsize=14)

# Add configuration details as text box
# config_text = f"Config 3: LR={config3_details['lr']}, Batch={config3_details['batch_size']}, " \
            #   f"Optimizer={config3_details['optimizer']}, Dropout={config3_details['dropout']}, " \
            #   f"Weight Decay={config3_details['weight_decay']}"
# ax3.text(.5, 1.08, config_text, transform=ax3.transAxes,
        #  fontsize=11, ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax3.set_title(f'ResNet50 Pretrained Config 3 - Loss Curves\n(Highest Test Accuracy: 81.44%) Config 3: LR={config3_details['lr']}, Batch={config3_details['batch_size']}, Optimizer={config3_details['optimizer']}, Dropout={config3_details['dropout']}, Weight Decay={config3_details['weight_decay']}',
              fontsize=14, fontweight='bold', pad=  10)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
plt.tight_layout()

# Save loss plot
loss_output_test = os.path.join(logs_dir, 'highest_test_acc_model_loss_curve.png')
plt.savefig(loss_output_test, dpi=300, bbox_inches='tight')
print(f"Highest test accuracy model loss plot saved to: {loss_output_test}")

# ========== PLOT 4: Accuracy Curves for Highest Test Accuracy Model (Config 3) ==========
fig4, ax4 = plt.subplots(figsize=(12, 7))

ax4.plot(df_best_test['epoch'], df_best_test['train_acc'],
         label='Train Accuracy',
         color='green',
         linewidth=2.5,
         marker='o',
         markersize=4)
ax4.plot(df_best_test['epoch'], df_best_test['val_acc'],
         label='Validation Accuracy',
         color='orange',
         linewidth=2.5,
         marker='s',
         markersize=4)

ax4.set_xlabel('Epoch', fontsize=14)
ax4.set_ylabel('Accuracy', fontsize=14)

# Add configuration details as text box
# config_text = f"Config 3: LR={config3_details['lr']}, Batch={config3_details['batch_size']}, " \
            #   f"Optimizer={config3_details['optimizer']}, Dropout={config3_details['dropout']}, " \
            #   f"Weight Decay={config3_details['weight_decay']}"
# ax4.text(0.5, 1.08, config_text, transform=ax4.transAxes,
        #  fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

ax4.set_title(f'ResNet50 Pretrained Config 3 - Accuracy Curves\n(Highest Test Accuracy: 81.44%)Config 3: LR={config3_details['lr']}, Batch={config3_details['batch_size']},Optimizer={config3_details['optimizer']}, Dropout={config3_details['dropout']},Weight Decay={config3_details['weight_decay']} ',
              fontsize=16, fontweight='bold', pad=30)
ax4.legend(fontsize=12)
ax4.grid(True, alpha=0.3)
plt.tight_layout()

# Save accuracy plot
acc_output_test = os.path.join(logs_dir, 'highest_test_acc_model_accuracy_curve.png')
plt.savefig(acc_output_test, dpi=300, bbox_inches='tight')
print(f"Highest test accuracy model accuracy plot saved to: {acc_output_test}")

# ========== PLOT 5: F1 Score Curves for Best Model (Config 2) - If Available ==========
if has_f1_best:
    fig5, ax5 = plt.subplots(figsize=(12, 7))

    ax5.plot(df_best['epoch'], df_best['train_f1'],
             label='Train F1 Score',
             color='purple',
             linewidth=2.5,
             marker='o',
             markersize=4)
    ax5.plot(df_best['epoch'], df_best['val_f1'],
             label='Validation F1 Score',
             color='magenta',
             linewidth=2.5,
             marker='s',
             markersize=4)

    ax5.set_xlabel('Epoch', fontsize=14)
    ax5.set_ylabel('Macro F1 Score', fontsize=14)

    ax5.set_title(f'ResNet50 Pretrained Config 2 - F1 Score Curves\n(Best Overall Model) Config 2: LR={config2_details['lr']}, Batch={config2_details['batch_size']}, Optimizer={config2_details['optimizer']}, Dropout={config2_details['dropout']}, Weight Decay={config2_details['weight_decay']}',
                  fontsize=16, fontweight='bold', pad=30)
    ax5.legend(fontsize=12)
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save F1 plot
    f1_output = os.path.join(logs_dir, 'best_model_f1_curve.png')
    plt.savefig(f1_output, dpi=300, bbox_inches='tight')
    print(f"Best model F1 plot saved to: {f1_output}")

# ========== PLOT 6: F1 Score Curves for Highest Test Accuracy Model (Config 3) - If Available ==========
if has_f1_test:
    fig6, ax6 = plt.subplots(figsize=(12, 7))

    ax6.plot(df_best_test['epoch'], df_best_test['train_f1'],
             label='Train F1 Score',
             color='purple',
             linewidth=2.5,
             marker='o',
             markersize=4)
    ax6.plot(df_best_test['epoch'], df_best_test['val_f1'],
             label='Validation F1 Score',
             color='magenta',
             linewidth=2.5,
             marker='s',
             markersize=4)

    ax6.set_xlabel('Epoch', fontsize=14)
    ax6.set_ylabel('Macro F1 Score', fontsize=14)

    ax6.set_title(f'ResNet50 Pretrained Config 3 - F1 Score Curves\n(Highest Test Accuracy: 81.44%) Config 3: LR={config3_details['lr']}, Batch={config3_details['batch_size']}, Optimizer={config3_details['optimizer']}, Dropout={config3_details['dropout']}, Weight Decay={config3_details['weight_decay']}',
                  fontsize=14, fontweight='bold', pad=10)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save F1 plot
    f1_output_test = os.path.join(logs_dir, 'highest_test_acc_model_f1_curve.png')
    plt.savefig(f1_output_test, dpi=300, bbox_inches='tight')
    print(f"Highest test accuracy model F1 plot saved to: {f1_output_test}")

print("\nSummary:")
print(f"Best Overall Model (Config 2) - Test Acc: 76.95%, AUC: 0.958")
print(f"Highest Test Acc Model (Config 3) - Test Acc: 81.44%, AUC: 0.961")

plt.show()
