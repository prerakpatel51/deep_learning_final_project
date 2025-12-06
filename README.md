# Skin Lesion Classification using Deep Learning

A comprehensive deep learning project for automated classification of skin lesions into 7 different categories using Convolutional Neural Networks (CNNs). This project implements and compares ResNet50 and VGG16 architectures with transfer learning for medical image classification.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Configuration](#configuration)
- [Visualizations](#visualizations)
- [Contributing](#contributing)

## Overview

This project addresses the critical problem of automated skin lesion classification to assist in early detection of skin cancer and other dermatological conditions. The system classifies skin lesions into 7 categories:

1. **Nevus (nv)** - Melanocytic nevi (moles)
2. **Melanoma (mel)** - Malignant melanoma
3. **Benign keratosis (bkl)** - Benign keratosis-like lesions
4. **Basal cell carcinoma (bcc)** - Basal cell carcinoma
5. **Actinic keratosis (akiec)** - Actinic keratoses
6. **Vascular lesions (vasc)** - Vascular lesions
7. **Dermatofibroma (df)** - Dermatofibroma

### Key Achievements
- **Best Model Performance**: 76.95% accuracy, 0.698 F1 score, 0.958 ROC-AUC
- **Architecture**: ResNet50 with ImageNet pretraining
- **Systematic Evaluation**: 16 different model configurations tested
- **Comprehensive Analysis**: Extensive visualizations and metrics

## Dataset

**Source**: HAM10000 (Human Against Machine with 10000 training images)
- **Kaggle**: `farjanakabirsamanta/skin-cancer-dataset`
- **Total Images**: 10,015 dermatoscopic images
- **Image Format**: JPG/JPEG
- **Resolution**: Resized to 360x360 pixels

### Data Split
- Training: 80%
- Validation: 10%
- Test: 10%

### Class Balancing
The dataset exhibits severe class imbalance. A custom balancing strategy was implemented:
- Target: 400 samples per class
- Oversampling for minority classes
- Undersampling for majority classes

## Features

- **Multiple Architectures**: ResNet50 and VGG16
- **Transfer Learning**: Pretrained ImageNet weights vs training from scratch
- **Data Augmentation**: Random flips, rotation, color jitter
- **Hyperparameter Search**: 4 different configurations per model
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Comprehensive Logging**: Training metrics saved to CSV
- **Rich Visualizations**: Confusion matrices, ROC curves, training histories
- **GPU Support**: CUDA acceleration with multi-GPU support via DataParallel

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 10GB+ disk space for dataset

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd final_project_deeplearining
```

2. Install required packages:
```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn pillow pyyaml tqdm
pip install kagglehub
```

3. Configure Kaggle credentials (for dataset download):
```bash
# Place kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Usage

### 1. Download and Prepare Data

The data download and organization happens automatically during training:

```bash
python data_and_train.py --config config.yaml
```

### 2. Train Models

Train all model configurations specified in `config.yaml`:

```bash
python data_and_train.py --config config.yaml
```

Options:
- `--debug`: Run in debug mode with reduced dataset and epochs
- `--target_per_class <N>`: Override target samples per class
- `--num_epochs <N>`: Override number of epochs

Example:
```bash
python data_and_train.py --config config.yaml --num_epochs 30
```

### 3. Visualize Dataset

Generate dataset visualizations (class distribution, sample images):

```bash
python visualize_dataset.py --config config.yaml
```

### 4. Evaluate Models

Evaluate all trained models on test set and generate visualizations:

```bash
python evaluate_models.py --config config.yaml
```

This will:
- Load all best model checkpoints
- Evaluate on test set
- Generate confusion matrices, ROC curves, training histories
- Save results to `results/model_evaluation_summary.csv`
- Identify best performing model

### 5. Plot Training Curves

Visualize training/validation curves for all models:

```bash
python plot_training_curves.py
```

### 6. Analyze Best Model

Generate detailed visualizations for the best model:

```bash
python plot_best_model.py
```

## Project Structure

```
final_project_deeplearining/
│
├── config.yaml                 # Central configuration file
├── data_and_train.py          # Main training script
├── evaluate_models.py         # Model evaluation script
├── visualize_dataset.py       # Dataset visualization script
├── plot_training_curves.py    # Training curve plotting
├── plot_best_model.py         # Best model analysis
├── train_job.sh               # SLURM job script
├── run_commands.txt           # Command reference
│
├── data/                      # Dataset directory
│   └── processed/             # Organized train/val/test splits
│       ├── train/
│       ├── val/
│       └── test/
│
├── checkpoints/               # Saved model weights
│   ├── *_BEST.pth            # Best models (validation accuracy)
│   └── *_epoch*.pth          # Epoch checkpoints
│
├── logs/                      # Training logs
│   └── *_training_log.csv    # CSV logs with epoch metrics
│
├── results/                   # Evaluation results
│   ├── model_evaluation_summary.csv
│   └── best_model_config.txt
│
├── visualizations/            # Generated plots
│   ├── class_distribution_*.png
│   ├── samples_*.png
│   ├── cm_*.png              # Confusion matrices
│   ├── roc_*.png             # ROC curves
│   ├── history_*.png         # Training histories
│   └── model_comparison_*.png
│
├── PROJECT_REPORT.txt         # Comprehensive project report
└── README.md                  # This file
```

## Results

### Best Model Configuration

**Model**: ResNet50 (Pretrained)
**Config ID**: 2

**Hyperparameters**:
- Learning Rate: 1e-4
- Batch Size: 32
- Optimizer: Adam
- Dropout: 0.5
- Weight Decay: 1e-2

**Performance Metrics**:
- Test Accuracy: **76.95%**
- Macro F1 Score: **0.698**
- Macro Precision: 0.713
- Macro Recall: 0.729
- Mean ROC-AUC: **0.958**

### Top 5 Models

| Rank | Model | Pretrained | Config | Accuracy | F1 Score | ROC-AUC |
|------|-------|------------|--------|----------|----------|---------|
| 1 | ResNet50 | Yes | 2 | 76.95% | 0.698 | 0.958 |
| 2 | VGG16 | Yes | 4 | 79.44% | 0.673 | 0.955 |
| 3 | ResNet50 | Yes | 3 | 81.44% | 0.667 | 0.961 |
| 4 | ResNet50 | Yes | 4 | 77.25% | 0.658 | 0.949 |
| 5 | VGG16 | Yes | 3 | 77.35% | 0.600 | 0.927 |

### Key Findings

1. **Transfer Learning is Critical**: Pretrained models vastly outperformed models trained from scratch
2. **ResNet50 > VGG16**: ResNet50 generally achieved better balanced performance
3. **Moderate Regularization**: Dropout of 0.5 with weight decay of 1e-2 proved most effective
4. **High Discriminative Ability**: All top models achieved ROC-AUC > 0.92

## Configuration

Edit `config.yaml` to customize:

### Data Configuration
```yaml
data:
  raw_dataset_name: "farjanakabirsamanta/skin-cancer-dataset"
  processed_dir: "data/processed"
  target_per_class: 400
  image_size: [360, 360]
  num_workers: 4
  test_split: 0.1
  val_split: 0.1
```

### Training Configuration
```yaml
training:
  num_epochs: 50
  patience: 10
  seed: 42
  batch_size: 32
```

### Model Configurations
Define custom hyperparameter combinations:
```yaml
model_configs:
  - id: 1
    lr: 1.0e-3
    batch_size: 32
    optimizer: "Adam"
    dropout: 0.5
    weight_decay: 1.0e-4
```

### Models to Train
```yaml
models:
  - name: "resnet50"
    pretrained: true
  - name: "vgg16"
    pretrained: true
```

## Visualizations

The project generates extensive visualizations:

### Dataset Visualizations
- Class distribution (before/after balancing)
- Sample images (original and augmented) for each class

### Training Analysis
- Training/validation loss curves
- Training/validation accuracy curves

### Model Evaluation
- Confusion matrices (per model)
- ROC curves with AUC scores (per model, per class)
- Model comparison bar charts (F1, Accuracy)

All visualizations are saved to the `visualizations/` directory.

## Hardware Requirements

### Minimum
- CPU: Multi-core processor
- RAM: 8GB
- Storage: 15GB
- GPU: Not required but highly recommended

### Recommended
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 20GB SSD
- GPU: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3070, V100)

## Training Time

Approximate training time per model configuration:
- **With GPU**: 30-60 minutes
- **Without GPU**: 6-12 hours

Total training time for all 16 configurations: 8-16 hours (GPU)

## Reproducibility

The project ensures reproducibility through:
- Fixed random seeds (seed=42)
- Deterministic CUDA operations
- Version-controlled configuration
- Comprehensive logging

To reproduce results:
```bash
python data_and_train.py --config config.yaml
python evaluate_models.py --config config.yaml
```

## Future Improvements

- [ ] Test modern architectures (EfficientNet, Vision Transformers)
- [ ] Implement ensemble methods
- [ ] Add Grad-CAM visualization for interpretability
- [ ] Experiment with advanced augmentation (Mixup, CutMix)
- [ ] Implement learning rate scheduling
- [ ] Cross-validation for robust evaluation
- [ ] External dataset validation (ISIC, BCN20000)
- [ ] Model compression for deployment
- [ ] Web/mobile application interface

## License

This project is for educational purposes as part of a Deep Learning course final project.

## Acknowledgments

- HAM10000 dataset creators (Tschandl et al., 2018)
- PyTorch and torchvision teams
- Kaggle community for dataset hosting

## Contact

For questions or collaboration:
- Author: Patel P.
- Project: Deep Learning Final Project
- Date: December 2025

## Citation

If you use this code or approach, please cite:
```
Skin Lesion Classification using Deep Learning
Author: Patel P.
Year: 2025
Institution: [Your Institution]
```

---

**Note**: This is an educational project and should not be used for actual medical diagnosis without proper validation and regulatory approval.
# deep_learning_final_project
