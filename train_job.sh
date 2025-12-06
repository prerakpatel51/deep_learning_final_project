#!/bin/bash
#SBATCH --job-name=skin_cancer_train
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=250GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Define python path
PYTHON_PATH=/home1/ppatel2025/miniconda3/envs/tito_env/bin/python

echo "Starting training job on $(hostname)"
echo "Date: $(date)"

# Run the training script
# Note: srun will inherit the allocation from sbatch
srun $PYTHON_PATH data_and_train.py

echo "Job finished at $(date)"
