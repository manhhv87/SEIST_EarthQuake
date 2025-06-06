#!/bin/bash
#SBATCH --job-name=MT
#SBATCH --partition=dgx-small
#SBATCH --ntasks=1                    # Total number of tasks (processes), equals number of GPUs
#SBATCH --gres=gpu:0                  # Request 4 GPUs
#SBATCH --cpus-per-task=4             # Number of CPU cores per task

# === Environment Setup ===
module load singularity
module load cuda

mkdir -p logs

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# === GPU Utilization Logging ===
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu --format=csv -l 10 > logs/gpu_memory_log.csv &
NSMI_PID=$!

### the command to run
singularity exec --nv $HOME/pytorch/pytorch-1.13.1-cuda11.6-cudnn8-py3.10 python3 main.py \
  --seed 0 \
  --mode "train_test" \
  --model-name "seist_s_dpk" \
  --log-base "./logs" \
  --device "cuda:0" \
  --data "./data/STEAD" \
  --dataset-name "stead" \
  --data-split true \
  --train-size 0.8 \
  --val-size 0.1 \
  --shuffle true \
  --workers 2 \
  --in-samples 8192 \
  --augmentation true \
  --epochs 200 \
  --patience 30 \
  --batch-size 500
  
# === Cleanup ===
kill $NSMI_PID