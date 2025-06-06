#!/bin/bash
#SBATCH --job-name=MT-MULTI
#SBATCH --partition=dgx-small
#SBATCH --gres=gpu:0                  # Request 1 GPUs
#SBATCH --cpus-per-task=4             # More CPUs to handle two processes
#SBATCH --ntasks=1                    # Total number of tasks (processes), equals number of GPUs

# === Environment Setup ===
module load singularity
module load cuda

mkdir -p logs

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# === GPU Utilization Logging ===
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu --format=csv -l 10 > logs/gpu_memory_log.csv &
NSMI_PID=$!

# === Retry logic for random ports ===
for attempt in {1..5}; do
  export MASTER_PORT=$((12000 + RANDOM % 10000))
  echo "[INFO] Attempt $attempt: Trying MASTER_PORT=$MASTER_PORT"

  singularity exec --nv $HOME/pytorch/pytorch-1.13.1-cuda11.6-cudnn8-py3.10 \
  torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=$MASTER_PORT \
    main.py \
      --seed 0 \
      --mode "train_test" \
      --model-name "seist_s_dpk" \
      --log-base "./logs" \
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

  status=$?
  if [ $status -eq 0 ]; then
    echo "[INFO] Training finished successfully."
    break
  else
    echo "[WARN] Failed on port $MASTER_PORT (exit code $status), retrying..."
    sleep 2
  fi
done

# === Cleanup ===
kill $NSMI_PID