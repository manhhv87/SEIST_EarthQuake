#!/bin/bash
#SBATCH --job-name=MT
#SBATCH --partition=gpu
##SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --error=%x.%j.err

### init virtual environment if needed
source ~/anaconda3/etc/profile.d/conda.sh
conda activate eq

module load singularity

### the command to run
singularity exec --nv $HOME/pytorch/pytorch-1.13.1-cuda11.6-cudnn8-py3.10 python3 main.py \
  --seed 0 \
  --mode "train_test" \
  --model-name "seist_m_dpk" \
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