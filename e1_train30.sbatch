#!/bin/bash
#SBATCH --job-name=flowsdf-e1-train30
#SBATCH --partition=qTRDGPU
#SBATCH -A psy53c17
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/users/mnizam1/aip/FlowSDF/logs/%x-%A_%a.out

# E1 on 30 training images (7 held out for validation); starting point for E1-cont, E2 and E3
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /data/users2/mnizam1/miniforge3/bin/activate flowsdf
cd /home/users/mnizam1/aip/FlowSDF

nvidia-smi --query-gpu=name,memory.total --format=csv
python main.py --config cfg/monuseg_e1_train30.yaml --seed $SLURM_ARRAY_TASK_ID
