#!/bin/bash
#SBATCH --job-name=flowsdf-e1cont
#SBATCH --partition=qTRDGPU
#SBATCH -A psy53c17
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/users/mnizam1/aip/FlowSDF/logs/%x-%A_%a.out

# E1-cont: original loss, 10k more epochs from the E1 (30 images) checkpoint at 20k epochs.
# Control for E2/E3, which branch from the same checkpoints with the same budget.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /data/users2/mnizam1/miniforge3/bin/activate flowsdf
cd /home/users/mnizam1/aip/FlowSDF

SEED=$SLURM_ARRAY_TASK_ID
INIT=$(ls runs/monuseg/*_monuseg_e1_train30_seed${SEED}/models_sanity/*0020000.pt)
echo "seed $SEED, init $INIT"
python main.py --config cfg/monuseg_e1cont.yaml --seed $SEED --init-ckpt "$INIT"
