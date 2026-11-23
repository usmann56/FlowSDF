#!/bin/bash
#SBATCH --job-name=flowsdf-eval-e1-val
#SBATCH --partition=qTRDGPU
#SBATCH -A psy53c17
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/users/mnizam1/aip/FlowSDF/logs/%x-%A_%a.out

# E1 (30 training images, both seeds, final checkpoint) on the 7 validation images,
# once with the sampler from sampler.py (task 0) and once with standard flow-matching Euler (task 1)
set -e
source /data/users2/mnizam1/miniforge3/bin/activate flowsdf
cd /home/users/mnizam1/aip/FlowSDF

SAMPLERS=(official standard)
SAMPLER=${SAMPLERS[$SLURM_ARRAY_TASK_ID]}
python eval_monuseg.py --config cfg/monuseg_e1_train30.yaml --split val \
    --raw-train-dir /data/users2/mnizam1/datasets/raw/monuseg_train \
    --ckpt runs/monuseg/*e1_train30_seed*/models/*0029999.pt \
    --sampler $SAMPLER --out runs/eval/e1_val_$SAMPLER
