#!/bin/bash
#SBATCH --job-name=flowsdf-eval-final
#SBATCH --partition=qTRDGPU
#SBATCH -A psy53c17
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/users/mnizam1/aip/FlowSDF/logs/%x-%A_%a.out

# Validation eval of the final runs, same settings as e1cont_val (K=25, official sampler, maps saved
# so eval_thresholds.py can pick the threshold per condition). Both seeds go into one folder.
set -e
source /data/users2/mnizam1/miniforge3/bin/activate flowsdf
cd /home/users/mnizam1/aip/FlowSDF

CONDS=(e2 e3 e3b)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
python eval_monuseg.py --config cfg/monuseg_$COND.yaml --split val \
    --raw-train-dir /data/users2/mnizam1/datasets/raw/monuseg_train \
    --ckpt runs/monuseg/*_final_${COND}_seed*/models/*0009999.pt \
    --sampler official --out runs/eval/final_${COND}_val
