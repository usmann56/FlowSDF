#!/bin/bash
#SBATCH --job-name=flowsdf-tune-e3
#SBATCH --partition=qTRDGPU
#SBATCH -A psy53c17
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/users/mnizam1/aip/FlowSDF/logs/%x-%A_%a.out

# Tuning stage 2: E3 (variance) and E3b (entropy) over gamma, with delta and lambda fixed from stage 1.
# 2k epochs from the E1 seed-0 checkpoint at 20k, then scored on the validation images.
set -e
: "${DELTA:?submit with: sbatch --export=ALL,DELTA=<best delta>,LAM=<best lambda> tune_e3.sbatch}"
: "${LAM:?submit with: sbatch --export=ALL,DELTA=<best delta>,LAM=<best lambda> tune_e3.sbatch}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /data/users2/mnizam1/miniforge3/bin/activate flowsdf
cd /home/users/mnizam1/aip/FlowSDF

WEIGHTINGS=(variance variance variance entropy entropy entropy)
CFGS=(cfg/monuseg_e3.yaml cfg/monuseg_e3.yaml cfg/monuseg_e3.yaml cfg/monuseg_e3b.yaml cfg/monuseg_e3b.yaml cfg/monuseg_e3b.yaml)
GAMMAS=(0.5 1 2 0.5 1 2)
i=$SLURM_ARRAY_TASK_ID
CFG=${CFGS[$i]}; NAME=tune_${WEIGHTINGS[$i]}_g${GAMMAS[$i]}
SETS="boundary.delta=$DELTA boundary.lam=$LAM boundary.gamma=${GAMMAS[$i]}"
INIT=$(ls runs/monuseg/*_monuseg_e1_train30_seed0/models_sanity/*0020000.pt)
echo "task $i: $NAME ($SETS)"

python main.py --config $CFG --seed 0 --init-ckpt "$INIT" --name $NAME --set learning.epochs=2000 $SETS
python eval_monuseg.py --config $CFG --split val \
    --raw-train-dir /data/users2/mnizam1/datasets/raw/monuseg_train \
    --ckpt runs/monuseg/*_${NAME}_seed0/models/*0001999.pt \
    --k 8 --amp --fg-thresh 0.05 --no-maps --out runs/eval/tune/$NAME
