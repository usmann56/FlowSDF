#!/bin/bash
#SBATCH --job-name=flowsdf-tune-e2
#SBATCH --partition=qTRDGPU
#SBATCH -A psy53c17
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=/home/users/mnizam1/aip/FlowSDF/logs/%x-%A_%a.out

# Tuning stage 1: E2 (uniform boundary BCE) over delta x lambda, plus an E1-cont reference (task 9).
# 2k epochs from the E1 seed-0 checkpoint at 20k, then scored on the validation images.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /data/users2/mnizam1/miniforge3/bin/activate flowsdf
cd /home/users/mnizam1/aip/FlowSDF

DELTAS=(3 3 3 5 5 5 7 7 7)
LAMS=(0.05 0.1 0.2 0.05 0.1 0.2 0.05 0.1 0.2)
i=$SLURM_ARRAY_TASK_ID
if [ $i -lt 9 ]; then
  CFG=cfg/monuseg_e2.yaml; NAME=tune_e2_d${DELTAS[$i]}_l${LAMS[$i]}
  SETS="boundary.delta=${DELTAS[$i]} boundary.lam=${LAMS[$i]}"
else
  CFG=cfg/monuseg_e1cont.yaml; NAME=tune_ref_e1cont; SETS=""
fi
INIT=$(ls runs/monuseg/*_monuseg_e1_train30_seed0/models_sanity/*0020000.pt)
echo "task $i: $NAME ($SETS)"

python main.py --config $CFG --seed 0 --init-ckpt "$INIT" --name $NAME --set learning.epochs=2000 $SETS
python eval_monuseg.py --config $CFG --split val \
    --raw-train-dir /data/users2/mnizam1/datasets/raw/monuseg_train \
    --ckpt runs/monuseg/*_${NAME}_seed0/models/*0001999.pt \
    --k 8 --amp --fg-thresh 0.05 --no-maps --out runs/eval/tune/$NAME
