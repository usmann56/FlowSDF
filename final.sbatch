#!/bin/bash
#SBATCH --job-name=flowsdf-final
#SBATCH --partition=qTRDGPU
#SBATCH -A psy53c17
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/users/mnizam1/aip/FlowSDF/logs/%x-%A_%a.out

# Final runs: E2 (uniform), E3 (variance), E3b (entropy) x seeds 0,1. Same budget as E1-cont:
# 10k epochs from each seed's E1 (30 images) checkpoint at 20k. delta/lambda from tuning stage 1,
# gamma from stage 2 (separately for variance and entropy).
set -e
: "${DELTA:?submit with: sbatch --export=ALL,DELTA=..,LAM=..,GAMMA_VAR=..,GAMMA_ENT=.. final.sbatch}"
: "${LAM:?missing LAM}"; : "${GAMMA_VAR:?missing GAMMA_VAR}"; : "${GAMMA_ENT:?missing GAMMA_ENT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /data/users2/mnizam1/miniforge3/bin/activate flowsdf
cd /home/users/mnizam1/aip/FlowSDF

CONDS=(e2 e3 e3b)
GAMMAS=(1 $GAMMA_VAR $GAMMA_ENT)  # gamma is unused by e2 (uniform weights)
i=$SLURM_ARRAY_TASK_ID
C=$((i / 2)); SEED=$((i % 2))
COND=${CONDS[$C]}; NAME=final_$COND
SETS="boundary.delta=$DELTA boundary.lam=$LAM boundary.gamma=${GAMMAS[$C]}"
INIT=$(ls runs/monuseg/*_monuseg_e1_train30_seed${SEED}/models_sanity/*0020000.pt)
echo "task $i: $NAME seed $SEED ($SETS), init $INIT"

python main.py --config cfg/monuseg_$COND.yaml --seed $SEED --init-ckpt "$INIT" --name $NAME --set $SETS
