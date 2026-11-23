#!/bin/bash
#SBATCH --job-name=flowsdf-full
#SBATCH --partition=qTRDGPU
#SBATCH -A psy53c17
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-12:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/users/mnizam1/aip/FlowSDF/logs/%x-%A_%a.out

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /data/users2/mnizam1/miniforge3/bin/activate flowsdf
cd /home/users/mnizam1/aip/FlowSDF

case $SLURM_ARRAY_TASK_ID in
  0) CFG=cfg/monuseg_full.yaml;      SEED=0 ;;   # bf16 baseline
  1) CFG=cfg/monuseg_full.yaml;      SEED=1 ;;   # bf16 baseline
  2) CFG=cfg/monuseg_full_fp32.yaml; SEED=0 ;;   # fp32 check
esac
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "config $CFG seed $SEED"
python main.py --config $CFG --seed $SEED
