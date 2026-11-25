#!/bin/bash
#SBATCH --job-name=flowsdf-eval-test
#SBATCH --partition=qTRDGPU
#SBATCH -A psy53c17
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-6
#SBATCH --output=/home/users/mnizam1/aip/FlowSDF/logs/%x-%A_%a.out

# Test set (14 images), run ONCE after all choices were made on validation. Same settings as the
# validation evals; maps are saved, so the per-condition thresholds chosen on validation are applied
# afterwards with eval_thresholds.py (no re-sampling).
set -e
source /data/users2/mnizam1/miniforge3/bin/activate flowsdf
cd /home/users/mnizam1/aip/FlowSDF

case $SLURM_ARRAY_TASK_ID in
  0) N=e1;        CFG=cfg/monuseg_e1_train30.yaml;  CK="runs/monuseg/*_monuseg_e1_train30_seed*/models/*0029999.pt" ;;
  1) N=e1cont;    CFG=cfg/monuseg_e1cont.yaml;      CK="runs/monuseg/*_monuseg_e1cont_seed*/models/*0009999.pt" ;;
  2) N=e2;        CFG=cfg/monuseg_e2.yaml;          CK="runs/monuseg/*_final_e2_seed*/models/*0009999.pt" ;;
  3) N=e3;        CFG=cfg/monuseg_e3.yaml;          CK="runs/monuseg/*_final_e3_seed*/models/*0009999.pt" ;;
  4) N=e3b;       CFG=cfg/monuseg_e3b.yaml;         CK="runs/monuseg/*_final_e3b_seed*/models/*0009999.pt" ;;
  5) N=full37;    CFG=cfg/monuseg_full.yaml;        CK="runs/monuseg/*_monuseg_full_seed*/models/*0029999.pt" ;;  # paper reproduction, bf16
  6) N=full37_fp32; CFG=cfg/monuseg_full_fp32.yaml; CK="runs/monuseg/*_monuseg_full_fp32_seed*/models/*0029999.pt" ;;
esac
echo "test eval: $N"; ls $CK
python eval_monuseg.py --config $CFG --split test \
    --raw-test-dir /data/users2/mnizam1/datasets/raw/monuseg_test \
    --ckpt $CK --sampler official --out runs/eval/test/$N
