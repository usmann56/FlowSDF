#!/bin/bash
#SBATCH --job-name=flowsdf-bench
#SBATCH --partition=qTRDGPU
#SBATCH -A psy53c17
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/home/users/mnizam1/aip/FlowSDF/logs/%x-%j.out

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /data/users2/mnizam1/miniforge3/bin/activate flowsdf
cd /home/users/mnizam1/aip/FlowSDF
nvidia-smi --query-gpu=name,memory.total --format=csv

python main.py --config cfg/bench_fp32_ckpt.yaml   --seed 10
python main.py --config cfg/bench_bf16_ckpt.yaml   --seed 11
python main.py --config cfg/bench_bf16_nockpt.yaml --seed 12   # may run out of memory; that's what we're testing

for s in 10 11 12; do
  f=$(ls -d runs/monuseg/*_seed$s | tail -1)
  echo "seed$s: $(grep -c 'Loss:' $f/simulate.log) epochs done, \
avg $(grep 'Loss:' $f/simulate.log | tail -n +6 | awk -F'Time: ' '{gsub("s","",$2); s+=$2; n++} END {printf "%.2f", s/n}') s/epoch, \
last loss $(grep 'Loss:' $f/simulate.log | tail -1 | awk '{print $4}')"
  grep -q OutOfMemoryError $f/simulate.log && echo "  -> out of memory"
done
