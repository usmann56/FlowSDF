#!/bin/bash
#SBATCH --job-name=flowsdf-smoke
#SBATCH --partition=qTRDGPU
#SBATCH -A psy53c17
#SBATCH --gres=gpu:A40:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH -c 16
#SBATCH --time=02:00:00
#SBATCH --output=/home/users/mnizam1/aip/FlowSDF/logs/%x-%j.out

source /data/users2/mnizam1/miniforge3/bin/activate flowsdf
cd /home/users/mnizam1/aip/FlowSDF

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python main.py --config cfg/monuseg_smoke.yaml
python sampler.py --config cfg/monuseg_smoke.yaml
