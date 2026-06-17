#!/bin/bash
#SBATCH --job-name=test_baseline_s1
#SBATCH --partition=gpua800
#SBATCH --gres=gpu:a800:1
#SBATCH --qos=4a800
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/gpfs/work/aac/yuangluo22/DHC_bibm/slurm_logs/test_baseline_s1_%j.out
#SBATCH --error=/gpfs/work/aac/yuangluo22/DHC_bibm/slurm_logs/test_baseline_s1_%j.err

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/miniconda3-22.11.1-l4fo6takdpx5xewhp463xsqr4jcd73dx/etc/profile.d/conda.sh
conda activate vapl

cd /gpfs/work/aac/yuangluo22/DHC_bibm
echo "Job start: $(date)"
echo "Node: $(hostname)"
nvidia-smi -L

echo "=== INFERENCE ==="
PYTHONPATH=code python code/test.py \
  --task synapse \
  --exp baseline_s1 \
  --cps AB \
  --speed 1 \
  -g 0

echo "=== EVALUATE ==="
PYTHONPATH=code python code/evaluate_our.py \
  --task synapse \
  --exp baseline_s1 \
  --cps AB

echo "Job end: $(date)"
