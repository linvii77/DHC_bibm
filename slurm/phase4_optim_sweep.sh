#!/bin/bash
# Phase 4: single-seed (seed=0) sweep for optimization v2
# Experiments: bg_only / pseudo_only / combined
# Submit: sbatch --export=RUN=v2_bg_s0 slurm/phase4_optim_sweep.sh
#         sbatch --export=RUN=v2_pseudo_s0 slurm/phase4_optim_sweep.sh
#         sbatch --export=RUN=v2_both_s0 slurm/phase4_optim_sweep.sh

#SBATCH --job-name=${RUN}
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1
#SBATCH --partition=gpua800 --qos=4a800
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%j.out

source /gpfs/work/aac/zimuzhang2302/DHC_bibm/slurm/hpc_header.sh
cd /gpfs/work/aac/zimuzhang2302/DHC_bibm

BASE="--task synapse --seed 0 --lambda_cs 0.2 --max_epoch 300 --patience 200 --base_lr 0.03 --split_unlabeled unlabeled_20p --embedding_dim 256 --num_variations 5 -g 0"

case "${RUN}" in
  v2_bg_s0)
    python code/train_dhc.py $BASE --exp v2_bg_s0 --proxy_ignore_bg
    ;;
  v2_pseudo_s0)
    python code/train_dhc.py $BASE --exp v2_pseudo_s0 \
      --pseudo_proxy --pseudo_proxy_conf 0.8 --pseudo_proxy_warmup 150
    ;;
  v2_both_s0)
    python code/train_dhc.py $BASE --exp v2_both_s0 \
      --proxy_ignore_bg \
      --pseudo_proxy --pseudo_proxy_conf 0.8 --pseudo_proxy_warmup 150
    ;;
  *)
    echo "Unknown RUN=${RUN}. Use: v2_bg_s0 / v2_pseudo_s0 / v2_both_s0"
    exit 1
    ;;
esac

# After training: evaluate best ckpt
EXP="${RUN}"
python code/test.py --task synapse --exp "${EXP}" --cps AB --ckpt best_model --speed 1
python code/evaluate_our.py --exp "${EXP}" --cps AB
