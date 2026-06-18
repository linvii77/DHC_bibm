#!/bin/bash
# Phase 5 optimization experiments — 3-layer incremental ablation
# HPC1: zimuzhang2302  HPC2: yuangluo22  Local: RTX4090
#
# Experiment matrix:
#   5A: Layer1 only   (class balance, max_samples=200)          → s0 on HPC1
#   5B: Layer1+Layer2 (balance + variation fix K=2 tau=5)       → s0 on HPC2
#   5C: Layer1+2+3    (+ safe pseudo_proxy backbone detach)     → s0 local
#
# All use: lambda_cs=0.2, cps_w=0.1, base_lr=0.03, 300 epochs, patience=200

# ─── Phase 5A: class balancing only ────────────────────────────────────────
# submit on HPC1 as zimuzhang2302
# sbatch <<'SLURM'
# #!/bin/bash
# #SBATCH --job-name=p5a_s0
# #SBATCH --partition=gpua800
# #SBATCH --qos=4a800
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --gres=gpu:1
# #SBATCH --mem=40G
# #SBATCH --time=06:00:00
# #SBATCH --output=/gpfs/work/aac/zimuzhang2302/DHC_bibm/logs/p5a_balance_s0/slurm_%j.log
#
# cd /gpfs/work/aac/zimuzhang2302/DHC_bibm
# source /gpfs/work/aac/zimuzhang2302/miniconda/bin/activate dhc
#
# python code/train_dhc.py \
#     --task synapse --exp p5a_balance_s0 --seed 0 \
#     --base_lr 0.03 -w 0.1 -r \
#     --lambda_cs 0.2 \
#     --max_samples_per_class 200 \
#     --max_epoch 300 --patience 200 \
#     --embedding_dim 256 --num_variations 5 \
#     --split_unlabeled unlabeled_20p \
#     -g 0
# SLURM

# ─── Phase 5B: class balancing + variation fix (K=2, tau_var=5.0) ──────────
# submit on HPC2 as yuangluo22
# sbatch <<'SLURM'
# #!/bin/bash
# #SBATCH --job-name=p5b_s0
# #SBATCH --partition=gpua800
# #SBATCH --qos=4a800
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --gres=gpu:1
# #SBATCH --mem=40G
# #SBATCH --time=06:00:00
# #SBATCH --output=/gpfs/work/aac/yuangluo22/DHC_bibm/logs/p5b_varfix_s0/slurm_%j.log
#
# cd /gpfs/work/aac/yuangluo22/DHC_bibm
# source /gpfs/work/aac/yuangluo22/miniconda/bin/activate dhc
#
# python code/train_dhc.py \
#     --task synapse --exp p5b_varfix_s0 --seed 0 \
#     --base_lr 0.03 -w 0.1 -r \
#     --lambda_cs 0.2 \
#     --max_samples_per_class 200 \
#     --use_variation --num_variations 2 --tau_var 5.0 --variation_warmup 100 \
#     --max_epoch 300 --patience 200 \
#     --embedding_dim 256 \
#     --split_unlabeled unlabeled_20p \
#     -g 0
# SLURM

# ─── Phase 5C: Layer1+2+3 (add pseudo_proxy with backbone detach) ──────────
# run locally on RTX4090
# nohup python code/train_dhc.py \
#     --task synapse --exp p5c_pseudo_s0 --seed 0 \
#     --base_lr 0.03 -w 0.1 -r \
#     --lambda_cs 0.2 \
#     --max_samples_per_class 200 \
#     --use_variation --num_variations 2 --tau_var 5.0 --variation_warmup 100 \
#     --pseudo_proxy --pseudo_proxy_warmup 200 --pseudo_proxy_w 0.05 --pseudo_proxy_w_rampup 50 \
#     --max_epoch 300 --patience 200 \
#     --embedding_dim 256 \
#     --split_unlabeled unlabeled_20p \
#     -g 0 \
#     > /tmp/p5c_pseudo_s0.log 2>&1 &
