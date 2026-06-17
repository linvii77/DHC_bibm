#!/bin/bash
# Phase 1: lambda_cs sweep (no_var arm, seed=0, max_epoch=300, patience=200)
# FIXED: base_lr=0.03 (30x larger, matching original DHC config)
# Usage: bash slurm/phase1_lcs_sweep.sh
# HPC: zimuzhang2302@login.hpc.xjtlu.edu.cn, partition=gpu4090, qos=4a800

set -e
SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SLURM_DIR")"
LOG_DIR="${ROOT}/slurm_logs"
mkdir -p "${LOG_DIR}"

submit_job() {
    local exp=$1
    local lcs=$2
    local rampup=$3
    cat > /tmp/run_${exp}.sh << EOF
#!/bin/bash
#SBATCH --job-name=${exp}
#SBATCH --output=${LOG_DIR}/${exp}_%j.out
#SBATCH --error=${LOG_DIR}/${exp}_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=gpua800
#SBATCH --qos=4a800

cd ${ROOT}
source slurm/hpc_header.sh

python code/train_dhc.py \\
    --task synapse \\
    --exp ${exp} \\
    --seed 0 \\
    --base_lr 0.03 \\
    -w 0.1 \\
    --lambda_cs ${lcs} \\
    --lambda_cs_rampup ${rampup} \\
    --max_epoch 300 \\
    --patience 200 \\
    --embedding_dim 256 \\
    --num_variations 5 \\
    --split_unlabeled unlabeled_20p \\
    -g 0 -r

echo "=== TEST (best ckpt) ==="
python code/test.py --task synapse --exp ${exp} --cps AB --ckpt best_model --speed 1 -g 0

echo "=== EVALUATE ==="
python code/evaluate_our.py --exp ${exp} --cps AB
EOF
    sbatch /tmp/run_${exp}.sh
    echo "Submitted: ${exp}  (lambda_cs=${lcs}, rampup=${rampup})"
}

submit_baseline() {
    local exp="hpc_baseline_s0"
    cat > /tmp/run_${exp}.sh << EOF
#!/bin/bash
#SBATCH --job-name=${exp}
#SBATCH --output=${LOG_DIR}/${exp}_%j.out
#SBATCH --error=${LOG_DIR}/${exp}_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=gpua800
#SBATCH --qos=4a800

cd ${ROOT}
source slurm/hpc_header.sh

python code/train_dhc.py \\
    --task synapse \\
    --exp ${exp} \\
    --seed 0 \\
    --base_lr 0.03 \\
    -w 0.1 \\
    --lambda_cs 0 \\
    --max_epoch 300 \\
    --patience 200 \\
    --split_unlabeled unlabeled_20p \\
    -g 0 -r

echo "=== TEST (best ckpt) ==="
python code/test.py --task synapse --exp ${exp} --cps AB --ckpt best_model --speed 1 -g 0

echo "=== EVALUATE ==="
python code/evaluate_our.py --exp ${exp} --cps AB
EOF
    sbatch /tmp/run_${exp}.sh
    echo "Submitted: ${exp}  (baseline, no proxy)"
}

echo "=== Phase 1: lambda_cs sweep (base_lr=0.03) ==="
submit_baseline
submit_job "hpc_no_var_lcs01_s0"        0.1  0
submit_job "hpc_no_var_lcs02_s0"        0.2  0
submit_job "hpc_no_var_lcs05_s0"        0.5  0
submit_job "hpc_no_var_lcs05_ramp100_s0" 0.5 100
echo "5 jobs submitted."
