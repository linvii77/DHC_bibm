#!/bin/bash
# Phase 3: Full ablation (3 seeds × 3 arms) for the paper main table.
# FIXED: base_lr=0.03, split_unlabeled=unlabeled_20p
# Run AFTER phase1+2 to fill in BEST_LCS and BEST_WARMUP.
# Usage: bash slurm/phase3_full_ablation.sh --lambda_cs 0.2 --warmup 50
# HPC: zimuzhang2302@login.hpc.xjtlu.edu.cn, partition=gpu4090, qos=4a800

set -e

BEST_LCS=0.2
BEST_WARMUP=50
while [[ $# -gt 0 ]]; do
    case $1 in
        --lambda_cs) BEST_LCS=$2; shift 2;;
        --warmup)    BEST_WARMUP=$2; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SLURM_DIR")"
LOG_DIR="${ROOT}/slurm_logs"
mkdir -p "${LOG_DIR}"

make_sbatch_header() {
    local exp=$1
    cat << EOF
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
EOF
}

submit_baseline() {
    local seed=$1
    local exp="ablation_baseline_s${seed}"
    {
        make_sbatch_header ${exp}
        cat << EOF
python code/train_dhc.py \\
    --task synapse --exp ${exp} --seed ${seed} \\
    --base_lr 0.03 -w 0.1 --lambda_cs 0 \\
    --max_epoch 300 --patience 200 \\
    --split_unlabeled unlabeled_20p \\
    -g 0 -r

echo "=== TEST ===" && python code/test.py --task synapse --exp ${exp} --cps AB --ckpt best_model --speed 1 -g 0
echo "=== EVAL ===" && python code/evaluate_our.py --exp ${exp} --cps AB
EOF
    } > /tmp/run_${exp}.sh
    sbatch /tmp/run_${exp}.sh && echo "Submitted: ${exp}"
}

submit_no_var() {
    local seed=$1
    local exp="ablation_no_var_s${seed}"
    {
        make_sbatch_header ${exp}
        cat << EOF
python code/train_dhc.py \\
    --task synapse --exp ${exp} --seed ${seed} \\
    --base_lr 0.03 -w 0.1 --lambda_cs ${BEST_LCS} \\
    --max_epoch 300 --patience 200 \\
    --embedding_dim 256 --num_variations 5 \\
    --split_unlabeled unlabeled_20p \\
    -g 0 -r

echo "=== TEST ===" && python code/test.py --task synapse --exp ${exp} --cps AB --ckpt best_model --speed 1 -g 0
echo "=== EVAL ===" && python code/evaluate_our.py --exp ${exp} --cps AB
EOF
    } > /tmp/run_${exp}.sh
    sbatch /tmp/run_${exp}.sh && echo "Submitted: ${exp}"
}

submit_var() {
    local seed=$1
    local exp="ablation_var_s${seed}"
    {
        make_sbatch_header ${exp}
        cat << EOF
python code/train_dhc.py \\
    --task synapse --exp ${exp} --seed ${seed} \\
    --base_lr 0.03 -w 0.1 --use_variation --lambda_cs ${BEST_LCS} \\
    --variation_warmup ${BEST_WARMUP} \\
    --max_epoch 300 --patience 200 \\
    --embedding_dim 256 --num_variations 5 \\
    --split_unlabeled unlabeled_20p \\
    -g 0 -r

echo "=== TEST ===" && python code/test.py --task synapse --exp ${exp} --cps AB --ckpt best_model --speed 1 -g 0
echo "=== EVAL ===" && python code/evaluate_our.py --exp ${exp} --cps AB
EOF
    } > /tmp/run_${exp}.sh
    sbatch /tmp/run_${exp}.sh && echo "Submitted: ${exp}"
}

echo "=== Phase 3: Full Ablation (base_lr=0.03) ==="
echo "Config: lambda_cs=${BEST_LCS}, variation_warmup=${BEST_WARMUP}"
for seed in 0 1 666; do
    submit_baseline $seed
    submit_no_var   $seed
    submit_var      $seed
done
echo "9 jobs submitted."
