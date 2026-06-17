#!/bin/bash
# Phase 3: Full ablation (3 seeds × 3 arms) for the paper main table.
# Run AFTER phase1+2 to fill in BEST_LCS and BEST_WARMUP.
# Usage: bash slurm/phase3_full_ablation.sh --lambda_cs 0.5 --warmup 50

set -e

BEST_LCS=0.5
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

submit_baseline() {
    local seed=$1
    local exp="ablation_baseline_s${seed}"
    cat > /tmp/run_${exp}.sh << EOF
#!/bin/bash
#SBATCH --job-name=${exp}
#SBATCH --output=${LOG_DIR}/${exp}_%j.out
#SBATCH --error=${LOG_DIR}/${exp}_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --qos=4a800

cd ${ROOT}
source env.sh

python code/train_dhc.py \\
    --task synapse \\
    --exp ${exp} \\
    --seed ${seed} \\
    --base_lr 0.001 \\
    -w 0.1 \\
    --lambda_cs 0 \\
    --max_epoch 300 \\
    --patience 200 \\
    -g 0 -r

echo "=== TEST (best ckpt) ==="
python code/test.py --task synapse --exp ${exp} --cps AB --ckpt best_model --speed 1 -g 0

echo "=== EVALUATE ==="
python code/evaluate_our.py --exp ${exp} --cps AB
EOF
    sbatch /tmp/run_${exp}.sh
    echo "Submitted: ${exp}"
}

submit_no_var() {
    local seed=$1
    local exp="ablation_no_var_s${seed}"
    cat > /tmp/run_${exp}.sh << EOF
#!/bin/bash
#SBATCH --job-name=${exp}
#SBATCH --output=${LOG_DIR}/${exp}_%j.out
#SBATCH --error=${LOG_DIR}/${exp}_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --qos=4a800

cd ${ROOT}
source env.sh

python code/train_dhc.py \\
    --task synapse \\
    --exp ${exp} \\
    --seed ${seed} \\
    --base_lr 0.001 \\
    -w 0.1 \\
    --lambda_cs ${BEST_LCS} \\
    --max_epoch 300 \\
    --patience 200 \\
    --embedding_dim 256 \\
    --num_variations 5 \\
    -g 0 -r

echo "=== TEST (best ckpt) ==="
python code/test.py --task synapse --exp ${exp} --cps AB --ckpt best_model --speed 1 -g 0

echo "=== EVALUATE ==="
python code/evaluate_our.py --exp ${exp} --cps AB
EOF
    sbatch /tmp/run_${exp}.sh
    echo "Submitted: ${exp}"
}

submit_var() {
    local seed=$1
    local exp="ablation_var_s${seed}"
    cat > /tmp/run_${exp}.sh << EOF
#!/bin/bash
#SBATCH --job-name=${exp}
#SBATCH --output=${LOG_DIR}/${exp}_%j.out
#SBATCH --error=${LOG_DIR}/${exp}_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --qos=4a800

cd ${ROOT}
source env.sh

python code/train_dhc.py \\
    --task synapse \\
    --exp ${exp} \\
    --seed ${seed} \\
    --base_lr 0.001 \\
    -w 0.1 \\
    --use_variation \\
    --lambda_cs ${BEST_LCS} \\
    --variation_warmup ${BEST_WARMUP} \\
    --max_epoch 300 \\
    --patience 200 \\
    --embedding_dim 256 \\
    --num_variations 5 \\
    -g 0 -r

echo "=== TEST (best ckpt) ==="
python code/test.py --task synapse --exp ${exp} --cps AB --ckpt best_model --speed 1 -g 0

echo "=== EVALUATE ==="
python code/evaluate_our.py --exp ${exp} --cps AB
EOF
    sbatch /tmp/run_${exp}.sh
    echo "Submitted: ${exp}"
}

echo "=== Submitting Phase 3: Full Ablation ==="
echo "Config: lambda_cs=${BEST_LCS}, variation_warmup=${BEST_WARMUP}"
echo ""

for seed in 0 1 666; do
    submit_baseline $seed
    submit_no_var   $seed
    submit_var      $seed
done

echo ""
echo "9 jobs submitted. Collect results with:"
echo "  for exp in ablation_baseline ablation_no_var ablation_var; do"
echo "    for seed in 0 1 666; do"
echo "      grep 'MEAN' slurm_logs/\${exp}_s\${seed}_*.out"
echo "    done"
echo "  done"
