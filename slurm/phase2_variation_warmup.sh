#!/bin/bash
# Phase 2: variation warmup sweep (var arm, seed=0, max_epoch=300, patience=200)
# Run AFTER phase1 to know the best lambda_cs.
# Usage: bash slurm/phase2_variation_warmup.sh --lambda_cs 0.5

set -e

BEST_LCS=0.5
while [[ $# -gt 0 ]]; do
    case $1 in
        --lambda_cs) BEST_LCS=$2; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SLURM_DIR")"
LOG_DIR="${ROOT}/slurm_logs"
mkdir -p "${LOG_DIR}"

submit_job() {
    local exp=$1
    local warmup=$2
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
    --seed 0 \\
    --base_lr 0.001 \\
    -w 0.1 \\
    --use_variation \\
    --lambda_cs ${BEST_LCS} \\
    --variation_warmup ${warmup} \\
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
    echo "Submitted: ${exp}  (warmup=${warmup}, lambda_cs=${BEST_LCS})"
}

submit_job "var_w0_s0"    0
submit_job "var_w30_s0"   30
submit_job "var_w50_s0"   50
submit_job "var_w100_s0"  100
