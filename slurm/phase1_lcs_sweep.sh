#!/bin/bash
# Phase 1: lambda_cs sweep (no_var arm, seed=0, max_epoch=300, patience=200)
# Submit all 4 jobs at once. Each is ~30 min on 1 GPU.
# Usage: bash slurm/phase1_lcs_sweep.sh

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
#SBATCH --qos=4a800

cd ${ROOT}
source env.sh

python code/train_dhc.py \\
    --task synapse \\
    --exp ${exp} \\
    --seed 0 \\
    --base_lr 0.001 \\
    -w 0.1 \\
    --lambda_cs ${lcs} \\
    --lambda_cs_rampup ${rampup} \\
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
    echo "Submitted: ${exp}  (lambda_cs=${lcs}, rampup=${rampup})"
}

# 4 runs: vary lcs {0.1, 0.5, 1.0} and rampup
submit_job "no_var_lcs01_s0"       0.1  0
submit_job "no_var_lcs05_s0"       0.5  0
submit_job "no_var_lcs10_s0"       1.0  0
submit_job "no_var_lcs05_ramp50_s0" 0.5 50
