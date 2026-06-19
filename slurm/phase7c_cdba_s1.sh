#!/bin/bash
# Phase 7c: CDBA structural fixes – seed=1
# Usage: bash slurm/phase7c_cdba_s1.sh
# HPC: zimuzhang2302@login.hpc.xjtlu.edu.cn, partition=gpua800, qos=4a800

set -e
SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SLURM_DIR")"
LOG_DIR="${ROOT}/slurm_logs"
mkdir -p "${LOG_DIR}"

EXP="p7c_cdba_s1"

cat > /tmp/run_${EXP}.sh << EOF
#!/bin/bash
#SBATCH --job-name=${EXP}
#SBATCH --output=${LOG_DIR}/${EXP}_%j.out
#SBATCH --error=${LOG_DIR}/${EXP}_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=gpua800
#SBATCH --qos=4a800

cd ${ROOT}
source slurm/hpc_header.sh

python code/train_dhc.py \\
    --task synapse \\
    --exp ${EXP} \\
    --seed 1 \\
    --base_lr 0.03 \\
    -w 0.1 -r \\
    --lambda_cs 0.2 \\
    --use_cdba \\
    --num_variations 5 \\
    --variation_warmup 100 \\
    --num_proxy_samples 4 \\
    --tau_var_cdba 5.0 \\
    --cdba_unlabeled_warmup 150 \\
    --lambda_sac_cdba 0.1 \\
    --max_samples_per_class 200 \\
    --max_epoch 300 \\
    --patience 200 \\
    --embedding_dim 256 \\
    --split_unlabeled unlabeled_20p \\
    -g 0

echo "=== TEST (best ckpt) ==="
python code/test.py --task synapse --exp ${EXP} --cps AB --ckpt best_model --speed 1 -g 0

echo "=== EVALUATE ==="
python code/evaluate_our.py --exp ${EXP} --cps AB
EOF

sbatch /tmp/run_${EXP}.sh
echo "Submitted: ${EXP}"
