#!/bin/bash
# Smoke test for p8b FusedProxy — 5 epochs, verify no crash
set -e
SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SLURM_DIR")"
LOG_DIR="${ROOT}/slurm_logs"
mkdir -p "${LOG_DIR}"

EXP="smoke_p8b"

cat > /tmp/run_${EXP}.sh << EOF
#!/bin/bash
#SBATCH --job-name=${EXP}
#SBATCH --output=${LOG_DIR}/${EXP}_%j.out
#SBATCH --error=${LOG_DIR}/${EXP}_%j.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=gpua800
#SBATCH --qos=4a800

cd ${ROOT}
source slurm/hpc_header.sh

python code/train_dhc.py \
    --task synapse \
    --exp ${EXP} \
    --seed 0 \
    --base_lr 0.03 \
    -w 0.1 -r \
    --lambda_cs 0.2 \
    --lambda_sac_cdba 0.1 \
    --num_variations 5 \
    --embedding_dim 256 \
    --fused_proxy_samples 8 \
    --fused_lambda_var 1.0 \
    --use_fused_proxy \
    --max_epoch 5 \
    --patience 200 \
    --split_unlabeled unlabeled_20p \
    -g 0

echo "=== SMOKE DONE: check loss_cs is nonzero above ==="
EOF

sbatch /tmp/run_${EXP}.sh
echo "Submitted smoke: ${EXP}"
