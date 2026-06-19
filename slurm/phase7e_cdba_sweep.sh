#!/bin/bash
# Phase 7e: CDBA unlabeled-weight sweep – seed=0
# Usage (local): RUN=p7e_w200_v0.5_s0 bash slurm/phase7e_cdba_sweep.sh
# Usage (HPC):   sbatch --export=RUN=p7e_w200_v0.5_s0 slurm/phase7e_cdba_sweep.sh
# RUN options:
#   p7e_w200_v0.5_s0   warmup=200, unl_weight=0.5, rampup=0  (later warmup only)
#   p7e_w150_v0.2_s0   warmup=150, unl_weight=0.2, rampup=0  (smaller weight only)
#   p7e_w200_v0.2_s0   warmup=200, unl_weight=0.2, rampup=0  (both)
#   p7e_w150_ramp50_s0 warmup=150, unl_weight=0.5, rampup=50 (gradual ramp)

set -e
SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SLURM_DIR")"
LOG_DIR="${ROOT}/slurm_logs"
mkdir -p "${LOG_DIR}"

EXP="${RUN}"

case "${RUN}" in
  p7e_w200_v0.5_s0)   UNL_WARMUP=200; UNL_W=0.5; UNL_RAMPUP=0 ;;
  p7e_w150_v0.2_s0)   UNL_WARMUP=150; UNL_W=0.2; UNL_RAMPUP=0 ;;
  p7e_w200_v0.2_s0)   UNL_WARMUP=200; UNL_W=0.2; UNL_RAMPUP=0 ;;
  p7e_w150_ramp50_s0) UNL_WARMUP=150; UNL_W=0.5; UNL_RAMPUP=50 ;;
  *) echo "Unknown RUN: ${RUN}"; exit 1 ;;
esac

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
    --seed 0 \\
    --base_lr 0.03 \\
    -w 0.1 -r \\
    --lambda_cs 0.2 \\
    --use_cdba \\
    --num_variations 5 \\
    --variation_warmup 100 \\
    --num_proxy_samples 4 \\
    --tau_var_cdba 5.0 \\
    --cdba_unlabeled_warmup ${UNL_WARMUP} \\
    --cdba_unlabeled_weight ${UNL_W} \\
    --cdba_unlabeled_weight_rampup ${UNL_RAMPUP} \\
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

if [ -n "${SLURM_JOB_ID}" ]; then
    bash /tmp/run_${EXP}.sh
else
    sbatch /tmp/run_${EXP}.sh
    echo "Submitted: ${EXP}"
fi
