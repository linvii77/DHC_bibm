#!/bin/bash
# HPC environment activation helper (zimuzhang2302@login.hpc.xjtlu.edu.cn)
# Source this at the top of each SLURM job script.

source /etc/profile.d/modules.sh
module load miniconda3/22.11.1-gcc-8.5.0-l4fo6ta

# conda activate doesn't work in non-interactive SLURM shells; use eval approach
eval "$(conda shell.bash hook)"
conda activate dhc

# Verify activation
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"

# Set PYTHONPATH relative to this script's location (job runs from DHC_bibm/)
export PYTHONPATH="$(pwd)/code:$PYTHONPATH"
