#!/bin/bash
#SBATCH --job-name=mcl-diffusion
#SBATCH --account=m25146
#SBATCH --partition=mesonet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_pipeline.out
#SBATCH --error=logs/%j_pipeline.err

set -e

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:   $(date)"
echo "========================================"

cd $SLURM_SUBMIT_DIR
source venv/bin/activate
nvidia-smi

# ---- Phase 1: Baseline ----
echo ""
echo "========================================"
echo "PHASE 1: Training Baseline (single network)"
echo "========================================"
python3 run.py baseline train --epochs 50 --batch_size 512 --lr 2e-4

# ---- Phase 2: MCL Ensemble ----
echo ""
echo "========================================"
echo "PHASE 2: Training MCL Ensemble (K=5)"
echo "========================================"
python3 run.py mcl train --K 5 --epochs 50 --batch_size 512 --lr 2e-4

# ---- Phase 3: Gating Network ----
echo ""
echo "========================================"
echo "PHASE 3: Training Gating Network"
echo "========================================"
python3 run.py routing train_gating \
    --ensemble_checkpoint mcl_diffusion.pt \
    --epochs 20 --batch_size 1024 --lr 1e-3

# ---- Phase 4: Full Evaluation ----
echo ""
echo "========================================"
echo "PHASE 4: Full Evaluation + Figures"
echo "========================================"
python3 run.py eval full \
    --ensemble_checkpoint mcl_diffusion.pt \
    --gating_checkpoint gating_network.pt \
    --n_samples 50000 \
    --output_dir figures

echo ""
echo "========================================"
echo "DONE — $(date)"
echo "========================================"
