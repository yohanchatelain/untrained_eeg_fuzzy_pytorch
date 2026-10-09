#!/bin/bash
# Phase 2 on ds003478: ensembles of K instances (after slurm/phase1_ds003478.sh).
#   sbatch slurm/phase2_ds003478.sh            (from the repo root)
#SBATCH --account=def-glatard   # change to your allocation
#SBATCH --job-name=phase2-ds003478
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail

module load python/3.11
source "${VENV:-$HOME/venv/untrained_eeg}/bin/activate"
export OMP_NUM_THREADS=1  # parallel probes, one thread each
python scripts/phase2_ensembles.py --workers "$SLURM_CPUS_PER_TASK"
