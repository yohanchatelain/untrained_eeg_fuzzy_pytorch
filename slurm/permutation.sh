#!/bin/bash
# Permutation test (probe refit on shuffled labels) of every model on one dataset's tasks.
#   CONFIG=configs/ds004504.yaml sbatch slurm/permutation.sh     (after slurm/phase1.sh)
#SBATCH --account=def-glatard   # change to your allocation
#SBATCH --job-name=permutation
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail

module load python/3.11
source "${VENV:-$HOME/venv/untrained_eeg}/bin/activate"
export OMP_NUM_THREADS=1  # parallel permutations, one thread each
python scripts/permutation.py --config "${CONFIG:-configs/ds003478.yaml}"
