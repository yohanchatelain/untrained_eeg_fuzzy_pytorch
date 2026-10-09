#!/bin/bash
# Phase 1 on ds003478: untrained vs pretrained models, eyes closed/open and BDI.
#   bash scripts/00_download_pretrained.sh     (once, on the login node: needs internet)
#   sbatch slurm/phase1_ds003478.sh            (from the repo root)
# The first run also preprocesses the EEG (two inputs, cached in data/cache/).
# Resubmit if it times out: finished instances are skipped.
#SBATCH --account=def-glatard   # change to your allocation
#SBATCH --job-name=phase1-ds003478
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail

module load python/3.11
source "${VENV:-$HOME/venv/untrained_eeg}/bin/activate"
python scripts/phase1_features.py
python scripts/phase1_probe.py
