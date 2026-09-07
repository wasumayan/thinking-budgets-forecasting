#!/bin/bash
# CPU-only: results/*.jsonl -> summary.csv, paired.csv, figures, tables.
#SBATCH --job-name=tbf-summ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/summ_%j.out

set -euo pipefail
module purge
module load anaconda3/2026.7
conda activate /scratch/gpfs/$USER/envs/tbf-tsfm
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs figures paper/tables
make summary figures tables
git add results/summary.csv results/paired.csv figures paper/tables && git commit -m "results: summary $(date -u +%F_%H%M)" || true
