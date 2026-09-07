#!/bin/bash
# TSFM baselines + seasonal naive on FreshTS-26 (and CiK if present). ~30-60 min on one A100.
#SBATCH --job-name=tbf-tsfm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --output=logs/tsfm_%j.out

set -euo pipefail
module purge
module load anaconda3/2026.7
conda activate /scratch/gpfs/$USER/envs/tbf-tsfm
export HF_HOME=/scratch/gpfs/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs results
make tsfm
if [ -d data/cik ]; then
  python -m tbf.tsfm --dataset cik --model snaive   --out results/cik__snaive.jsonl
  python -m tbf.tsfm --dataset cik --model chronos2 --out results/cik__chronos2.jsonl
fi
