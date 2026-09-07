#!/bin/bash
# Usage: sbatch [--dependency=afterok:<tsfm_job>] scripts/slurm/llm_array.sbatch configs/tier1.txt
# configs/tierN.txt is produced by: python scripts/expand_grid.py --tier N --with-time > configs/tierN.txt
# Each line: "<config_id>\t<HH:MM:SS>". The array index selects the line. Wall-time is the max over the file
# (set below); per-config time in the file is informational. Keep <= 24h for the gpu-short QOS.
#SBATCH --job-name=tbf-llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --array=0-29%8
#SBATCH --output=logs/llm_%A_%a.out
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=${USER}@princeton.edu

set -euo pipefail
LIST=${1:?config list file}
CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST" | cut -f1)
[ -z "$CFG" ] && { echo "no config at index $SLURM_ARRAY_TASK_ID"; exit 0; }
echo "config: $CFG"

module purge
module load anaconda3/2026.7
export HF_HOME=/scratch/gpfs/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs results

if [ "${TBF_APPTAINER:-0}" = "1" ]; then
  apptainer exec --nv --bind /scratch/gpfs/$USER:/scratch/gpfs/$USER vllm-openai_v0.28.0.sif \
    python -m tbf.run --config-id "$CFG"
else
  conda activate /scratch/gpfs/$USER/envs/tbf-llm
  python -m tbf.run --config-id "$CFG"
fi
