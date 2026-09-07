#!/bin/bash
# Run on the LOGIN node (della-gpu) — compute nodes have no internet. ~15-30 min.
set -euo pipefail
module purge
module load anaconda3/2026.7
conda activate /scratch/gpfs/$USER/envs/tbf-tsfm
export PYTHONPATH=src:${PYTHONPATH:-}
: "${TBF_CONTACT_EMAIL:?set TBF_CONTACT_EMAIL for the Wikimedia User-Agent}"
make build-data
python - <<'EOF'
import json; b=json.load(open("data/freshts26/BUILD.json")); print(json.dumps({k:b[k] for k in ("n_windows","n_series","per_domain","n_strict_2026","dropped") if k in b}, indent=1))
EOF
