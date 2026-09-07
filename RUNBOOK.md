# RUNBOOK — human steps (mayan), in order

Dates are ET. Deadline: **Wed Sept 16, 07:59 ET** (submit Tue Sept 15 evening).

## Sun 9/6 – Mon 9/7 (before code is ready)
1. OpenReview: make sure your profile exists and is complete (some venues require a profile older than a few days). Note the FMTS venue: `https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/FMTS` (confirm on fmts-workshop.github.io).
2. Accept the Llama-3.1 license on HF (optional; tier 3) and create an HF token: `hf auth login` on `della-gpu`.
3. On `della-gpu.princeton.edu` (login node has internet):
   ```bash
   echo 'export HF_HOME=/scratch/gpfs/$USER/.cache/huggingface' >> ~/.bashrc; source ~/.bashrc
   module load anaconda3/2026.7
   conda create -y --prefix /scratch/gpfs/$USER/envs/tbf-llm python=3.12
   conda activate /scratch/gpfs/$USER/envs/tbf-llm && pip install vllm==0.28.0 pyarrow pandas pyyaml tqdm
   conda create -y --prefix /scratch/gpfs/$USER/envs/tbf-tsfm python=3.11
   conda activate /scratch/gpfs/$USER/envs/tbf-tsfm && pip install -r requirements-tsfm.txt
   for m in Qwen/Qwen3-8B Qwen/Qwen3-4B Qwen/Qwen3-1.7B Qwen/Qwen3-0.6B amazon/chronos-2 google/timesfm-2.5-200m-pytorch NX-AI/TiRex; do hf download $m; done
   # tier 3 (later): Qwen/Qwen3-4B-Thinking-2507 Qwen/Qwen3-4B-Instruct-2507 google/gemma-3-4b-it meta-llama/Llama-3.1-8B-Instruct
   hf download ServiceNow/context-is-key --repo-type dataset
   ```
   If `pip install vllm` fights the cluster, use Apptainer: `apptainer pull docker://vllm/vllm-openai:v0.28.0` and set `TBF_APPTAINER=1` in the sbatch scripts (they support both).
4. Open the repo in Claude Code (`claude`) and say "Implement SPEC.md per CLAUDE.md; start with make test" — CLAUDE.md is read automatically. Wait for `make test` and `make smoke` green. Push to GitHub (private repo is fine; the paper links an anonymized copy later).

## Tue 9/8
5. On `della-gpu`: `git clone` the repo into `/scratch/gpfs/$USER/tbf`, `export TBF_CONTACT_EMAIL=<you>@princeton.edu`, then `make build-data` (runs on the login node, ~15–30 min, ~700 HTTP calls). Check `data/freshts26/BUILD.json`: ≥ 1,200 windows, ≥ 100 series. Run `pytest tests/test_leakage.py`.
6. `sbatch scripts/slurm/tsfm.sbatch` (1 GPU, < 1 h).
7. Go/no-go: `salloc --gres=gpu:1 --constraint=gpu80 --time=00:59:00` then
   `python -m tbf.run --config-id freshts26__qwen3-8b__direct__off__full__s0 --limit 50` and the same with `__b2048__`. Need valid rate ≥ 0.9 and sensible `thinking_tokens_used`. If not, fix prompts/parser with Claude Code before launching anything else.
8. `python scripts/expand_grid.py --tier 1 > configs/tier1.txt && sbatch --dependency=afterok:<tsfm_jobid> scripts/slurm/llm_array.sbatch configs/tier1.txt`.
9. Send Claude the go/no-go numbers and `BUILD.json`.

## Wed 9/9 – Fri 9/11
10. When tier 1 finishes: `sbatch scripts/slurm/summarize.sbatch`; commit `results/summary.csv`, `results/paired.csv`, `figures/`; push. Send Claude the repo link / files → paper drafting starts.
11. Launch tier 2 (`--tier 2`), then tier 3. Re-run summarize after each. Commit and push each time.
12. Check `squeue -u $USER` twice a day; failed array tasks are resumable — just resubmit the same config id.

## Sat 9/12 – Tue 9/15
13. Read the draft; check every number against `summary.csv`; run `make paper` (latexmk) in `paper/`; check page count with the final option.
14. Anonymize: no names, no non-anonymous repo link (use "code and dataset will be released"), no acknowledgements.
15. Submit on OpenReview by Tue 9/15 evening. Save the submission id and PDF to the Frontier Research project.
