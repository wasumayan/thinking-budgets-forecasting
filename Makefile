PY ?= python
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: test smoke build-data tsfm summary figures tables paper clean-smoke

test:
	$(PY) -m pytest -q tests

smoke:
	$(PY) -m tbf.tsfm --dataset fixtures --model snaive --out results/smoke_snaive.jsonl
	$(PY) -m tbf.run --backend hf --model qwen3-0.6b --dataset fixtures --setup direct --think off --limit 4 --out results/smoke_direct_off.jsonl
	$(PY) -m tbf.run --backend hf --model qwen3-0.6b --dataset fixtures --setup direct --budget 64 --limit 4 --out results/smoke_direct_b64.jsonl
	$(PY) -m tbf.summarize --results-glob "results/smoke_*.jsonl" --dataset fixtures --out results/summary.csv --paired-out results/paired.csv
	$(PY) scripts/make_figures.py --summary results/summary.csv --paired results/paired.csv --out figures --smoke

build-data:
	$(PY) -m tbf.data.build_freshts26 --series configs/series.yaml --out data/freshts26
	$(PY) -m pytest -q tests/test_leakage.py

tsfm:
	$(PY) -m tbf.tsfm --dataset freshts26 --model snaive    --out results/freshts26__snaive.jsonl
	$(PY) -m tbf.tsfm --dataset freshts26 --model chronos2  --out results/freshts26__chronos2.jsonl
	$(PY) -m tbf.tsfm --dataset freshts26 --model timesfm25 --out results/freshts26__timesfm25.jsonl
	$(PY) -m tbf.tsfm --dataset freshts26 --model tirex     --out results/freshts26__tirex.jsonl

summary:
	$(PY) -m tbf.summarize --results-glob "results/*.jsonl" --out results/summary.csv --paired-out results/paired.csv

figures:
	$(PY) scripts/make_figures.py --summary results/summary.csv --paired results/paired.csv --out figures

tables:
	$(PY) scripts/make_tables.py --summary results/summary.csv --paired results/paired.csv --out paper/tables

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

clean-smoke:
	rm -f results/smoke_*.jsonl
